# docker run -it --rm --gpus '"device=7"' -v $(pwd):/app -v $(pwd)/../cache:/app/cache --name ${USER}_lcc ${USER}_lc python -m scripts.rl_test

# tokens are 0: pad, 1, 2: actions, 3 -> 258: observations
import jax
import sys
import os
sys.path.append(os.path.abspath('/home/duser/AlphaTrade')) 
import jax.numpy as jnp
#import flax.linen as nn
import datetime
import numpy as np
import optax
import time

from typing import Sequence, NamedTuple, Any, Dict, Callable, Optional

import distrax
import gymnax
import functools
from gymnax.environments import spaces
from gymnax_exchange.jaxrl.utils import FlattenObservationWrapper, LogWrapper
from jax._src import dtypes
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv 
#import flax
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
#Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks", False) #finds a whole assortment of leaks if true... bizarre.
import datetime
import gymnax_exchange.utils.colorednoise as cnoise
jax.numpy.set_printoptions(linewidth=250)
import dataclasses
import jax
import jax.numpy as jnp
#import optax
import distrax


from jax_rwkv.src.auto import get_rand_model
from gymnax_exchange.jaxrl.rl_processing import get_ppo_agent, calculate_gae, get_jit_ppo, PAD_FLAG, OBS_FLAG, ACT_FLAG
#from utils.jstring import JString

j_calculate_gae = jax.jit(jax.vmap(calculate_gae, in_axes=(0, 0, 0, 0, 0, None, None)))

config = {
    "LR": 1e-3,
    "NUM_ENVS": 400,
    "NUM_STEPS": 10,#128,
    "TOTAL_TIMESTEPS": 5e6,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 1,
    "GAMMA": 0.99 ** (1/5),
    "GAE_LAMBDA": 0.95 ** (1/5),
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu",
    "ANNEAL_LR": True,
    "DEBUG": True,
    "WANDB": True,

    "TASKSIDE": "random", # "random", "buy", "sell"
        "REWARD_LAMBDA": 0.001, #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "WINDOW_INDEX": -1, # 2 fix random episode #-1,
        "MAX_TASK_SIZE": 100,
        "EPISODE_TIME": 60 * 5, # time in seconds
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "ATFOLDER": "/home/duser/AlphaTrade/training_oneDay"
}

config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)


jit_ppo_update = get_jit_ppo(config)

def handle_continuous(observation):
    return jnp.array(observation).astype(jnp.float8_e4m3b11fnuz).view(jnp.uint8).astype(jnp.int32)

rng = jax.random.key(2)

env = MarketMakingEnv(
        alphatradePath=config["ATFOLDER"],
        #task=config["TASKSIDE"],
        window_index=config["WINDOW_INDEX"],
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        rewardLambda=config["REWARD_LAMBDA"],
       ep_type=config["DATA_TYPE"],
    )
env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=config["REWARD_LAMBDA"],
        episode_time=config["EPISODE_TIME"],
    )
env = FlattenObservationWrapper(env)
env = LogWrapper(env)

num_tokens = 1 + env.action_space(env_params).n + 256
config["MIN_ACTION_TOK"] = 1
config["MAX_ACTION_TOK"] = 8

RWKV, params = get_rand_model(0, "6", 3, 256, num_tokens, dtype=jnp.float32, rwkv_type="ScanRWKV")
forward, params = get_ppo_agent(RWKV, params, seed=1)
v_forward_jit = jax.jit(jax.vmap(forward, in_axes=(0, 0, None, 0)))
init_state = RWKV.default_state(params)
if isinstance(init_state, tuple):
    init_state = tuple([jnp.repeat(s[None], config["NUM_ENVS"], axis=0) for s in init_state])
else:
    init_state = jnp.repeat(init_state[None], config["NUM_ENVS"], axis=0)
state = init_state

def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac

solver = optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(linear_schedule, eps=1e-5)
)
optimizer = solver.init(params)

rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

v_env_step = jax.jit(jax.vmap(
    env.step, in_axes=(0, 0, 0, None)
))

global_timestep = 1

for _ in range(int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]):
    initial_state = state
    tokens_list = []
    flags_list = []
    values_list = []
    rewards_list = []
    log_prob_list = []
    dones_list = []

    update_returns = []
    for t in range(config["NUM_STEPS"]):
        rng, _rng = jax.random.split(rng)
        tokenized = handle_continuous(obsv)
        pi, value, state = v_forward_jit(tokenized, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32) * tokenized.shape[-1])
        pi = distrax.Categorical(logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        _, value1, state = v_forward_jit(action, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = v_env_step(rng_step, env_state, action, env_params)

        state = jax.vmap(jax.lax.select)(done, init_state, state)
        
        tokens_list.append(tokenized)
        tokens_list.append(action[:, None] + config["MIN_ACTION_TOK"])
        
        flags_list.append(jnp.ones_like(tokenized) * OBS_FLAG)
        flags_list.append(jnp.ones_like(tokenized)[:, :1] * ACT_FLAG)

        values_list.append(value)
        values_list.append(value1)

        rewards_list.append(jnp.zeros(shape=tokenized.shape))
        rewards_list.append(reward[:, None])

        log_prob_list.append(jnp.zeros_like(value))
        log_prob_list.append(log_prob[:, None])

        dones_list.append(jnp.zeros(value.shape, dtype=jnp.bool))
        dones_list.append(done[:, None])

        return_values = info["returned_episode_returns"][info["returned_episode"]]
        # print(return_values.shape)
        for r in return_values:
            # print(global_timestep, ":", r)
            update_returns.append(r)
        global_timestep += 1

    tokens_list = jnp.concatenate(tokens_list, axis=1)
    flags_list = jnp.concatenate(flags_list, axis=1)
    values_list = jnp.concatenate(values_list, axis=1)
    rewards_list = jnp.concatenate(rewards_list, axis=1)
    log_probs_list = jnp.concatenate(log_prob_list, axis=1)[..., 1:]
    dones_list = jnp.concatenate(dones_list, axis=1)
    #buf = JString(tokens_list, jnp.ones_like(tokens_list[:, 0]) * tokens_list.shape[1])
    buf = {
    "tokens": jnp.array(tokens_list),
    "length": jnp.ones_like(tokens_list[:, 0]) * tokens_list.shape[1],}


    dones_list = jnp.cumsum(dones_list, axis=1, dtype=jnp.bool)
    flags_list = jnp.where(jnp.concatenate((dones_list[:, :1], dones_list[:, :-1]), axis=1), PAD_FLAG, flags_list)
    # print(dones_list)
    # print(flags_list)
    
    # print(tokens_list.shape, flags_list.shape, values_list.shape, rewards_list.shape, log_prob_list.shape)

    _, last_value, _ = v_forward_jit(handle_continuous(obsv), state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))
    
    advantages, targets = j_calculate_gae(flags_list, dones_list, values_list, rewards_list, last_value[..., -1], config["GAMMA"], config["GAE_LAMBDA"])
    # print("value", values_list)
    # print("target", targets)
    print("UPDATING")
    if len(update_returns) > 0:
        print("avg returns:", sum(update_returns) / len(update_returns))
    else:
        print("None ended")

    for _ in range(config["UPDATE_EPOCHS"]):
        params, optimizer, (loss, value_loss, loss_actor, entropy, state) = jit_ppo_update(solver, v_forward_jit, params, optimizer, buf, flags_list, values_list, log_probs_list, advantages, targets, initial_state)
        print(loss, value_loss, loss_actor, entropy)

    state = jax.vmap(jax.lax.select)(dones_list[:, -1], init_state, state)