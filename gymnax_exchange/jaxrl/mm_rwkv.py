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
from dataclasses import dataclass

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

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

import wandb


wandbOn = True # False
if wandbOn:
    import wandb

@jax.tree_util.register_pytree_node_class
@dataclass
class JString:
    tokens: jnp.ndarray
    length: jnp.ndarray

    def __init__(self, tokens, length=None):
        self.tokens = jnp.array(tokens)
        self.length = (
            length if length is not None
            else jnp.ones_like(tokens[:, 0]) * tokens.shape[1]
        )

    def tree_flatten(self):
        # The children are the arrays that JAX can trace.
        children = (self.tokens, self.length)
        # No auxiliary static data.
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tokens, length = children
        return cls(tokens, length)

##Get the AT folder
try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
except:
        # ATFolder = "./testing_oneDay"
        #ATFolder = "/training_oneDay"
        ATFolder = "/home/duser/AlphaTrade/training_oneDay"

config = {
    "LR": 1e-3,
    "NUM_ENVS": 24,
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
        "REWARD_LAMBDA": 0.1, #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "WINDOW_INDEX": 2, # 2 fix random episode #-1,
        "MAX_TASK_SIZE": 100,
        "EPISODE_TIME": 60*60,  # 
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "ATFOLDER": ATFolder
}


config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)


jit_ppo_update = get_jit_ppo(config)

def handle_continuous(observation):
    return jnp.array(observation).astype(jnp.float8_e4m3b11fnuz).view(jnp.uint8).astype(jnp.int32)

rng = jax.random.key(0)
rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

env = MarketMakingEnv(
        key_reset,
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
if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_rwkv_Train",
            config=config,
            save_code=True,  # 
        )
if wandbOn:
    def log_all_metrics(action, info, global_timestep):
        """JAX-compatible callback for all metrics logging"""
        # Environment metrics
        metrics = {
            "global_step": int(jnp.max(info["timesteps"]) if info["timesteps"].size > 0 else 0),
            "reward": float(jnp.mean(info["reward"]) if info["reward"].size > 0 else 0),
            "PnL": float(jnp.mean(info["PnL"]) if info["PnL"].size > 0 else 0),
            "inventory": float(jnp.mean(info["inventories"]) if info["inventories"].size > 0 else 0),
            "buyQuant": float(jnp.mean(info["buyQuant"]) if info["buyQuant"].size > 0 else 0),
            "sellQuant": float(jnp.mean(info["sellQuant"]) if info["sellQuant"].size > 0 else 0),
            "other_exec_quants": float(jnp.mean(info["other_exec_quants"]) if info["other_exec_quants"].size > 0 else 0),
        }

        # Action distribution
        unique_actions, counts = jnp.unique(action, return_counts=True)
        for a, c in zip(unique_actions, counts):
            metrics[f"actions/{int(a)}"] = int(c)

        # Episode returns
        if "returned_episode_returns" in info:
            returns = info["returned_episode_returns"]
            metrics.update({
                "returns/avg": float(jnp.mean(returns)),
                "returns/max": float(jnp.max(returns)),
                "returns/min": float(jnp.min(returns))
            })

        # Add global timestep
        metrics["global_timestep"] = int(global_timestep)
        
        wandb.log(metrics)
    
env = FlattenObservationWrapper(env)
env = LogWrapper(env)

num_tokens = 1 + env.action_space(env_params).n + 256
config["MIN_ACTION_TOK"] = 1
config["MAX_ACTION_TOK"] = 4

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

    all_actions = []
    update_returns = []
    for t in range(config["NUM_STEPS"]):
        rng, _rng = jax.random.split(rng)
        tokenized = handle_continuous(obsv)
        pi, value, state = v_forward_jit(tokenized, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32) * tokenized.shape[-1])
        pi = distrax.Categorical(logits=pi[..., -1, config["MIN_ACTION_TOK"]:config["MAX_ACTION_TOK"] + 1])
        action = pi.sample(seed=_rng)
        ##
        current_actions = jax.device_get(action)
        all_actions.extend(current_actions.flatten().tolist())
        ##
        log_prob = pi.log_prob(action)
        _, value1, state = v_forward_jit(action, state, params, jnp.ones(config["NUM_ENVS"], dtype=jnp.int32))

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = v_env_step(rng_step, env_state, action, env_params)
        # Inside your training loop (after getting action and info):
        if wandbOn:
            # Create metric package
            log_package = (action, info, global_timestep)
            
            # Dispatch logging callback
            jax.debug.callback(log_all_metrics, *log_package)

        
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
    buf = JString(tokens_list, jnp.ones_like(tokens_list[:, 0]) * tokens_list.shape[1])
  


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