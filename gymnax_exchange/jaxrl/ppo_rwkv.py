# docker run -it --rm --gpus '"device=0"' -v $(pwd):/app -v $(pwd)/../cache:/app/cache --name ${USER}_rwkvcontainer ${USER}_rwkv python -m testing.finetune

import jax
import jax.numpy as jnp

import numpy as np

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

# jax.config.update("jax_compilation_cache_dir", "/app/cache/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import distrax
import optax

import tyro
import time
from dataclasses import dataclass
from typing import Optional, Literal, NamedTuple

from jax_rwkv.src.auto import models, get_model
from jax_rwkv.src.jax_rwkv.base_rwkv import layer_norm
from jax_rwkv.src.jax_rwkv.utils import sample_logits
from jax_rwkv.src.utils import process_long_seq, simple_sampler

from functools import partial


config = {
    "LR": 2.5e-4,
        "NUM_ENVS": 256,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 4e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "AlphaTradeMM",
        "ANNEAL_LR": True,
        "DEBUG": True,
        
        "TASKSIDE": "random", # "random", "buy", "sell"
        "REWARD_LAMBDA": 0.001, #0.001,
        "ACTION_TYPE": "pure", # "delta"
        "WINDOW_INDEX": -1, # 2 fix random episode #-1,
        "MAX_TASK_SIZE": 100,
        "EPISODE_TIME": 60 * 5, # time in seconds
        "DATA_TYPE": "fixed_time", # "fixed_time", "fixed_steps"
        "ATFOLDER": "/home/duser/AlphaTrade/training_oneDay"
}

config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"])

@dataclass
class Args:
    seed: int = 0
    model_choice: Literal["6g0.1B"] =  "6g0.1B"

    num_iters: int = 1
    dtype: Optional[str] = None
    rwkv_type: str = "ScanRWKV"

    sequence_length: int = 128

    num_sequences: int = 1

def get_ppo_agent(RWKV, params,obs_size,action_size, seed=0):
    key = jax.random.key(seed)
    # Modify the input layer to accept non-textual data
    input_dim = obs_size  # Example input dimension, adjust based on your data: size of obs space
    params['input_layer'] = {'weight': jax.random.normal(key, shape=(input_dim, params['emb']['weight'].shape[1]))}
    params['head'] = {'weight': jax.random.normal(key, (args.n_actions, params['ln_out']['scale'].shape[0]))}
    params['value_head'] = {'weight': jax.random.normal(key, (1, params['ln_out']['scale'].shape[0]))}
    
    def forward(input_data, state, params, length=None):
        # Project input data to the model's hidden dimension
        x = input_data @ params['input_layer']['weight']

          # RWKV processing
        x, state = RWKV.forward_seq(x, state, params)
        x_ln = layer_norm(x, params['ln_out'])
        
        # Get logits and value
        logits = x_ln @ params['head']['weight'].T
        value = x_ln @ params['value_head']['weight'].T
        return logits, value, state
    return forward, params

def act(forward_fn, params, obs, key, state):
    logits, value, new_state = forward_fn(obs, state, params)
    action = distrax.Categorical(logits=logits).sample(seed=key)
    return action, new_state, value, logits



def compute_advantages(rewards, values, dones, last_value):
    advantages = []
    last_advantage = 0
    next_value = last_value
    
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + config["GAMMA"] * next_value * mask - values[t]
        last_advantage = delta + config["GAMMA"] * config["GAE_LAMBDA"] * mask * last_advantage
        advantages.insert(0, last_advantage)
        next_value = values[t]
    
    returns = jnp.array(advantages) + jnp.array(values)
    return jnp.array(advantages), returns

def ppo_loss(params, forward_fn, batch, key):
    obs, actions, old_log_probs, advantages, returns = batch
    logits, values, _ = forward_fn(obs, None, params)
    
    # Normalize advantages
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    # Calculate policy loss
    pi = distrax.Categorical(logits=logits)
    log_probs = pi.log_prob(actions)
    ratio = jnp.exp(log_probs - old_log_probs)
    actor_loss = -jnp.minimum(ratio * advantages, 
                             jnp.clip(ratio, 1-config["CLIP_EPS"], 1+config["CLIP_EPS"]) * advantages).mean()
    
    # Calculate value loss
    value_loss = 0.5 * jnp.square(values - returns).mean()
    
    # Entropy bonus
    entropy = pi.entropy().mean()
    
    total_loss = actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
    return total_loss, (actor_loss, value_loss, entropy)

def update_step(params, opt_state, batch, forward_fn, solver, key):
    (loss, (actor_loss, value_loss, entropy)), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
        params, forward_fn, batch, key
    )
    updates, opt_state = solver.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, actor_loss, value_loss, entropy

if __name__ == '__main__':
    args = Args()
    RWKV, params, _ = get_model(args.model_choice, None)
    
    # Initialize with dummy observation
    obs = jnp.zeros((args.num_sequences, 23))  # Your observation space
    key = jax.random.PRNGKey(args.seed)
    state = RWKV.default_state(params)
    
    # Initialize PPO
    forward_fn, params = get_ppo_agent(RWKV, params, args)
    solver = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"])
    )
    opt_state = solver.init(params)
    
    # Training loop
    for _ in range(config["NUM_UPDATES"]):
        key, subkey = jax.random.split(key)
        
        # Collect trajectories
        actions, values, log_probs = [], [], []
        for _ in range(args.sequence_length):
            key, subkey = jax.random.split(key)
            action, state, value, logits = act(forward_fn, params, obs, subkey, state)
            actions.append(action)
            values.append(value)
            log_probs.append(distrax.Categorical(logits=logits).log_prob(action))
        
        # Calculate advantages and returns (simplified)
        advantages = jnp.array(values) - jnp.mean(values)
        returns = jnp.array(values) + advantages
        
        # Update network
        batch = (obs, jnp.array(actions), jnp.array(log_probs), advantages, returns)
        params, opt_state, loss, actor_loss, value_loss, entropy = update_step(
            params, opt_state, batch, forward_fn, solver, key
        )
        
        print(f"Loss: {loss:.3f}, Policy: {actor_loss:.3f}, Value: {value_loss:.3f}, Entropy: {entropy:.3f}")
        
 
