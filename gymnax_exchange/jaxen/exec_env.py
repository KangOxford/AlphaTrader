# ============== testing scripts ===============
from jax import config
config.update("jax_enable_x64",True)
import jax
import jax.numpy as jnp
import gymnax
import sys
sys.path.append('/Users/sasrey/AlphaTrade')
sys.path.append('/homes/80/kang/AlphaTrade')
# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
import chex
import timeit

import faulthandler
faulthandler.enable()

print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())

chex.assert_gpu_available(backend=None)

# #Code snippet to disable all jitting.
from jax import config
config.update("jax_disable_jit", False)
# config.update("jax_enable_x64",True)
# # config.update("jax_disable_jit", True)
# ============== testing scripts ===============



from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxes.jaxob_new import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv

import time 

@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    best_asks: chex.Array
    best_bids: chex.Array
    init_time: chex.Array
    time: chex.Array
    customIDcounter: int
    window_index:int
    step_counter: int
    init_price:int
    task_to_execute:int
    quant_executed:int
    total_revenue:int
    max_steps_in_episode: int


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    stateArray_list: chex.Array
    obs_sell_list: chex.Array
    obs_buy_list: chex.Array
    episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    # max_steps_in_episode: int = 100 # TODO should be a variable, decied by the data_window
    # messages_per_step: int=1 # TODO never used, should be removed?
    time_per_step: int= 0##Going forward, assume that 0 implies not to use time step?
    time_delay_obs_act: chex.Array = jnp.array([0, 0]) #0ns time delay.
    

class ExecutionEnv(BaseLOBEnv):
    def __init__(self,alphatradePath,task,debug=False):
        super().__init__(alphatradePath)
        self.n_actions = 4 # [A, M, P, PP] Agressive, MidPrice, Passive, Second Passive
        self.task = task
        # self.task_size = 500 # num to sell or buy for the task
        self.task_size = 200 # num to sell or buy for the task
        self.n_fragment_max=2
        self.n_ticks_in_book=20 
        # self.debug : bool = False

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        # return EnvParams(self.messages,self.books)
        return EnvParams(self.messages,self.books,self.stateArray_list,self.obs_sell_list,self.obs_buy_list)
    

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Dict, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data
        data_messages=job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        action_msgs = self.getActionMsgs(action, state, params)
        #Currently just naive cancellation of all agent orders in the book. #TODO avoid being sent to the back of the queue every time. 
        cnl_msgs=job.getCancelMsgs(state.ask_raw_orders if self.task=='sell' else state.bid_raw_orders,-8999,self.n_actions,-1 if self.task=='sell' else 1)
        #Add to the top of the data messages
        total_messages=jnp.concatenate([cnl_msgs,action_msgs,data_messages],axis=0) # TODO DO NOT FORGET TO ENABLE CANCEL MSG
        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int64)
        #Process messages of step (action+data) through the orderbook
        asks,bids,trades,bestasks,bestbids=job.scan_through_entire_array_save_bidask(total_messages,(state.ask_raw_orders,state.bid_raw_orders,trades_reinit),self.stepLines) 
        # ========== get reward and revenue ==========
        executed = jnp.where((trades[:, 0] > 0)[:, jnp.newaxis], trades, 0)
        mask2 = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
        agentTrades = jnp.where(mask2[:, jnp.newaxis], executed, 0)
        def truncate_agent_trades(agentTrades, remainQuant):
            quantities = agentTrades[:, 1]
            cumsum_quantities = jnp.cumsum(quantities)
            cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
            truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, jnp.newaxis] > cut_idx, jnp.zeros_like(agentTrades[0]), agentTrades.at[:, 1].set(jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, jnp.where(jnp.arange(len(quantities)) == cut_idx, remainQuant - cumsum_quantities[cut_idx - 1], 0))))
            return jnp.where(remainQuant >= jnp.sum(quantities), agentTrades, jnp.where(remainQuant <= quantities[0], jnp.zeros_like(agentTrades).at[0, :].set(agentTrades[0]).at[0, 1].set(remainQuant), truncated_agentTrades))
        agentTrades = truncate_agent_trades(agentTrades, state.task_to_execute-state.quant_executed)
        new_execution = agentTrades[:,1].sum()
        revenue = (agentTrades[:,0]//self.tick_size * agentTrades[:,1]).sum()
        agentQuant = agentTrades[:,1].sum()
        vwap =(executed[:,0]//self.tick_size* executed[:,1]).sum()//(executed[:,1]).sum()
        advantage = revenue - vwap * agentQuant ### (weightedavgtradeprice-vwap)*agentQuant ### revenue = weightedavgtradeprice*agentQuant
        Lambda = 0.0 # FIXME shoud be moved to EnvState or EnvParams
        # Lambda = 0.5 # FIXME shoud be moved to EnvState or EnvParams
        drift = agentQuant * (vwap - state.init_price//self.tick_size)
        rewardValue = advantage + Lambda * drift
        reward = jnp.sign(agentTrades[0,0]) * rewardValue # if no value agentTrades then the reward is set to be zero
        # ========== get reward and revenue END ==========
        #Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
        state = EnvState(asks,bids,trades,bestasks[-self.stepLines:],bestbids[-self.stepLines:],state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,state.step_counter+1,state.init_price,state.task_to_execute,state.quant_executed+new_execution,state.total_revenue+revenue,state.max_steps_in_episode)
        done = self.is_terminal(state,params)
        return self.get_obs(state,params),state,reward,done,{"window_index":state.window_index,"total_revenue":state.total_revenue,"quant_executed":state.quant_executed,"task_to_execute":state.task_to_execute}



    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        idx_data_window = jax.random.randint(key, minval=0, maxval=self.n_windows, shape=())

        def stateArray2state(stateArray):
            state0 = stateArray[:,0:6];state1 = stateArray[:,6:12];state2 = stateArray[:,12:18];state3 = stateArray[:,18:20];state4 = stateArray[:,20:22]
            state5 = stateArray[0:2,22:23].squeeze(axis=-1);state6 = stateArray[2:4,22:23].squeeze(axis=-1);state10= stateArray[4:5,22:23][0].squeeze(axis=-1)
            return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,0,state10,self.task_size,0,0,self.max_steps_in_episode_arr[idx_data_window])
        stateArray = params.stateArray_list[idx_data_window]
        state_ = stateArray2state(stateArray)
        obs_sell = params.obs_sell_list[idx_data_window]
        obs_buy = params.obs_buy_list[idx_data_window]
        state = EnvState(*state_)
        obs = obs_sell if self.task == "sell" else obs_buy
        return obs,state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return ((state.time-state.init_time)[0]>params.episode_time) | (state.task_to_execute-state.quant_executed<=0)
    
    def getActionMsgs(self, action: Dict, state: EnvState, params: EnvParams):
        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs ---------------
        types=jnp.ones((self.n_actions,),jnp.int64)
        sides=-1*jnp.ones((self.n_actions,),jnp.int64) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int64) #if self.task=='buy'
        trader_ids=jnp.ones((self.n_actions,),jnp.int64)*self.trader_unique_id #This agent will always have the same (unique) trader ID
        order_ids=jnp.ones((self.n_actions,),jnp.int64)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions) #Each message has a unique ID
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2)) #time from last (data) message of prev. step + some delay
        #Stack (Concatenate) the info into an array 
        # --------------- 01 rest info for deciding action_msgs ---------------
        
        # --------------- 02 info for deciding prices ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        # jax.debug.breakpoint()
        best_ask, best_bid = state.best_asks[-1,0], state.best_bids[-1,0]
        A = best_bid if self.task=='sell' else best_ask # aggressive would be at bids
        M = (best_bid + best_ask)//2//self.tick_size*self.tick_size 
        P = best_ask if self.task=='sell' else best_bid
        PP= best_ask+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bid-self.tick_size*self.n_ticks_in_book
        # --------------- 02 info for deciding prices ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        remainingTime = params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int64)
        marketOrderTime = jnp.array(60, dtype=jnp.int64) # in seconds, means the last minute was left for market order
        ifMarketOrder = (remainingTime <= marketOrderTime)
        def market_order_logic(state: EnvState,  A: float):
            quant = state.task_to_execute - state.quant_executed
            price = A + (-1 if self.task == 'sell' else 1) * (self.tick_size * 100) * 100
            #FIXME not very clean way to implement, but works:
            quants = jnp.asarray((quant//4,quant//4,quant//4,quant-3*quant//4),jnp.int64) 
            prices = jnp.asarray((price, price , price, price),jnp.int64)
            # (self.tick_size * 100) : one dollar
            # (self.tick_size * 100) * 100: choose your own number here(the second 100)
            return quants, prices
        def normal_order_logic(state: EnvState, action: jnp.ndarray, A: float, M: float, P: float, PP: float):
            quants = action.astype(jnp.int64) # from action space
            prices = jnp.asarray((A, M, P, PP), jnp.int64)
            return quants, prices
        market_quants, market_prices = market_order_logic(state, A)
        normal_quants, normal_prices = normal_order_logic(state, action, A, M, P, PP)
        quants = jnp.where(ifMarketOrder, market_quants, normal_quants)
        prices = jnp.where(ifMarketOrder, market_prices, normal_prices)
        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,order_ids],axis=1)
        # jax.debug.breakpoint()
        action_msgs=jnp.concatenate([action_msgs,times],axis=1)
        return action_msgs
        # ============================== Get Action_msgs ==============================


    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # ========= self.get_obs(state,params) =============
        # -----------------------1--------------------------
        best_asks=state.best_asks[:,0]
        best_bids =state.best_bids[:,0]
        mid_prices=(best_asks+best_bids)//2//self.tick_size*self.tick_size 
        second_passives = best_asks+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bids-self.tick_size*self.n_ticks_in_book
        spreads = best_asks - best_bids
        # -----------------------2--------------------------
        timeOfDay = state.time
        deltaT = state.time - state.init_time
        # -----------------------3--------------------------
        initPrice = state.init_price
        priceDrift = mid_prices[-1] - state.init_price
        # -----------------------4--------------------------
        taskSize = state.task_to_execute
        executed_quant=state.quant_executed
        # -----------------------5--------------------------
        shallowImbalance = state.best_asks[:,1]- state.best_bids[:,1]
        # ========= self.get_obs(state,params) =============
        obs = jnp.concatenate((best_bids,best_asks,mid_prices,second_passives,spreads,timeOfDay,deltaT,jnp.array([initPrice]),jnp.array([priceDrift]),jnp.array([taskSize]),jnp.array([executed_quant]),shallowImbalance))
        # jax.debug.breakpoint()
        return obs

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions


    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(0,100,(self.n_actions,),dtype=jnp.int64)
    
    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        space = spaces.Box(-10000,99999999,(608,),dtype=jnp.int64) 
        #space = spaces.Box(-10000,99999999,(510,),dtype=jnp.int64)
        return space

    #FIXME:Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5) 
    #           fields in the bid/ask arrays to return something of value. Not sure if actually needed.   
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int64),
                "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int64),
                "trades": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nTradesLogged),dtype=jnp.int64),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:",ATFolder)
    except:
        # ATFolder = '/home/duser/AlphaTrade'
        ATFolder = '/homes/80/kang/AlphaTrade'
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    env=ExecutionEnv(ATFolder,"sell")
    env_params=env.default_params
    print(env_params.message_data.shape, env_params.book_data.shape)

    start=time.time()
    obs,state=env.reset(key_reset,env_params)
    print("State after reset: \n",state)
    print("Time for reset: \n",time.time()-start)

    print(env_params.message_data.shape, env_params.book_data.shape)
    # # ==================== ACTION ====================
    # # ---------- acion from random sampling ----------
    # i=0;test_action=env.action_space().sample(key_policy)
    # print(f"Sampled {i}th actions are: ",test_action)
    # start=time.time()
    # obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
    # print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
    # print(f"Time for {i} step: \n",time.time()-start)
    # # ---------- acion from random sampling ----------
    # # ==================== ACTION ====================
# jitted_func = jax.jit(env.step)
    # jax.jit(env.step) 

    for i in range(1,100):
        # ==================== ACTION ====================
        # ---------- acion from random sampling ----------
        test_action=env.action_space().sample(key_policy)
        print(f"Sampled {i}th actions are: ",test_action)
        start=time.time()
        obs,state,reward,done,info=env.step(key_step, state,test_action, env_params)
        print(f"State after {i} step: \n",state,done,file=open('output.txt','a'))
        print(f"Time for {i} step: \n",time.time()-start)
        # ---------- acion from random sampling ----------
        # ==================== ACTION ====================

    # ####### Testing the vmap abilities ########
    
    enable_vmap=True
    if enable_vmap:
        # with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)

        test_actions=vmap_act_sample(vmap_keys)
        print(test_actions)

        start=time.time()
        obs, state = vmap_reset(vmap_keys, env_params)
        print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

        start=time.time()
        n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
        print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)



    # enable_vmap=True
    # if enable_vmap:
    #     vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        
    #     with jax.profiler.trace("/homes/80/kang/AlphaTrade/wandb/jax-trace"):
    #         vmap_step = jax.block_until_ready(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    #     vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))

    #     num_envs = 10
    #     vmap_keys = jax.random.split(rng, num_envs)

    #     test_actions=vmap_act_sample(vmap_keys)
    #     print(test_actions)

    #     start=time.time()
    #     obs, state = vmap_reset(vmap_keys, env_params)
    #     print("Time for vmap reset with,",num_envs, " environments : \n",time.time()-start)

    #     start=time.time()
    #     n_obs, n_state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
    #     print("Time for vmap step with,",num_envs, " environments : \n",time.time()-start)
