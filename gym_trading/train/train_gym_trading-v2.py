# %% ==========================================================================
# clear warnings
import warnings
import time
import gym
from sb3_contrib import RecurrentPPO
# from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_trading.envs.broker import Flag
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData

warnings.filterwarnings("ignore")
Flow = ExternalData.get_sample_order_book_data()

env = gym.make("GymTrading-v1",Flow = Flow) 
check_env(env)

num_cpu = 10 
venv = DummyVecEnv([lambda: Monitor(gym.make("GymTrading-v1",Flow = Flow))] * num_cpu)
# %%

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func

def biquadrate_schedule(initial_value):
    if isinstance(initial_value, str):initial_value = float(initial_value)
    def func(progress):return progress * progress * progress * progress * initial_value
    return func

model = RecurrentPPO(
    "MlpLstmPolicy", 
    venv, 
    verbose=1,
    learning_rate = biquadrate_schedule(3e-4),
    tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn/")

model.learn(total_timesteps=int(5e8), tb_log_name="RNN_PPO_init")
string = time.ctime().replace(" ","-").replace(":","-")
model.save("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1"+string)

# model = RecurrentPPO(
#     "MlpLstmPolicy", 
#     venv, 
#     verbose=1,
#     # n_steps = Flag.max_episode_steps, #n_steps: The number of steps to run for each environment per update
#     learning_rate = linear_schedule(3e-4),
#     tensorboard_log="/Users/kang/GitHub/NeuralLOB/venv_rnn/")
# model.learn(total_timesteps=int(3e6), tb_log_name="RNN_PPO_init")
# model.save("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1")

# tensorboard --logdir /Users/kang/GitHub/NeuralLOB/venv_rnn/

# %% test the train result

model = RecurrentPPO.load("/Users/kang/GitHub/NeuralLOB/tensorboard_rnn/rnn_ppo_gym_trading-v1Wed-Aug-31-19-58-55-2022.zip")
start = time.time()
env = gym.make("GymTrading-v1",Flow = Flow) ## TODO

info_list = []
Epoch = 0
for i in range(int(1e3)):
    obs = env.reset()
    # ;print(obs)
    done = False
    running_reward = 0
    for i in range(int(1e8)):
        if i//int(1e5) == i/int(1e5):
            print("Epoch {}, training time {}".format(Epoch,(time.time()-start)/60))
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            info_list.append(info)
            obs = env.reset()
            Epoch += 1
            break 

import pandas as pd
df = pd.DataFrame(info_list)
df.to_csv("info_df"+string+".csv")
# %% analyse the result

df1=df[(df.Advantage <= 100) & (df.Advantage>=-100)]
grouped = df1.groupby(df.Left)    
df2=grouped.mean()
df3 = df2.set_index("Left").drop(["Step"],axis =1)
fig = df3.plot().get_figure()
fig.savefig(string+'.png', dpi=300)



































