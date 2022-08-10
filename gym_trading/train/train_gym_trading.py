# %% ==========================================================================
# clear warnings
import warnings
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData
# from gym_trading.utils import timing
from time import time


warnings.filterwarnings("ignore")
Flow = ExternalData.get_sample_order_book_data()

env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
# env = gym.DummyVecEnv([lambda: gym.make("GymTrading-v1",Flow = Flow)])


check_env(env)
model = PPO("MultiInputPolicy", env, verbose=1)


start = time()
model.learn(total_timesteps=int(1e8), n_eval_episodes = int(1e5))
duration = time() - start
print(duration)


model.save("gym_trading-v1") 










