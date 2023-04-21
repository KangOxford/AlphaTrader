from gym_exchange import Config
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_exchange.environment.training_env.train_env import TrainEnv
from train import utils
import warnings; warnings.filterwarnings("ignore") # clear warnings
import wandb
from train.sb3 import WandbCallback


path = utils.get_path_by_platform()


if __name__ == "__main__":
    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": int(1e12),
    }

    run = wandb.init(
        project="AlphaTrade",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )


    def make_env():
        env = Monitor(TrainEnv())  # record stats such as returns
        env = Monitor(env)  # record stats such as returns
        return env
    venv = DummyVecEnv([make_env])


    model = RecurrentPPO(
        config["policy_type"],
         venv,
         verbose=1,
         learning_rate=utils.linear_schedule(1e-3),
         tensorboard_log=f"{path}train/output/runs/{run.id}")

    model.learn(
       tb_log_name="RNN_PPO_Wandb",
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=1,
        ),
        log_interval=1,

    )

    run.finish()

