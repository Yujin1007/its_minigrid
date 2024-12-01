from turtledemo.penrose import start

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

import os
import wandb
from typing import Any, Dict, List, Union
from numpy.typing import NDArray
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
from loguru import logger

# from custom_sb3 import PPO
# from reinforce_model import REINFORCE
from toy_envs.grid_nav import *
from toy_examples_main import examples
from gridnav_rl_callbacks import WandbCallback, GridNavVideoRecorderCallback, RLBCTrainingCallback
from cfg_utils import load_map_from_example_dict, load_starting_pos_from_example_dict, load_goal_pos_from_example_dict, get_output_path, get_output_folder_name


@hydra.main(version_base=None, config_path="config", config_name="train_its_config")
def train(cfg: DictConfig):
    cfg.logging.run_name = get_output_folder_name(cfg.log_folder)
    cfg.logging.run_path = get_output_path()

    logger.info(f"Logging to {cfg.logging.run_path}\nRun name: {cfg.logging.run_name}")

    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)


    # Initialize the environment
    map_array = load_map_from_example_dict(cfg.env.example_name)
    # starting_pos = load_starting_pos_from_example_dict(cfg.env.example_name)
    #goal_pos = load_goal_pos_from_example_dict(cfg.env.example_name)
    goal_pos = np.argwhere(map_array == G)[0]
    start_pos = np.argwhere(map_array == A)[0]
    # if goal_pos.size == 0:
    #     goal_pos = np.argwhere(map_array == G)[0]

    grid_class = GridNavigationEnv
    with wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            tags=cfg.logging.wandb_tags,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            mode=cfg.logging.wandb_mode,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
    ) as wandb_run:
        lr = cfg.rl_algo.lr
        ent_coef = cfg.rl_algo.ent_coef
        vf_coef = cfg.rl_algo.vf_coef
        episode_length = cfg.env.episode_length

        make_env_fn = lambda: Monitor(
            grid_class(map_array=np.copy(map_array), goal_pos=goal_pos, starting_pos=start_pos, render_mode="rgb_array",
                       episode_length=episode_length))

        training_env = make_vec_env(
            make_env_fn,
            n_envs=cfg.compute.n_cpu_workers,
            seed=cfg.seed,
            vec_env_cls=SubprocVecEnv,
        )
        # training_env = make_vec_env(
        #     make_env_fn,
        #     n_envs=cfg.compute.n_cpu_workers,
        #     seed=cfg.seed,
        #     vec_env_cls=SubprocVecEnv,
        # )
        eval_env = grid_class(map_array=np.copy(map_array), starting_pos=start_pos, goal_pos=goal_pos, render_mode="rgb_array",
                         episode_length=episode_length)

        # Define the model
        model = PPO("MlpPolicy",
                    training_env,
                    n_steps=cfg.env.episode_length,
                    n_epochs=cfg.rl_algo.n_epochs,
                    batch_size=cfg.rl_algo.batch_size,
                    learning_rate=lr,
                    # tensorboard_log=os.path.join(cfg.logging.run_path, "tensorboard"),
                    tensorboard_log=None,
                    gamma=cfg.rl_algo.gamma,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    verbose=1)

        rng = np.random.default_rng(seed=42)
        bc_model = BC(
            observation_space=eval_env.observation_space,
            action_space=eval_env.action_space,
            rng=rng,
        )

        # # Make an alias for the wandb in the run_path
        # if cfg.logging.wandb_mode != "disabled" and not cfg.sweep.enabled:
        #     os.symlink(os.path.abspath(wandb_run.dir), os.path.join(cfg.logging.run_path, "wandb"),
        #                target_is_directory=True)

        checkpoint_dir = os.path.join(cfg.logging.run_path, "checkpoint")

        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            model_save_freq=cfg.logging.model_save_freq // cfg.compute.n_cpu_workers,
            verbose=2,
        )

        video_callback = GridNavVideoRecorderCallback(
            SubprocVecEnv([make_env_fn]),
            rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
            render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers,
            map_array=np.copy(map_array),
            goal_pos=goal_pos,
        )
        """
        Parameters:
            eval_env (gym.Env): The evaluation environment for collecting deterministic trajectories.
            bc_model (any BC model): The behavior cloning model.
            bc_training_interval (int): Timesteps between BC training sessions.
            bc_epochs (int): Number of epochs for BC training.
            bc_batch_size (int): Batch size for BC training.
            bc_save_path (str): Path to save the BC model.
            verbose (int): Verbosity level.
        """
        bc_callback = RLBCTrainingCallback(eval_env=eval_env,
                                           bc_model=bc_model,
                                           bc_training_interval=cfg.bc_algo.training_interval,
                                           bc_batch_size=cfg.bc_algo.batch_size,
                                           bc_epochs=cfg.bc_algo.epochs,
                                           bc_save_path=os.path.join(cfg.logging.run_path, "eval"),
                                           run_path=cfg.logging.run_path,
                                           )

        callback_list = [wandb_callback, video_callback, bc_callback]

        # Train the model
        model.learn(
            total_timesteps=cfg.rl_algo.total_timesteps,
            progress_bar=True,
            callback=CallbackList(callback_list))

        logger.info("Saving final model")
        model.save(str(os.path.join(checkpoint_dir, "final_model")))

        logger.info("Done.")
        wandb_run.finish()



if __name__ == "__main__":
    # train_or_sweep()
    train()
