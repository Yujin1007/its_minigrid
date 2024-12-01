import os
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import numpy as np
from imitation.algorithms.bc import BC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
# from cfg_utils import load_map_from_example_dict, load_starting_pos_from_example_dict, load_goal_pos_from_example_dict, get_output_path, get_output_folder_name, get_model_path
from cfg_utils import *
from toy_envs.grid_nav import *
# Import callbacks
from gridnav_rl_callbacks import WandbCallback, GridNavVideoRecorderCallback, EpisodeTerminationCallback
from bc_simple import collect_demonstrations, rollout_steps
from imitation.data.types import Trajectory

def rollout_steps(model, env, max_steps=None, iseval=False):
    """
    Perform a single rollout of the environment and return a trajectory.

    Parameters:
        model: Trained model to predict actions.
        env: Environment to interact with.
        max_steps: Optional; maximum number of steps for the rollout.

    Returns:
        A single trajectory containing observations, actions, and metadata.
    """
    obs = env.reset()
    if isinstance(obs, tuple):  # Handle VecEnv API
        obs, _ = obs

    trajectory = {"obs": [], "acts": [], "infos": []}
    done = False
    step_count = 0

    while not done:
        print("--obs--/n", obs)
        action, _ = model.predict(obs)  # Predict action
        # action = action.item()  # Convert to scalar if necessary

        # Append current state and action
        trajectory["obs"].append(obs.copy())
        trajectory["acts"].append(action)

        # Take an environment step

        obs, reward, done, info = env.step(action)
        print(action)
        print("--obs--",obs.shape(),"/n",obs)
        print("--rew--",reward.shape(),"/n",reward)
        print("-done---",done.shape(),"/n",done)
        if isinstance(obs, tuple):  # Handle VecEnv API
            obs, info = obs

        trajectory["infos"].append(info)
        step_count += 1

        # Stop if max_steps is specified and reached
        if max_steps and step_count >= max_steps:
            break

    # Append the final observation
    trajectory["obs"].append(obs.copy())
    if iseval:
        if reward == 1:
            print("goal reached : ", len(trajectory['obs']))

        else:
            print("failed in finding solution")
    return Trajectory(
        obs=np.array(trajectory["obs"]),
        acts=np.array(trajectory["acts"]),
        infos=trajectory["infos"],
        terminal=True,
    )
def collect_demonstrations(model, env, num_episodes=10, max_steps=None, iseval=False):
    """
    Collect multiple trajectories (demonstrations) using the provided model.

    Parameters:
        model: Trained model to predict actions.
        env: Environment to interact with.
        num_episodes: Number of episodes to collect.
        max_steps: Optional; maximum number of steps per episode.

    Returns:
        A list of Trajectory objects.
    """
    trajectories = []
    for _ in range(num_episodes):
        trajectory = rollout_steps(model, env, max_steps, iseval)
        trajectories.append(trajectory)

    return trajectories
def collect_demonstrations_vectorized(model, env, num_episodes=20):
    """
    Collect demonstrations from a vectorized environment.

    Parameters:
        model: Trained model to predict actions.
        env: Vectorized environment (e.g., SubprocVecEnv).
        num_episodes: Total number of episodes to collect.

    Returns:
        A list of Trajectory objects.
    """
    trajectories = []
    episode_count = 0
    max_envs = env.num_envs  # Number of parallel environments

    # Initialize storage for each parallel environment
    parallel_trajectories = [{"obs": [], "acts": [], "infos": []} for _ in range(max_envs)]

    # Reset the environment
    obs = env.reset()
    if isinstance(obs, tuple):  # Handle VecEnv API
        obs, _ = obs

    # Continue collecting until the required number of episodes is reached
    while episode_count < num_episodes:
        # Predict actions for all environments
        actions, _ = model.predict(obs, deterministic=True)
        actions = actions.flatten()

        # Take a step in all environments
        next_obs, rewards, dones, infos = env.step(actions)

        # Store transitions for each environment
        for i in range(max_envs):
            parallel_trajectories[i]["obs"].append(obs[i].copy())
            parallel_trajectories[i]["acts"].append(actions[i])
            parallel_trajectories[i]["infos"].append(infos[i])

            # Check if the episode is done
            if dones[i]:
                # Final observation for this environment
                parallel_trajectories[i]["obs"].append(next_obs[i].copy())

                # Create a trajectory for the finished episode
                trajectories.append(
                    Trajectory(
                        obs=np.array(parallel_trajectories[i]["obs"]),
                        acts=np.array(parallel_trajectories[i]["acts"]),
                        infos=parallel_trajectories[i]["infos"],
                        terminal=True,
                    )
                )
                episode_count += 1

                # Reset this trajectory buffer for the next episode
                parallel_trajectories[i] = {"obs": [], "acts": [], "infos": []}

        # Update current observations
        obs = next_obs

    return trajectories
@hydra.main(version_base=None, config_path="config", config_name="train_bc_config2")
def train_bc_with_hydra(cfg: DictConfig):
    """
    Train a Behavior Cloning (BC) policy with Hydra-loaded configuration and callbacks.

    Parameters:
        cfg: Configuration loaded by Hydra.
    """
    # Set up logging directories
    cfg.logging.run_name = get_output_folder_name(cfg.log_folder)
    cfg.logging.run_path = get_output_path()

    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)
    checkpoint_dir = os.path.join(cfg.logging.run_path, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    map_array = load_map_from_example_dict(cfg.env.example_name)
    starting_pos = load_starting_pos_from_example_dict(cfg.env.example_name)
    goal_pos = load_goal_pos_from_example_dict(cfg.env.example_name)
    grid_class = GridNavigationEnv
    episode_length = cfg.env.episode_length
    pretrained_model_path = get_model_path(cfg.teacher.path, cfg.teacher.checkpoint)
    with wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.run_name,
            tags=cfg.logging.wandb_tags,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            mode=cfg.logging.wandb_mode,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
    )as wandb_run:
        make_env_fn = lambda: Monitor(
            grid_class(map_array=np.copy(map_array), starting_pos=starting_pos, goal_pos=goal_pos,
                       render_mode="rgb_array",
                       episode_length=episode_length))

        env = make_vec_env(
            make_env_fn,
            n_envs=cfg.compute.n_cpu_workers,
            seed=cfg.seed,
            vec_env_cls=SubprocVecEnv,
        )

    # make_env_fn = lambda: Monitor(
    #     grid_class(map_array=np.copy(map_array), starting_pos=starting_pos, goal_pos=goal_pos, render_mode="rgb_array",
    #                episode_length=episode_length))
    #
    # env = DummyVecEnv([make_env_fn]) if cfg.compute.n_cpu_workers == 1 else SubprocVecEnv([make_env_fn])

    # Generate demonstrations
        pretrained_model = PPO.load(pretrained_model_path, env=env)
        trajectories = collect_demonstrations_vectorized(pretrained_model, env, num_episodes=cfg.imitation.num_episodes)

        # Initialize BC
        bc = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            rng=np.random.default_rng(cfg.seed),
        )

        # Callbacks
        wandb_callback = WandbCallback(
            model_save_path=str(checkpoint_dir),
            model_save_freq=cfg.logging.model_save_freq,
            verbose=2,
        )
        video_callback = GridNavVideoRecorderCallback(
            env,
            rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
            render_freq=cfg.logging.video_save_freq,
        )
        episodic_callback = EpisodeTerminationCallback()
        callback_list = [wandb_callback, video_callback, episodic_callback]

        # Train BC with callbacks
        # bc.train(n_epochs=cfg.imitation.bc_epochs, callbacks=callback_list)
        for epoch in range(cfg.imitation.bc_epochs):
            bc.train(n_epochs=1)  # Train for one epoch at a time

            # Manually invoke each callback
            for callback in callback_list:
                callback.on_training_end(locals())

        # Save the trained policy
        model_save_path = os.path.join(checkpoint_dir, "imitation_policy.pth")
        bc.policy.save(model_save_path)
        print(f"BC policy saved to {model_save_path}")


if __name__ == "__main__":
    train_bc_with_hydra()