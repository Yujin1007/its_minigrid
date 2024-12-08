import os
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.bc import BC
from toy_envs.grid_nav import *
from toy_examples_main import examples
from gridnav_rl_callbacks import WandbCallback, GridNavVideoRecorderCallback, EpisodeTerminationCallback
from cfg_utils import *
from omegaconf import DictConfig, OmegaConf
import hydra
from imitation.data.types import Trajectory
import numpy as np
from toy_envs.toy_env_utils import masking_obs
import numpy as np
from stable_baselines3 import PPO  # Use the same class as your saved BC policy

from imitation.algorithms.dagger import SimpleDAggerTrainer
frames = []
W = -1    # wall
O = 0     #open space
R = 1     #red object
B = 2     #blue object
G = 3     #goal
A = 4     #agent
def evaluate_policy(model, env, save_path, num_episodes=10,render=False):
    """
    Evaluate a trained policy on the environment.

    Parameters:
        policy: The trained policy (e.g., loaded from BC or RL model).
        env: The environment to evaluate the policy on.
        num_episodes: Number of episodes to run for evaluation.
        render: Whether to render the environment during evaluation.

    Returns:
        A dictionary with evaluation metrics.
    """
    rewards = []
    success_count = 0
    policy = model.policy
    global frames
    # video_callback = GridNavVideoRecorderCallback(
    #     env,
    #     rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
    #     render_freq=cfg.logging.video_save_freq,
    # )

    for episode in range(num_episodes):
        frames = []
        _ = rollout_steps(model=policy, env=env, iseval=True, episode=episode)
        save_file = os.path.join(save_path, f"testing_{episode}.gif")
        imageio.mimsave(save_file, frames, duration=1 / 20, loop=0)

        writer = imageio.get_writer(f'testing_{episode}.mp4', fps=20)

        for im in frames:
            writer.append_data(im)

        writer.close()

def rollout_steps(model, env, max_steps=None, iseval=False, isStudent=True, episode=0):
    """
    Perform a single rollout of the environment and return a trajectory.

    Parameters:
        model: Trained model to predict actions.
        env: Environment to interact with.
        max_steps: Optional; maximum number of steps for the rollout.

    Returns:
        A single trajectory containing observations, actions, and metadata.
    """
    global frames
    # obj_idx = int(episode%2)
    # obs = env.reset(obj_idx)
    obs = env.reset()
    if iseval:
        print("initial scence \n", obs)
    if isinstance(obs, tuple):  # Handle VecEnv API
        obs, _ = obs

    trajectory = {"obs": [], "acts": [], "infos": []}
    obs_raw_array = []
    done = False
    step_count = 0
    frame = []
    frame.append(env.render())
    timestep = 0
    while not done:
        timestep +=1
        if isStudent:
            masked_observation = masking_obs(obs)
            action, _ = model.predict(masked_observation, deterministic=True)  # Predict action
        else:
            action, _ = model.predict(obs, deterministic=True)  # Predict action
        action = action.item()  # Convert to scalar if necessary

        # Append current state and action
        # trajectory["obs"].append(obs.copy())
        obs_raw_array.append(obs.copy())
        trajectory["acts"].append(action)

        # Take an environment step

        obs, reward, done, _, info = env.step(action)
        # callback._on_step().
        if isinstance(obs, tuple):  # Handle VecEnv API
            obs, info = obs

        trajectory["infos"].append(info)
        step_count += 1
        frame.append(env.render())
        # Stop if max_steps is specified and reached
        if max_steps and step_count >= max_steps:
            break
    frames.append(frame)
    print(f"Time step : {timestep}, info: {info}")
    frame = []
    # Append the final observation
    # trajectory["obs"].append(obs.copy())
    obs_raw_array.append(obs.copy())
    for obs in obs_raw_array:
        masked_observation = masking_obs(obs)
        trajectory["obs"].append(masked_observation)
    if iseval:
        if reward == 1:
            print("goal reached : ", len(trajectory['obs']))

        else:
            print("failed in finding solution")
        print(trajectory["acts"])

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
        trajectory = rollout_steps(model, env, max_steps, iseval, isStudent=False)
        trajectories.append(trajectory)

    return trajectories

def train(cfg: DictConfig):
    global frames
    # pretrained_model_path = cfg.bc_algo.pre_trained_path# "toy_teacher/rl_train_logs/2024-11-21-151501_map=map_rl=ppo-epochs=10-eplen=100_s=9_nt=None/checkpoint/final_mocgdel.zip"  # Update with your saved model path
    # model_save_path = cfg.log_path
    map_array = load_map_from_example_dict(cfg.env.example_name)
    # starting_pos = load_starting_pos_from_example_dict(cfg.env.example_name)
    goal_pos = np.argwhere(map_array == G)[0]
    # goal_pos =
    episode_length = cfg.env.episode_length
    # grid_class = GridNavigationEnv
    grid_class = GridNavigationCurriculumEnv
    env = grid_class(map_array=np.copy(map_array), goal_pos=goal_pos, render_mode="rgb_array",
                                 episode_length=episode_length)



    # Step 2: Generate Demonstrations

    # Collect demonstrations
    log_path = os.path.join(cfg.teacher.path, "checkpoint")
    log_path = os.path.join(log_path, cfg.teacher.checkpoint)
    eval_path = os.path.join(cfg.teacher.path, "eval")
    eval_path = os.path.join(eval_path, cfg.teacher.checkpoint)
    pretrained_model = PPO.load(log_path, env=env)
    for level in range(3):
        env.curriculum = level
        _ = collect_demonstrations(pretrained_model, env, num_episodes=5)

        if not os.path.exists(eval_path):
            # If it doesn't exist, create it
            os.makedirs(eval_path)
        # imageio.mimsave(f"{log_path}/student_failed_state_Teacher.gif", frames, duration=1 / 20, loop=0)

        for i, im in enumerate(frames):

            imageio.mimsave(f"{eval_path}/Level{level}_{i}.gif", im, duration=1 / 20, loop=0)
        frames = []
    # writer.close()

def dict_to_namespace(d):
    """
    Recursively convert a dictionary into a SimpleNamespace.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d
if __name__ == "__main__":
    cfg_dict = {
        "teacher":{
            "path": "toy_teacher/rl_train_logs/empty_map2_2024-12-05-160310_nt=DenseRewardCurriculum_empty2_ent_coef05",
            "checkpoint": "model_18000000_steps",

        },
        "env":{
            "example_name": "empty_map2",
            "episode_length": 100
        },

    }
    cfg = dict_to_namespace(cfg_dict)
    train(cfg)
