from imitation.data.rollout import generate_trajectories
from random import random
from toy_envs.toy_env_utils import *
from torch.backends.cudnn import deterministic

from imitation.data.types import Trajectory
from toy_envs.grid_nav import *

def generate_unmasked_trajectories(expert_policy, env, n_episodes=10):
    """
    Generate trajectories for the expert using unmasked observations.

    Parameters:
        expert_policy: The expert policy.
        env: The DualObservationWrapper environment.
        n_episodes: Number of episodes to collect.

    Returns:
        Trajectories collected by the expert.
    """
    trajectories = []

    for _ in range(n_episodes):
        trajectory = {"obs": [], "acts": [], "infos": []}

        obs, _ = env.reset()
        masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        trajectory["obs"].append(masked_obs)
        while not done:
            action, _ = expert_policy.predict(obs, deterministic=True)
            action = action.item()
            obs, reward, done, _, info = env.step(action)
            masked_obs = masking_obs(obs)
            # obs = obs["unmasked"]  # Always use unmasked for the expert
            trajectory["acts"].append(action)
            trajectory["infos"].append(info)
            trajectory["obs"].append(masked_obs)

        trajectories.append(
            Trajectory(
                obs=np.array(trajectory["obs"]),
                acts=np.array(trajectory["acts"]),
                infos=trajectory["infos"],
                terminal=True,
            )
        )

    return trajectories

def collect_augmented_trajectories(trajectories, expert_policy, bc_policy, env, beta=0.8):
    """
    Generate trajectories for the expert using unmasked observations.

    Parameters:
        expert_policy: The expert policy.
        env: The DualObservationWrapper environment.
        n_episodes: Number of episodes to collect.

    Returns:
        Trajectories collected by the expert.
    """
    obs, _  = env.reset()
    masked_obs = masking_obs(obs)
    # obs = obs["unmasked"]  # Use unmasked observation for the expert
    done = False


    trajectory = {"obs": [], "acts": [], "infos": []}
    trajectory["obs"].append(masked_obs)
    while not done:
        if random() < beta:
            action, _ = bc_policy.predict(masked_obs)
        else:
            action, _ = expert_policy.predict(obs, deterministic=True)
        action = action.item()
        obs, reward, done, _, info = env.step(action)
        # obs = obs["unmasked"]  # Always use unmasked for the expert
        trajectory["acts"].append(action)
        trajectory["infos"].append(info)
        trajectory["obs"].append(masked_obs)
    trajectories.append(
        Trajectory(
            obs=np.array(trajectory["obs"]),
            acts=np.array(trajectory["acts"]),
            infos=trajectory["infos"],
            terminal=True,
        )
    )
    return trajectories

def evaluate_policy(policy, env, n_episodes=10, collect_failure=False):
    student_failed_states = []
    for episode in range(n_episodes):
        frames = []
        obs, _ = env.reset()
        frames.append(env.render())
        masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        states = [obs]
        while not done:
            action, _ = policy.predict(masked_obs, deterministic=True)
            action = action.item()
            obs, reward, done, _, info = env.step(action)
            masked_obs = masking_obs(obs)
            states.append(obs)
            frames.append(env.render())
        if reward == 0: #student failed
            student_failed_states = student_failed_states + states
        imageio.mimsave(f"./toy_student/dagger/testing_{episode}.gif", frames, duration=1 / 20, loop=0)
        writer = imageio.get_writer(f"./toy_student/dagger/testing_{episode}.mp4", fps=20)

        for im in frames:
            writer.append_data(im)

        writer.close()

    return student_failed_states
