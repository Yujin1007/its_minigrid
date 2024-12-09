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

def collect_augmented_trajectories(trajectories, expert_policy, bc_policy, env, beta=0.8, full_visibility=False):
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
        if full_visibility:
            action_bc = bc_policy.predict(obs)
        else:
            action_bc = bc_policy.predict(masked_obs)
        action = expert_policy.predict(obs, deterministic=True)
        if action == 4: # stuck
            print("stuck done")
            break
        if isinstance(action, tuple):  # Handle VecEnv API
            action, _ = action
            action = action.item()
        if isinstance(action_bc, tuple):  # Handle VecEnv API
            action_bc, _ = action_bc
            action_bc = action_bc.item()

        if action_bc != action:
            dumb_env = copy.deepcopy(env)
            obs, _, _, _, info = dumb_env.step(action)
            _, reward, done, _, _ = env.step(action_bc)
        else:
            obs, reward, done, _, info = env.step(action)
        masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Always use unmasked for the expert
        trajectory["acts"].append(action)
        trajectory["infos"].append(info)
        if full_visibility:
            trajectory["obs"].append(masked_obs.copy())
        else:
            trajectory["obs"].append(obs.copy())
    trajectories.append(
        Trajectory(
            obs=np.array(trajectory["obs"]),
            acts=np.array(trajectory["acts"]),
            infos=trajectory["infos"],
            terminal=True,
        )
    )
    return trajectories

def evaluate_policy(policy, env, n_episodes=10, collect_failure=False, full_visibility=False):
    # student_failed_states = []
    frame = []
    success_cnt = 0
    for episode in range(n_episodes):
        frames = []
        obs, _ = env.reset()
        frames.append(env.render())
        if full_visibility:
            masked_obs = obs
        else:
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
        # if reward == 0: #student failed
        #     student_failed_states = student_failed_states + states
        if info["goal"]:
            success_cnt += 1

        # imageio.mimsave(f"./toy_student/dagger/testing_{episode}.gif", frames, duration=1 / 20, loop=0)
        # writer = imageio.get_writer(f"./toy_student/dagger/testing_{episode}.mp4", fps=20)
        #
        # for im in frames:
        #     writer.append_data(im)
        #
        # writer.close()
        frame.append(frames)
    evaluation={
        "success_cnt": success_cnt,
        "fail_cnr": n_episodes - success_cnt,
        "success_rate": success_cnt/n_episodes*100,
    }
    return evaluation, frame


def compare_policy(bc, dagger, env, n_episodes=10, full_visibility=False):
    frame1 = []
    frame2 = []
    success_cnt1 = 0
    success_cnt2 = 0
    bc_wins = []
    dagger_wins = []
    for episode in range(n_episodes):
        frames1 = []
        frames2 = []
        bc_succeed = False
        dagger_succeed = False

        obs, _ = env.reset()
        env2 = copy.deepcopy(env)
        obs2 = copy.deepcopy(obs)
        frames1.append(env.render())
        if full_visibility:
            masked_obs = obs
        else:
            masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        states = [obs]
        while not done:
            action, _ = bc.predict(masked_obs, deterministic=True)
            action = action.item()
            obs, reward, done, _, info = env.step(action)
            masked_obs = masking_obs(obs)
            states.append(obs)
            frames1.append(env.render())
        # if reward == 0: #student failed
        #     student_failed_states = student_failed_states + states
        if info["goal"]:
            success_cnt1 += 1
            bc_succeed = True
        frame1.append(frames1)

        frames2.append(env2.render())
        obs = obs2
        if full_visibility:
            masked_obs = obs
        else:
            masked_obs = masking_obs(obs)
        # obs = obs["unmasked"]  # Use unmasked observation for the expert
        done = False
        states = [obs]
        while not done:
            action, _ = dagger.predict(masked_obs, deterministic=True)
            action = action.item()
            obs, reward, done, _, info = env2.step(action)
            masked_obs = masking_obs(obs)
            states.append(obs)
            frames2.append(env2.render())
        # if reward == 0: #student failed
        #     student_failed_states = student_failed_states + states
        if info["goal"]:
            success_cnt2 += 1
            dagger_succeed = True
        frame2.append(frames2)
        if bc_succeed and not dagger_succeed:
            bc_wins.append(episode)
        if dagger_succeed and not bc_succeed:
            dagger_wins.append(episode)
    evaluation={
        "BC":{
            "success_cnt": success_cnt1,
            "fail_cnr": n_episodes - success_cnt1,
            "success_rate": success_cnt1 / n_episodes * 100,
            "wins": bc_wins
        },
        "DAgger": {
            "success_cnt": success_cnt2,
            "fail_cnr": n_episodes - success_cnt2,
            "success_rate": success_cnt2 / n_episodes * 100,
            "wins": dagger_wins
        },

    }
    return evaluation, frame1, frame2