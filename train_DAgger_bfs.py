from imitation.algorithms.bc import BC
from imitation.policies.base import FeedForward32Policy
from stable_baselines3.common.policies import ActorCriticPolicy
import json
from policies.custom_rollout import *
from cfg_utils import *
from omegaconf import DictConfig, OmegaConf
import hydra
from imitation.data.types import Trajectory
import numpy as np
from policies.bfs import BFS
from policies.BC import customBC
import torch

# from test_its import policy

frames = []
W = -1    # wall
O = 0     #open space
R = 1     #red object
B = 2     #blue object
G = 3     #goal
A = 4     #agent




def rollout_steps(env, model, max_steps=None, iseval=False, isStudent=True,full_visibility=False):
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
    bfs_path = []
    while len(bfs_path) == 0:
        obs = env.reset()
        if iseval:
            print("initial scence \n", obs)
        if isinstance(obs, tuple):  # Handle VecEnv API
            obs, _ = obs
        bfs_path =  bfs_shortest_path(env.map, env._starting_pos, env.goal_pos)



    trajectory = {"obs": [], "acts": [], "infos": []}
    obs_raw_array = []
    done = False
    step_count = 0
    frames.append(env.render())
    while not done:
        if isStudent:
            if full_visibility:
                action, _ = model.predict(obs, deterministic=True)  # Predict action
            else:
                masked_observation = masking_obs(obs)
                action, _ = model.predict(masked_observation, deterministic=True)

        else:
            if len(bfs_path) == 0: # use RL expert, collect online data
                action, _ = model.predict(obs, deterministic=True)  # Predict action
            
            else:
                action = bfs_path[step_count]

        # action = action  # Convert to scalar if necessary

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
        frames.append(env.render())
        # Stop if max_steps is specified and reached
        if max_steps and step_count >= max_steps:
            break

    # Append the final observation
    # trajectory["obs"].append(obs.copy())
    obs_raw_array.append(obs.copy())
    for obs in obs_raw_array:
        if full_visibility:
            trajectory["obs"].append(obs)
        else:
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
def collect_demonstrations(model, env, num_episodes=10, max_steps=None, iseval=False, full_visibility=False):
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
    print("expert trajectory collect")
    for i in range(num_episodes):
        trajectory = rollout_steps(env, model, max_steps, iseval, isStudent=False, full_visibility=full_visibility)
        trajectories.append(trajectory)
        print(f"{i} ", end=" ")

    return trajectories

@hydra.main(version_base=None, config_path="config", config_name="train_bc_config")
def train(cfg: DictConfig):
    map_array = load_map_from_example_dict(cfg.env.example_name)
    grid_class = GridNavigationCurriculumEnv

    cfg.logging.run_name = get_output_folder_name(cfg.log_folder)
    cfg.logging.run_path = get_output_path()
    episode_length = cfg.env.episode_length
    total_episode = cfg.bc_algo.total_episode
    visibility = cfg.bc_algo.visibility
    n_epoch = cfg.bc_algo.n_epochs
    batch = cfg.bc_algo.batch_size
    # logger.info(f"Logging to {cfg.logging.run_path}\nRun name: {cfg.logging.run_name}")
    goal_pos = np.argwhere(map_array == G)[0]
    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)

    env = grid_class(map_array=np.copy(map_array), goal_pos=goal_pos, render_mode="rgb_array",
                             episode_length=episode_length)
    env.curriculum= cfg.env.curriculum
    expert = BFS(env)
    flag_bc_init = True
    rng = np.random.default_rng(seed=42)
    trajectories = []
    eval_episode = 10
    if visibility == "full":
        full_visibility = True
    elif visibility == "occluded":
        full_visibility = False
    else:
        full_visibility = False
    if flag_bc_init:
        # trajectories = generate_unmasked_trajectories(pretrained_model, env)
        global frames
        trajectories = collect_demonstrations(model="bfs", env=env, num_episodes=total_episode, full_visibility=full_visibility)
        save_path = os.path.join(os.path.join(cfg.log_path, "eval"), "expert.gif")

        imageio.mimsave(save_path, frames[:100], duration=1 / 20, loop=0)
        bc = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            rng=rng,
            # policy=ActorCriticPolicy,
            # net_arch=[128,128],
        )

        # initialize BC policy
        bc.train(n_epochs=n_epoch)

        evaluation_bc, _ = evaluate_policy(bc.policy, env, eval_episode, full_visibility)
        print(f"BC policy: {evaluation_bc}")
        _, frames = evaluate_policy(bc.policy, env, 4)
        for i,frame in enumerate(frames):
            save_path = os.path.join(os.path.join(cfg.log_path,"eval"), f"BC_{i}.gif")
            imageio.mimsave(save_path, frame, duration=1 / 20, loop=0)                    #DAgger
        frames = []
        model_save_path = os.path.join(cfg.log_path, "BC")
        model_save_dir = os.path.dirname(model_save_path)
        os.makedirs(model_save_dir, exist_ok=True)

        # bc.policy.save(model_save_path)
        torch.save(bc.policy.state_dict(), model_save_path)

    else:
        bc = BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
            # policy=ActorCriticPolicy,
            # net_arch=[128,128],
        )
        evaluation_bc = {}
    for i in range(total_episode):
        print("#####\nIteration : ",i,"\n####")
        trajectories = collect_augmented_trajectories(trajectories, expert, bc.policy, env, full_visibility)
        bc.set_demonstrations(trajectories)
        bc.train(n_epochs=n_epoch)
        if i%100 == 0:
            model_save_path = os.path.join(cfg.log_path, f"DAgger_{i}")
            model_save_dir = os.path.dirname(model_save_path)
            os.makedirs(model_save_dir, exist_ok=True)

    evaluation_dagger, _ = evaluate_policy(bc.policy, env, eval_episode, full_visibility)
    print(f"BC policy: {evaluation_dagger}")
    _, frames= evaluate_policy(bc.policy, env, 4)
    for i,frame in enumerate(frames):
        save_path = os.path.join(os.path.join(cfg.log_path,"eval"), f"DAgger_{i}.gif")
        imageio.mimsave(save_path, frame, duration=1 / 20, loop=0)
    model_save_path = os.path.join(cfg.log_path, "DAgger")
    model_save_dir = os.path.dirname(model_save_path)
    os.makedirs(model_save_dir, exist_ok=True)

    # bc.policy.save(model_save_path)
    torch.save(bc.policy.state_dict(), model_save_path)

    print("Imitation learning agent saved!")

    evaluation={"BC":evaluation_bc, "DAgger":evaluation_dagger}
    save_path = os.path.join(os.path.join(cfg.log_path,"eval"), "evaluation.json")
    with open(save_path, "w") as f:
        json.dump(evaluation, f, indent=4)

@hydra.main(version_base=None, config_path="config", config_name="train_bc_config")
def test(cfg):
    grid_class = GridNavigationCurriculumEnv

    episode_length = cfg.env.episode_length
    goal_pos = np.argwhere(map_array == G)[0]

    env = grid_class(map_array=np.copy(map_array), goal_pos=goal_pos, render_mode="rgb_array",
                     episode_length=episode_length)
    env.curriculum = cfg.env.curriculum
    visibility = cfg.env.visibility
    if visibility == "full":
        full_visibility = True
    elif visibility == "occluded":
        full_visibility = False
    else:
        full_visibility = False
    policy_path = "/Users/yujinkim/Desktop/its_minigrid/toy_student/dagger_bfs/train_logs/empty_map3_visibility=occluded_level=2_2024-12-08-175456_nt=DAgger_InitBC_1000ep_10epoch"
    bc_policy_path = os.path.join(policy_path, "BC")
    bc = FeedForward32Policy(observation_space=env.observation_space,
                                 action_space=env.action_space,
                                 lr_schedule=lambda _: 0.001)
    bc.load_state_dict(torch.load(bc_policy_path,weights_only=True))
    dagger_policy_path =  os.path.join(policy_path, "DAgger")
    dagger = FeedForward32Policy(observation_space=env.observation_space,
                             action_space=env.action_space,
                             lr_schedule=lambda _: 0.001)
    dagger.load_state_dict(torch.load(dagger_policy_path,weights_only=True))
    # bc = FeedForward32Policy.load(policy_path)
    evaluation, frame_bc, frame_dagger = compare_policy(bc, dagger, env, 100, full_visibility)

    print(f"policy: {evaluation}")

    save_path = os.path.join(os.path.join(policy_path, "eval"), "testtime_evaluation.json")
    with open(save_path, "w") as f:
        json.dump(evaluation, f, indent=4)

    for i, frame in enumerate(frames[evaluation["BC"]["wins"]]):
        save_path = os.path.join(os.path.join(policy_path, "eval"), f"bc_wins{i}.gif")
        imageio.mimsave(save_path, frame, duration=1 / 20, loop=0)
    for i, frame in enumerate(frames[evaluation["DAgger"]["wins"]]):
        save_path = os.path.join(os.path.join(policy_path, "eval"), f"dagger_wins{i}.gif")
        imageio.mimsave(save_path, frame, duration=1 / 20, loop=0)


if __name__ == "__main__":
    # train_or_sweep()
    # test()
    train()

