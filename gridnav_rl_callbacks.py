import os
from typing import Any, Dict, Optional
import imageio
import gymnasium
import wandb
import torch as th
import numpy as np
from numpy import array
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import Image as WandbImage
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback
from imitation.data.types import Trajectory
from PIL import Image, ImageDraw, ImageFont
from toy_envs.toy_env_utils import masking_obs
import matplotlib.pyplot as plt

from loguru import logger
import csv
import time

from toy_envs.toy_env_utils import update_location, render_map_and_agent
from stable_baselines3.common.callbacks import EvalCallback
W = -1    # wall
O = 0     #open space
R = 1     #red object
B = 2     #blue object
G = 3     #goal
A = 4     #agent

def append_to_csv(items, item_headers, filename):
    """
    Logs a set of items with corresponding headers to a CSV file.
    Creates the file with headers if it doesn't exist.

    """

    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write headers if new file
        if not file_exists:
            writer.writerow(item_headers)

        # Write the new row
        writer.writerow(items)

def convert_obs_to_frames(map_array, obs):
    """
    Frames is represented as map_array with the agent's position marked by 1

    obs represents the agent's (x, y) position in the grid

    Parameters:
        map_array: np.ndarray (row_size, col_size)
        obs: np.ndarray (n_timesteps, 2)

    Returns:
        frames: np.ndarray (n_timesteps, row_size, col_size)
    """
    frames = []
    for i in range(len(obs)):
        frame = np.copy(map_array)
        frame[int(obs[i][0]), int(obs[i][1])] = 1
        frames.append(frame)

    return np.array(frames)

    
def convert_obs_history_to_frames(map_array, obs):
    """
    Frames is represented as map_array with the agent's position marked by 1

    obs represents the agent's (x, y) position in the grid

    Parameters:
        map_array: np.ndarray (row_size, col_size)
        obs: np.ndarray (n_timesteps, 2)

    Returns:
        frames: np.ndarray (n_timesteps, row_size, col_size)
    """
    frames = []
    for i in range(len(obs)):
        frame = np.copy(map_array)
        frame[np.nonzero(obs[i]==1)] = 1
        frames.append(frame)

    return np.array(frames)

# class GridNavSeqRewardCallback(BaseCallback):
#     """
#     Custom callback for calculating seq matching reward in the GridNavigation environment.
#     """
#     def __init__(self, map_array, verbose=0, use_history=False):
#         super(GridNavSeqRewardCallback, self).__init__(verbose)
#
#         self._map = map_array
#
#         # self._ref_seq = ref_seq
#
#         logger.info(f"[GridNavSeqRewardCallback] Loaded reference sequence.")
#
#         # self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, cost_fn_name)
#         self.use_history = use_history
#
#     def on_rollout_end(self) -> None:
#         """
#         This method is called after the rollout ends.
#         You can access and modify the rewards in the ReplayBuffer here.
#         """
#         if self.use_history:
#             self.on_rollout_end_history()
#         else:
#             self.on_rollout_end_mdp()
#
#     def on_rollout_end_history(self):
#         matching_reward_list = []
#         for env_i in range(self.model.env.num_envs):
#             obs_to_use = self.model.rollout_buffer.observations[1:, env_i]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
#
#             final_obs = np.reshape(obs_to_use[-1], self._map.shape)
#
#             cur_pos_np = np.nonzero(final_obs == 2)
#             agent_pos = np.array([cur_pos_np[0][0], cur_pos_np[1][0]])
#
#             final_location, _, _ = update_location(agent_pos=agent_pos.astype(np.int64), action=int(self.model.rollout_buffer.actions[-1, env_i]), map_array=self._map)
#
#             # final_obs = update_obs(final_obs, prev_agent_pos=agent_pos,new_agent_pos = final_location ).flatten()
#             # obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
#             #
#             # frames = convert_obs_to_frames(self._map, obs_to_use)
#
#             # matching_reward, _ = self._matching_fn(frames, self._ref_seq)  # size: (n_steps,)
#
#             # matching_reward_list.append(matching_reward)
#
#         # rewards = np.stack(matching_reward_list, axis=1)  # size: (n_steps, n_envs)
#
#         # self.model.rollout_buffer.rewards += rewards
#
#     def on_rollout_end_mdp(self) -> None:
#         pass
#         matching_reward_list = []
#         # for env_i in range(self.model.env.num_envs):
#         #     obs_to_use = self.model.rollout_buffer.observations[1:, env_i]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
#         #     # Keep track of the history. Agent cannot revisit the same
#         #     history = [obs_to_use[i].tolist() for i in range(len(obs_to_use))]
#         #     final_obs = update_location(agent_pos=obs_to_use[-1].astype(np.int64), action=int(self.model.rollout_buffer.actions[-1, env_i]), map_array=self._map, history=history)
#             # obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
#             # frames = convert_obs_to_frames(self._map, obs_to_use)
#             # matching_reward, _ = self._matching_fn(frames, self._ref_seq)  # size: (n_steps,)
#             # matching_reward_list.append(matching_reward)
#
#             # for i in range(len(frames)):
#             #     print(f"[{i}] action={self.model.rollout_buffer.actions[i, env_i]} matching_reward={matching_reward[i]}")
#             #     if i > 0:
#             #         print(f"before:\n{frames[i-1]}")
#             #     print(f"after:\n{frames[i]}")
#             #     input("stop")
#
#         # rewards = np.stack(matching_reward_list, axis=1)  # size: (n_steps, n_envs)
#
#         # self.model.rollout_buffer.rewards += rewards
#
#         # print(f"GridNavSeqRewardCallback took {time.time() - start_time} seconds")
#
#
#     def on_rollout_end_no_buffer(self, states, actions, rewards):
#         obs_to_use = states[1:]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
#         final_obs,_,_ = update_location(agent_pos=obs_to_use[-1].astype(np.int64), action=int(actions[-1]), map_array=self._map)
#         obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
#         frames = convert_obs_to_frames(self._map, obs_to_use)
#
#         # matching_rewards, _ = self._matching_fn(frames, self._ref_seq)  # size: (n_steps,)
#
#         # return rewards + matching_rewards
#         return rewards
#
#     def _on_step(self) -> bool:
#         """
#         Just need to define this method to avoid NotImplementedError
#
#         Return:
#             If the callback returns False, training is aborted early.
#         """
#         return True


def plot_info_on_frame(pil_image, info, font_size=20):
    """
    Parameters:
        pil_image: PIL.Image
            The image to plot the info on
        info: Dict
            The information to plot on the image
        font_size: int
            The size of the font to use for the text
    
    Effects:
        pil_image is modified to include the info
    """
    # TODO: this is a hard-coded path
    font = ImageFont.truetype("./arial.ttf/arial.ttf", font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = pil_image.height - font_size  # Beginning of the y position of the text
    
    i = 0
    for k in info:
        # TODO: This is pretty ugly
        if not any([text in k for text in ["TimeLimit", "render_array", "geom_xpos", "episode"]]):
            reward_text = f"{k}:{info[k]}"
            # Plot the text from bottom to top
            text_position = (x, y - (font_size + 10)*(i+1))
            draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1


class GridNavVideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gymnasium.Env,
        rollout_save_path: str,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        map_array: np.ndarray = None,
        goal_pos: np.ndarray = None,
        use_history: bool = False,
        verbose=0
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to
        TensorBoard

        Pararmeters
            eval_env: A gym environment from which the trajectory is recorded
                Assumes that there's only 1 environment
            rollout_save_path: The path to save the rollouts (states and rewards)
            render_freq: Render the agent's trajectory every eval_freq call of the callback.
            n_eval_episodes: Number of episodes to render
            deterministic: Whether to use deterministic or stochastic policy
            goal_seq_name: The name of the reference sequence to compare with (This defines the unifying metric that all approaches attempting to solve the same task gets compared against)
            seq_name: The name of the reference sequence to compare with
                You only need to set this if you want to calculate the OT reward
            matching_fn_cfg: The configuration for the matching function
        """
        super().__init__(verbose)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._rollout_save_path = rollout_save_path  # Save the state of the environment

        self._map = map_array
        self._goal_pos = goal_pos
        # self._ref_seq = ref_seq
        logger.info(f"[GridNavVideoRecorderCallback] Loaded reference sequence.")


        # self._gt_reward_fn = self.set_ground_truth_fn()

        self._calc_matching_reward = False
        # if matching_fn_cfg != {}:
        #     self._calc_matching_reward = True
        #     self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, cost_fn_name)
        # else:
        #     self._calc_matching_reward = False

        self.use_history = use_history

    # def set_ground_truth_fn(self):
    #     """
    #     Set the ground truth function for the matching function
    #     """
    #     def nav_key_point_following(obs_seq, ref_seq):
    #         """
    #         Counting the number of key points that the agent has followed
    #         """
    #         score = 0
    #         j = 0
    #
    #         score_at_each_timestep = []
    #
    #         for i in range(len(obs_seq)):
    #             if j < len(ref_seq):
    #                 if np.array_equal(obs_seq[i], ref_seq[j]):
    #                     score += 1
    #                     j += 1
    #
    #             score_at_each_timestep.append(score/len(ref_seq))
    #
    #         return score_at_each_timestep
    #
    #     return nav_key_point_following

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            raw_screens = []
            screens = []
            states = []
            infos = []
            rewards = []
            actions = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in
                the captured `screens` list

                :param _locals: A dictionary containing all local variables of the
                 callback's scope
                :param _globals: A dictionary containing all global variables of the
                 callback's scope
                """
                screen = self._eval_env.render()
                image_int = np.uint8(screen)

                raw_screens.append(Image.fromarray(image_int))
                screens.append(Image.fromarray(image_int))  # The frames here will get plotted with info later
                infos.append(_locals.get('info', {}))

                states.append(_locals["observations"])
                rewards.append(_locals["rewards"])
                actions.append(_locals["actions"])

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            # Originally, states is a list of np.arrays size (1, env_obs_size)
            #   We want to concatenate them to get a single np.array size (n_timesteps, env_obs_size)
            states = np.concatenate(states)
            actions = np.concatenate(actions)

            # obs_to_use = states[1:]  # Skip the first observation because we are calculating the reward based on what it looks like in the next state
            # agent_pos = np.argwhere(obs_to_use[-1] == A)[0]
            agent_pos = np.argwhere(states[-1] == A)[0]
            goal_pos = np.argwhere(states[0] == G)[0]
            # print("argwhere agent pos:", agent_pos)
            # final_obs, _, _ = update_location(agent_pos=agent_pos, action=int(actions[-1]),
            #                                   map_array=obs_to_use[-1], goal=self._goal_pos)

            final_obs, final_map, is_goal = update_location(agent_pos=agent_pos, action=int(actions[-1]),
                                              map_array=states[-1], goal=self._goal_pos)
            if is_goal:
                terminal_map = np.ones_like(final_map) * -1
                terminal_map[final_obs[0], final_obs[1]] = A
                terminal_map[goal_pos[0], goal_pos[1]] = R
                final_map = terminal_map

            # if self.use_history:
            #     final_obs = np.reshape(obs_to_use[-1], self._map.shape)
            #     cur_pos_np = np.nonzero(final_obs == 2)
            #     agent_pos = np.array([cur_pos_np[0][0], cur_pos_np[1][0]])
            #     final_location, _, _ = update_location(agent_pos=agent_pos.astype(np.int64), action=int(actions[-1]), map_array=self._map, goal=self._goal_pos)
            #     final_obs = update_obs(final_obs, prev_agent_pos=agent_pos,new_agent_pos = final_location ).flatten()
            #     obs_seq = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
            # else:
            #     # Keep track of the history. Agent cannot revisit the same
            #     history = [obs_to_use[i].tolist() for i in range(len(obs_to_use))]
            #     agent_pos = np.argwhere(obs_to_use[-1] == A)[0]
            #     final_obs,_, _ = update_location(agent_pos=agent_pos, action=int(actions[-1]), map_array=self.obs_to_use[-1], goal=self._goal_pos)
                # new_pos, map_array, is_goal
                # obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
                # obs_seq = convert_obs_to_frames(self._map, obs_to_use)

            # Add the first frame and replace final frame to the screens
            raw_screens = [Image.fromarray(np.uint8(render_map_and_agent(states[0], states[0].astype(np.int64))))] + raw_screens
            # raw_screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))
            # screens = [Image.fromarray(np.uint8(render_map_and_agent(self._map, states[0].astype(np.int64))))] + screens
            # screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))
            # raw_screens.insert(0, raw_screens[-1])
            raw_screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(final_map, final_obs)))
            # raw_screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(states[-1], final_obs)))
            # Save the raw_screens locally
            imageio.mimsave(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts__.gif"), raw_screens, duration=1/30, loop=0)
            
            # gt_rewards = self._gt_reward_fn(obs_seq, self._ref_seq)


            for i in range(len(infos)):
                infos[i]["gt_r"] = f"{sum(rewards[i]):.4f}"

            self.logger.record("rollout/episode_reward", 
                                np.max(rewards),
                                exclude=("stdout", "log"))

            append_to_csv([np.max(rewards), self.num_timesteps], ["ordered_target_frames_achieved", "timestep"], os.path.join(self._rollout_save_path,f"performance.csv"))

            # self.logger.record("rollout/mean_gt_reward_per_epsisode",
            #                     np.mean(rewards),
            #                     exclude=("stdout", "log", "json", "csv"))


            # Plot info on the frames  
            for i in range(1, len(screens)):
                plot_info_on_frame(screens[i], infos[i-1])

            screens = [np.uint8(s).transpose(2, 0, 1) for s in screens]

            # Log to wandb
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(array([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )

            # Save the rollouts locally    
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_states.npy"), "wb") as f:
                np.save(f, np.array(states))
            
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_rewards.npy"), "wb") as f:
                np.save(f, np.array(rewards))

        return True


    def _on_step_no_buffer(self, raw_screens, screens, states, actions, rewards, infos, num_timesteps) -> bool:
        obs_to_use = states[0:]
        agent_pos = np.argwhere(obs_to_use[-1] == A)[0]
        # print("argwhere agentpos:",agent_pos)
        final_obs, _, _ = update_location(agent_pos=agent_pos, action=int(actions[-1]),
                                          map_array=obs_to_use[-1], goal=self._goal_pos)

        obs_to_use = np.concatenate([obs_to_use, np.expand_dims(final_obs, 0)], axis=0)
        obs_seq = convert_obs_to_frames(self._map, obs_to_use)

        # Add the first frame and replace final frame to the screens
        # raw_screens = [Image.fromarray(np.uint8(render_map_and_agent(self._map, states[0].astype(np.int64))))] + raw_screens
        # raw_screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))
        # screens = [Image.fromarray(np.uint8(render_map_and_agent(self._map, states[0].astype(np.int64))))] + screens
        # screens[-1] = Image.fromarray(np.uint8(render_map_and_agent(self._map, final_obs)))

        # Save the raw_screens locally
        imageio.mimsave(os.path.join(self._rollout_save_path, f"{num_timesteps}_rollouts.gif"), raw_screens, duration=1/30, loop=0)
        
        # gt_rewards = self._gt_reward_fn(obs_seq, self._ref_seq)
        #
        # for i in range(len(infos)):
        #     infos[i]["gt_r"] = f"{gt_rewards[i]:.4f}"
        #
        log_dict = {"rollout/mean_gt_reward_per_epsisode": np.mean(rewards)}


        # Plot info on the frames  
        for i in range(1, len(screens)):
            plot_info_on_frame(screens[i], infos[i-1])

        screens = [np.uint8(s).transpose(2, 0, 1) for s in screens]

        # Log to wandb
        log_dict["trajectory/video"] = wandb.Video(th.ByteTensor(array([screens])), fps=40)

        # Save the rollouts locally    
        # with open(os.path.join(self._rollout_save_path, f"{num_timesteps}_rollouts_states.npy"), "wb") as f:
        #     np.save(f, np.array(states))
        #
        # with open(os.path.join(self._rollout_save_path, f"{num_timesteps}_rollouts_rewards.npy"), "wb") as f:
        #     np.save(f, np.array(rewards))

        wandb.log(log_dict, step=num_timesteps)

        return True
    def on_training_epoch_end(self, locals_dict, globals_dict):
        """
        Save video or log progress at the end of each training epoch.

        Parameters:
            locals_dict: dict
                Local variables from the training loop.
            globals_dict: dict
                Global variables from the training loop.
        """
        # Example: Save a sample video or log
        epoch = locals_dict.get("epoch", "N/A")
        print(f"Epoch {epoch} ended. GridNavVideoRecorderCallback invoked.")
        # Optional: Add custom logic for saving videos or logging

class WandbCallback(SB3WandbCallback):
    def __init__(
        self,
        model_save_path: str,
        model_save_freq: int,
        **kwargs,
    ):
        super().__init__(
            model_save_path=model_save_path,
            model_save_freq=model_save_freq,
            **kwargs,
        )

    def save_model(self) -> None:
        model_path = os.path.join(
        self.model_save_path, f"model_{self.model.num_timesteps}_steps.zip"
        )
        self.model.save(model_path)

    def on_training_epoch_end(self, locals_dict, globals_dict):
        """
        Custom behavior for Wandb logging at the end of each training epoch.

        Parameters:
            locals_dict: dict
                Local variables from the training loop.
            globals_dict: dict
                Global variables from the training loop.
        """
        # Example: Log metrics to Wandb
        if "epoch" in locals_dict:
            wandb.log({"epoch": locals_dict["epoch"]})
        print(f"Epoch {locals_dict.get('epoch', 'N/A')} ended. WandbCallback invoked.")



class EpisodeTerminationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeTerminationCallback, self).__init__(verbose)
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.episode_rewards = []  # Store rewards for each episode
        self.episode_lengths = []  # Store steps for each completed episode

    def _on_step(self) -> bool:
        # `done` indicates if an episode is finished
        done = self.locals.get("dones", [False])[0]
        reward = self.locals.get("rewards", [0])[0]

        # Accumulate rewards for the current episode
        self.current_episode_reward += reward
        self.current_episode_steps += 1
        if done:
            # Episode is terminated; log the reward or perform any action
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_steps)
            self.logger.record("episode/episode_reward",
                               self.current_episode_reward,
                               exclude=("stdout", "log"))
            self.logger.record("episode/episode_length",
                               self.current_episode_steps,
                               exclude=("stdout", "log"))
            # Reset step counter for the next episode
            self.current_episode_steps = 0
            # Reset the current episode reward
            self.current_episode_reward = 0

        return True  # Returning False will stop training

    def on_training_epoch_end(self, locals_dict, globals_dict):
        """
        Save video or log progress at the end of each training epoch.

        Parameters:
            locals_dict: dict
                Local variables from the training loop.
            globals_dict: dict
                Global variables from the training loop.
        """
        # Example: Save a sample video or log
        epoch = locals_dict.get("epoch", "N/A")
        print(f"Epoch {epoch} ended. GridNavVideoRecorderCallback invoked.")
        # Optional: Add custom logic for saving videos or logging




class RLBCTrainingCallback(BaseCallback):
    """
    Custom callback to integrate RL training (PPO) and BC training.

    - Collects trajectories from the RL agent every `bc_training_interval` timesteps.
    - Trains the BC model with collected trajectories.
    - Saves the BC model after training.
    """

    def __init__(self,
                 eval_env,
                 bc_model,
                 bc_training_interval,
                 bc_epochs,
                 bc_batch_size,
                 bc_save_path,
                 run_path,
                 verbose=0):
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
        super().__init__(verbose)
        self.eval_env = eval_env
        self.bc_model = bc_model
        self.bc_training_interval = bc_training_interval
        self.bc_epochs = bc_epochs
        self.bc_batch_size = bc_batch_size
        self.bc_save_path = bc_save_path
        self.run_path = run_path
        self.frames = []
        self.episodic_frames = []

    def _rollout_steps(self, isStudent=False):
        """
        Perform a single rollout of the environment and return a trajectory.

        Parameters:
            model: Trained model to predict actions.
            env: Environment to interact with.
            max_steps: Optional; maximum number of steps for the rollout.

        Returns:
            A single trajectory containing observations, actions, and metadata.
        """
        obs = self.eval_env.reset()
        if isinstance(obs, tuple):  # Handle VecEnv API
            obs, _ = obs

        trajectory = {"obs": [], "acts": [], "infos": []}
        obs_raw_array = []
        done = False
        self.episodic_frames = [self.eval_env.render()]
        while not done:
            if isStudent:
                masked_observation = masking_obs(obs)
                action, _ = self.bc_model.policy.predict(masked_observation, deterministic=True)  # Predict action
                trajectory["obs"].append(obs.copy())
            else:
                action, _ = self.model.predict(obs, deterministic=True)  # Predict action
                obs_raw_array.append(obs.copy())

            action = action.item()  # Convert to scalar if necessary
            # Append current state and action
            # trajectory["obs"].append(obs.copy())
            # obs_raw_array.append(obs.copy())
            trajectory["acts"].append(action)

            # Take an environment step

            obs, reward, done, _, info = self.eval_env.step(action)
            self.episodic_frames.append(self.eval_env.render())
            # callback._on_step().
            if isinstance(obs, tuple):  # Handle VecEnv API
                obs, info = obs

            trajectory["infos"].append(info)


        # Append the final observation
        # trajectory["obs"].append(obs.copy())
        if not isStudent:
            obs_raw_array.append(obs.copy())
            for obs in obs_raw_array:
                masked_observation = masking_obs(obs)
                trajectory["obs"].append(masked_observation)
        else:
            trajectory["obs"].append(obs.copy())
        self.frames.append(self.episodic_frames)
        self.episodic_frames = []
        return Trajectory(
            obs=np.array(trajectory["obs"]),
            acts=np.array(trajectory["acts"]),
            infos=trajectory["infos"],
            terminal=True,
        )

    def _collect_demonstrations(self,num_episodes=30, isStudent=False):
        """
        Collect multiple trajectories (demonstrations) using the provided model.

        Parameters:
            num_episodes: Number of episodes to collect.
        Returns:
            A list of Trajectory objects.
        """
        self.frames = []
        self.episodic_frames = []
        trajectories = []
        # for _ in range(num_episodes):
        i=0
        cnt_iter = 0
        while i < num_episodes:
            trajectory = self._rollout_steps(isStudent=isStudent)
            if isStudent: #student demonstration. collect only failed demonstration
                if not trajectory.infos[-1]["goal"]:
                    trajectories.append(trajectory.obs)
                    i+=1
            else: # teacher demonstration. collect any demonstration.
                trajectories.append(trajectory)
                i+=1
            cnt_iter += 1
            if cnt_iter >= 500:
                print(f"Among 500 run, {i} failed cases, consider BC train is over. ")
                break
        return trajectories
    def _train_bc(self, trajectory):
        """
        Trains the BC model using collected trajectories.
        """
        # Prepare dataset
        self.bc_model.set_demonstrations(trajectory)

        self.bc_model.train(n_epochs=self.bc_epochs)

        # Save BC model
        model_name = "bc_model_"+str(self.num_timesteps)+".pth"
        model_path = os.path.join(self.bc_save_path, model_name)
        self.bc_model.policy.save(model_path)
        student_states = self._collect_demonstrations(isStudent=True, num_episodes=10)

        return student_states

    def _save_frames(self):
        save_path = os.path.join(self.bc_save_path, str(self.num_timesteps))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, frames in enumerate(self.frames):
            save_name = os.path.join(save_path, f"{i}.gif")
            imageio.mimsave(save_name, frames, duration=1 / 20, loop=0)
    def _save_student_states(self, states):
        file_name = os.path.join(self.run_path, "student_states.npz")
        np.savez(file_name, *states)
    def _on_step(self):
        """
        Triggered at every environment step.
        """
        # Check if it's time to train the BC model
        if self.num_timesteps % self.bc_training_interval == 0:
            if self.verbose > 0:
                print(f"Collecting trajectory for BC training at timestep {self.num_timesteps}")

            # Collect trajectory
            trajectory = self._collect_demonstrations(isStudent=False)

            # Train BC model
            if self.verbose > 0:
                print("Training BC model...")
            student_states = self._train_bc(trajectory)
            self._save_frames()
            self._save_student_states(student_states)

            self.training_env.env_method("_set_initial_states", student_states)

            self.eval_env._set_initial_states(student_states)


        return True

class Curriculum(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    parent: EvalCallback

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        continue_training = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because the mean reward {self.parent.best_mean_reward:.2f} "
                f" is above the threshold {self.reward_threshold}"
            )
        return continue_training

# class SetInitialStatesCallback(BaseCallback):
#     def __init__(self, param_name, new_value, bc_training_interval, verbose=0):
#         """
#         :param param_name: Name of the environment parameter to modify.
#         :param new_value: Value to set for the parameter.
#         :param trigger_step: Number of steps between updates.
#         :param verbose: Verbosity level.
#         """
#         super().__init__(verbose)
#         self.param_name = param_name
#         self.new_value = new_value
#         self.bc_training_interval = bc_training_interval
#
#     def _on_step(self) -> bool:
#         # Check if the current step is a multiple of trigger_step
#         if self.num_timesteps % self.bc_training_interval == 0:
#             # Use env_method to call `set_param` on all environments
#             self.training_env.env_method("set_param", self.param_name, self.new_value)
#             if self.verbose > 0:
#                 print(f"Updated '{self.param_name}' to {self.new_value} at step {self.num_timesteps}")
#         return True