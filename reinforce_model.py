import torch
import time
import wandb
import sys

import numpy as np
from stable_baselines3.common.utils import safe_mean
from tqdm import tqdm
from PIL import Image


# A simple MLP policy for REINFORCE
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.fc2 = torch.nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
    

class REINFORCE(object):
    def __init__(self,
                 env,
                 n = 10,
                 learning_rate=0.01,
                 ent_coef=0.01,
                 gamma=0.99,
                 policy_gradient=False,
                 use_relative_reward=False,
                 video_save_freq=500):
        """
        policy_gradient (bool): whether to use sum of reward or reward to go
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MLP(env.observation_space.shape[0], env.action_space.n).to(self.device)

        self.env = env

        self.n = n

        self.last_video_save_step = 0
        self.video_save_freq = video_save_freq

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.ent_coef = ent_coef
        self.gamma = gamma

        self.policy_gradient = policy_gradient
        self.use_relative_reward = use_relative_reward
        
    def learn(self, total_timesteps, progress_bar=True, callback=[]):
        self.start_time = time.time()

        self.num_timesteps = 0

        print(f"Training for {total_timesteps} timesteps, {total_timesteps // self.env.episode_length} episodes")   

        for i in tqdm(range(0, total_timesteps, self.env.episode_length * self.n)):
            # Collect rollouts
            #   TODO: hack to assume that the last callback is the seq_matching_callback
            rollout_dicts = []
            
            for _ in range(self.n):
                rollout_dict = self.collect_rollouts(self.env, callback[-1])
                rollout_dicts.append(rollout_dict)

            self.num_timesteps += self.env.episode_length

            total_loss, policy_loss, entropy_loss = self.train(rollout_dicts)

            rewards_for_all_rollouts = np.array([np.mean(rollout_dict["rewards"]) for rollout_dict in rollout_dicts])

            print(f"[{i}] avg reward: {np.mean(rewards_for_all_rollouts)}, l={total_loss:.2f}, pol={policy_loss:.2f}, ent={entropy_loss:.2f}, {rollout_dicts[0]['all_log_probs'][-1].data}")

            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)

            wandb.log({"time/iterations": i,
                    "rollout/ep_rew_mean": np.mean(rewards_for_all_rollouts),
                    "rollout/ep_rew_sum": np.sum(rewards_for_all_rollouts)/self.n,
                    "time/time_elapsed": int(time_elapsed),
                    "train/total_loss": total_loss,
                    "train/policy_loss": policy_loss,
                    "train/entropy_loss": entropy_loss,
                    }, step=self.num_timesteps)
            
            if self.last_video_save_step == 0 or (self.num_timesteps - self.last_video_save_step) >= self.video_save_freq:
                self.evaluate_policy(self.env, 
                                     seq_matching_callback=callback[-1], 
                                     video_save_callback=callback[-2])
                
                self.last_video_save_step = self.num_timesteps

    def train(self, rollout_dicts):
        for i in range(len(rollout_dicts)):
            rollout_dict = rollout_dicts[i]

            log_probs = torch.stack(rollout_dict["log_probs"])
            rewards = torch.tensor(rollout_dict["rewards"]).to(self.device)
            all_log_probs = torch.stack(rollout_dict["all_log_probs"])
            
            # Policy loss
            #   -log(p(a|s)) * R(tau)
            if self.policy_gradient:
                # Compute rewad to go:
                #   R(t) = sum_{t'=t}^{T} r_{t'}
                rewards_to_go = torch.zeros_like(rewards)
                rewards_to_go[-1] = rewards[-1]  # Initialize the last reward to go

                # Iterate backward to compute the reward to go in a cumulative manner
                for t in reversed(range(len(rewards) - 1)):
                    rewards_to_go[t] = rewards[t] + self.gamma * rewards_to_go[t + 1]
                
                policy_loss = - torch.sum(log_probs * rewards_to_go)
            else:
                policy_loss = - torch.sum(log_probs) * torch.sum(rewards)

            # Entropy loss
            #   H(p) = -sum(p(x) * log(p(x)))
            entropy_loss = torch.sum(torch.exp(all_log_probs) * all_log_probs)

            if i == 0:
                total_policy_loss = policy_loss
                total_entropy_loss = entropy_loss
                total_loss = policy_loss + self.ent_coef * entropy_loss
            else:
                total_policy_loss += policy_loss
                total_entropy_loss += entropy_loss
                total_loss += policy_loss + self.ent_coef * entropy_loss
        
        total_policy_loss /= len(rollout_dicts)  # Find the average policy loss
        total_entropy_loss /= len(rollout_dicts)  # Find the average entropy loss
        total_loss /= len(rollout_dicts)  # Find the average total loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), total_policy_loss.item(), total_entropy_loss.item()
    
    def collect_rollouts(self, env, seq_matching_callback, deterministic=False, save_video=False):
        state, _ = env.reset()
        done = False
        log_probs = []
        all_log_probs = []
        rewards = []
        states = []
        actions = []
        infos = []

        raw_screens = []
        screens = []

        check_rollout = False
        
        while not done:
            # Select action
            state = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(self.device)
            probs = self.policy(state)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()  # Same as categorical distribution
            log_prob = torch.log(probs[:, action])
            all_log_probs.append(torch.log(probs))

            # Take step
            next_state, reward, done, _, info = env.step(action)

            if save_video:
                screen = env.render()
                raw_screens.append(Image.fromarray(np.uint8(screen)))
                screens.append(Image.fromarray(np.uint8(screen)))
                infos.append(info)

            rewards.append(reward)
            log_probs.append(log_prob)
            states.append(state.cpu().numpy())
            actions.append(action)
            state = next_state

            if state[0] > 0 and state[0] < 2 and state[1] == 4:
                check_rollout = True

        states = np.concatenate(states)

        seq_matching_rewards = seq_matching_callback.on_rollout_end_no_buffer(np.array(states), np.array(actions), np.array(rewards))

        if self.use_relative_reward:
            # Augment the rewards so that the rewards are based on differences between the raw rewards
            seq_matching_rewards_aug = np.zeros_like(seq_matching_rewards)
            seq_matching_rewards_aug[0] = seq_matching_rewards[0]
            for i in range(1, len(seq_matching_rewards)):
                seq_matching_rewards_aug[i] = seq_matching_rewards[i] - seq_matching_rewards[i-1]

            seq_matching_rewards = seq_matching_rewards_aug

        if check_rollout:
            print("States, Actions (0=DOWN, 1=RIGHT, 2=UP, 3=LEFT, 4=Stay)")
            print(states)
            print(actions)
            print(seq_matching_rewards)
            input("stop")

        return {
            "log_probs": log_probs,
            "all_log_probs": all_log_probs,
            "raw_rewards": rewards,
            "rewards": seq_matching_rewards,
            "states": states,
            "actions": actions,
            "raw_screens": raw_screens,
            "screens": screens,
            "info": infos
        }
    
    def evaluate_policy(self, env, seq_matching_callback, video_save_callback):
        self.policy.eval()

        rollout_dict = self.collect_rollouts(env, seq_matching_callback, deterministic=True, save_video=True)

        video_save_callback._on_step_no_buffer(
            raw_screens=rollout_dict["raw_screens"],
            screens=rollout_dict["screens"],
            states=rollout_dict["states"],
            actions=rollout_dict["actions"],
            rewards=rollout_dict["raw_rewards"],
            infos=rollout_dict["info"],
            num_timesteps=self.num_timesteps
        )

        # Evaluate the policy on all the states
        states = [[[i, j] for j in range(env.map.shape[1])] for i in range(env.map.shape[0])]
        # flatten the states
        states = np.array([item for sublist in states for item in sublist])

        print("States, Actions (0=DOWN, 1=RIGHT, 2=UP, 3=LEFT, 4=Stay)")

        action_to_str = {0: "DOWN", 1: "RIGHT", 2: "UP", 3: "LEFT", 4: "STAY"}
        for state in states:
            if env.map[state[0], state[1]] != -1:
                state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(self.device)
                probs = self.policy(state_tensor)
                probs = torch.log(probs).detach().cpu().numpy()[0]
                probs_str = ", ".join([f"{i}: {probs[i]:.2f}" for i in range(len(probs))])
                print(f"State={state} Best Action={np.argmax(probs)}, {action_to_str[int(np.argmax(probs))]} Actions={probs_str}")

        input("Check evaluate policy")

        self.policy.train()

    
    def save(self, path):
        pass