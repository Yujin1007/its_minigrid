import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

import imageio
from pygame.examples.music_drop_fade import starting_pos
from stable_baselines3.common.utils import obs_as_tensor
from torchgen.native_function_generation import self_to_out_signature
from wandb.wandb_agent import agent

from toy_envs.bfs_search import map_array
from toy_envs.toy_env_utils import update_location, render_map_and_agent, bfs_shortest_path
import copy
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3
STAY = 4

W = -1    # wall
O = 0     #open space
R = 1     #red object
B = 2     #blue object
G = 3     #goal
A = 4     #agent

class GridNavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, map_array, goal_pos, render_mode=None, episode_length=20, full_observability=True):
        self.grid_size = map_array.shape  # The size of the square grid

        # Observations is the agent's location in the grid
        # self.observation_space = spaces.Box(np.zeros((2,)), np.array([grid_size - 1 for grid_size in self.grid_size]), shape=(2,), dtype=np.int64)
        self.observation_space = spaces.Box(low=W,high=A,shape=self.grid_size, dtype=np.int64)

        # We have 5 actions, corresponding to "right", "up", "left", "down", "do nothing"
        self.action_space = spaces.Discrete(5)
        self.full_observability = full_observability
        self.num_steps = 0
        self.goal_pos = goal_pos
        self.episode_length = episode_length
        self.map = map_array
        self.initial_states = [copy.deepcopy(map_array)]
        self.test_attribute = 1
        self.flag_bring_map = False
        self.is_goal=False
        self.curriculum = 0

        self.initial_map, self._starting_pos = self._choose_initial_state()
        self._agent_pos = np.copy(self._starting_pos)
        self._new_agent_pos = self._agent_pos
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    
    def step(self, action):
        # print("after changing attiribute:",self._agent_pos)
        self._new_agent_pos, self.map, self.is_goal = update_location(agent_pos=self._agent_pos, action=action, map_array=self.map,goal=self.goal_pos)
        # self._history.append(self._agent_pos.tolist())

        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward()
        # # reward = 0
        # if self.is_goal:
        #     reward = 1.0
        # else:
        #     reward = 0.0
        # if self._new_agent_pos.tolist() == self._agent_pos.tolist():
        #     reward = -1
        # else:
        #     reward = 0  # We are using sequence matching function to produce reward
        self._agent_pos = self._new_agent_pos

        self.num_steps += 1

        terminated = self.num_steps >= self.episode_length or self.is_goal
        return observation, reward, terminated,  False, info

    def _get_obs(self):
        # map = self.map.copy()
        # map[self._agent_pos[0], self._agent_pos[1]] = A
        # TODO make observation for partial observable student

        if self.full_observability:
            pass
        if self.is_goal:
            terminal_map = np.ones_like(self.map) * -1
            terminal_map[self._new_agent_pos[0],self._new_agent_pos[1]] = A
            terminal_map[self.goal_pos[0],self.goal_pos[1]] = R
            # print(terminal_map)
            obs = terminal_map
            # return self._agent_pos
        # print(self.map)
        else:
            obs =  self.map
        feasibility = self._check_obs_feasibility(obs)
        if feasibility:
            return obs
        else:
            print("obs:\n",obs)
            print("map\n", self.map)
            print("termination \n", terminal_map)
            return None
    def _get_info(self):
        return {"step": self.num_steps, "goal": self.is_goal}
    def _get_reward(self):
        target_pos = np.argwhere(self.map == R)[0]
        l1_norm_to_goal = np.linalg.norm(target_pos-self.goal_pos, ord=1)
        l1_norm_to_target = np.linalg.norm(target_pos-self._new_agent_pos, ord=1)
        l1_norm = l1_norm_to_goal + l1_norm_to_target
        if self.is_goal:
            return 1
        else:
            return -l1_norm
    def _set_initial_states(self, new_initial_states) -> None:
        # Note: this value should be used only at the next reset
        flat_new_initial_states = [item for sublist in new_initial_states for item in sublist]
        self.initial_states = flat_new_initial_states
        # print("renew initial states", len(self.initial_states))
        # print(self.initial_states.shape)

    def _choose_initial_state(self):

        initial_map= copy.deepcopy(random.choice(self.initial_states))

        if len(np.argwhere(initial_map == R)) == 0:
            obj_candidate = np.argwhere(initial_map == B)
            obj_pos = random.choice(obj_candidate)
            initial_map[obj_pos[0], obj_pos[1]] = R
        agent_pos = np.argwhere(initial_map == A)

        if len(agent_pos) == 0:
            agent_candidate = np.argwhere(initial_map == O)
            starting_pos = random.choice(agent_candidate)
        else:
            starting_pos = agent_pos[0]

        return initial_map, starting_pos
    def reset(self, obj_idx=None, seed=0):
        """
        This is a deterministic environment, so we don't use the seed."""
        self.num_steps = 0
        self.is_goal=False
        self.initial_map, self._starting_pos = self._choose_initial_state()

        self._agent_pos = np.copy(self._starting_pos)
        self._new_agent_pos = self._agent_pos

        # self._history = [self._agent_pos.tolist()]
        # if obj_idx is None:
        #     obj_idx = np.argwhere(self.initial_map== R)
        #     if len(obj_idx) == 0: # no red object in the map, choose random one
        #         obj_pos = self.obj_candidate[np.random.randint(0,len(self.obj_candidate))]
        #     else:
        #         obj_pos = obj_idx[0]
        # else:
        #     obj_pos = self.obj_candidate[obj_idx]
        self.map = np.copy(self.initial_map)
        # self.map[obj_pos[0], obj_pos[1]] = R
        # self.map[self._starting_pos[0], self._starting_pos[1]] = A

        # print("###################initial map:\n", self.map)
        return self._get_obs(), self._get_info()
    def _check_obs_feasibility(self,obs):
        agent_pos = np.argwhere(obs == A)
        target_pos = np.argwhere(obs == R)
        if (len(agent_pos) == 0 or len(target_pos) == 0):
            return False
        else:
            return True
    def render(self):
        """
        Render the environment as an RGB image.
        
        The agent is represented by a yellow square, empty cells are white, and holes are blue."""
        if self.render_mode == "rgb_array":
            return render_map_and_agent(self.map, self._agent_pos)
        else:
            return

class GridNavigationEmptyEnv(GridNavigationEnv):
    def _choose_initial_state(self):
        initial_map = copy.deepcopy(random.choice(self.initial_states))
        # print("map\n",self.initial_states)
        target_candidiate = [np.array([1,5]),np.array([1,6]),np.array([1,7]),
                             np.array([2,5]),np.array([2,6]),np.array([2,7]),
                             np.array([6,5]),np.array([6,6]),np.array([6,7]),
                             np.array([7,5]),np.array([7,6]),np.array([7,7])]
        target_obj = random.choice(list(target_candidiate))
        initial_map[target_obj[0], target_obj[1]] = R

        agent_candidate = np.argwhere((initial_map == O) | (initial_map == G)) # it can locate either empty space or goal pos
        starting_pos, blue_obj = random.sample(list(agent_candidate), 2)

        initial_map[blue_obj[0], blue_obj[1]] = B
        initial_map[starting_pos[0], starting_pos[1]] = A

        return initial_map, starting_pos
class GridNavigationCurriculumEnv(GridNavigationEnv):
    def _choose_initial_state(self):
        default_obj_pos = [np.array([1,5]), np.array([7,7])]
        # default_obj_pos = [np.array([1,6]), np.array([7,6])]

        initial_map = copy.deepcopy(random.choice(self.initial_states))
        if self.curriculum == 0: #randomize only agent pos
            blue_obj, red_obj = random.sample(default_obj_pos,2)
            initial_map[blue_obj[0], blue_obj[1]] = B
            initial_map[red_obj[0], red_obj[1]] = R

            agent_candidate = np.argwhere(
                (initial_map == O) | (initial_map == G))  # it can locate either empty space or goal pos
            starting_pos = random.choice(agent_candidate)
            initial_map[starting_pos[0], starting_pos[1]] = A

        elif self.curriculum == 1: #randomize agent and blue object pos
            target_obj = random.choice(default_obj_pos)
            initial_map[target_obj[0], target_obj[1]] = R
            agent_candidate = np.argwhere(
                (initial_map == O) | (initial_map == G))  # it can locate either empty space or goal pos
            starting_pos, blue_obj = random.sample(list(agent_candidate), 2)

            initial_map[blue_obj[0], blue_obj[1]] = B
            initial_map[starting_pos[0], starting_pos[1]] = A

        elif self.curriculum == 2: #randomize agent, blue and red object pos
            target_candidates = [np.array([1,5]),np.array([1,6]),np.array([1,7]),
                             np.array([2,5]),np.array([2,6]),np.array([2,7]),
                             np.array([6,5]),np.array([6,6]),np.array([6,7]),
                             np.array([7,5]),np.array([7,6]),np.array([7,7])]
            # target_obj = random.choice(list(target_candidates))
            target_obj = random.choice(target_candidates)

            initial_map[target_obj[0], target_obj[1]] = R

            agent_candidate = np.argwhere((initial_map == O) | (initial_map == G)) # it can locate either empty space or goal pos
            starting_pos, blue_obj = random.sample(list(agent_candidate), 2)

            initial_map[blue_obj[0], blue_obj[1]] = B
            initial_map[starting_pos[0], starting_pos[1]] = A

        else: # randomize all
            agent_candidate = np.argwhere(
                (initial_map == O) | (initial_map == G))  # it can locate either empty space or goal pos
            starting_pos, blue_obj, red_obj = random.sample(list(agent_candidate), 3)

            initial_map[blue_obj[0], blue_obj[1]] = B
            initial_map[red_obj[0], red_obj[1]] = R
            initial_map[starting_pos[0], starting_pos[1]] = A


        return initial_map, starting_pos
    def _trigger_curriculum(self) -> None:
        self.curriculum += 1
        print(f"########\nCurrent curriculum is Level {self.curriculum}\n######\n")

if __name__ == "__main__":

    env = GridNavigationEnv(
        # np.array([
        #     [A, O, O, W, O, O, O, O, O],
        #     [O, O, O, W, O, O, R, O, O],
        #     [O, O, O, O, O, O, O, O, O],
        #     [O, O, O, W, O, O, O, O, O],
        #     [W, O, W, W, W, W, O, W, W],
        #     [O, O, O, W, O, O, O, O, O],
        #     [O, O, O, O, O, O, O, O, O],
        #     [O, O, O, W, O, O, B, O, O],
        #     [O, O, O, W, O, O, O, O, O]
        # ]),
        map_array = np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, R, O, O, O],
            [O, O, O, W, O, O, B, O, O],
            [W, A, W, W, W, W, G, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, B, O, O],
            [O, O, O, W, O, O, O, O, O]
        ]),

        render_mode="rgb_array",
        goal_pos=np.array([4,6]),)
    env.reset()
    env.render()
    
    # path = [LEFT,LEFT,LEFT]
    # path = [2, 1, 2, 1, 1, 2, 1, 0, 0, 3, 0, 1]
    path = bfs_shortest_path(env.map, env._agent_pos, env.goal_pos)
    # path = [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, UP,UP, RIGHT, DOWN,DOWN,DOWN, UP, LEFT, LEFT, LEFT, LEFT, LEFT,DOWN,DOWN,DOWN,DOWN,RIGHT,RIGHT,RIGHT,RIGHT, UP, RIGHT, UP,UP, DOWN,DOWN, LEFT,DOWN,DOWN,DOWN, RIGHT,UP,UP,UP,STAY]
    # path = [3, 4, 2, 1, 3, 1, 3, 1, 2, 1, 1, 2, 2, 4, 1, 1, 0, 0, 0]
    frames = []
    frames.append(env.render())

    for action in path:
        env.step(action)
        frames.append(env.render())


    imageio.mimsave("testing.gif", frames, duration=1/20, loop=0)

    writer = imageio.get_writer('testing.mp4', fps=20)

    for im in frames:
        writer.append_data(im)

    writer.close()
#
# class GridNavigationEnvHistory(GridNavigationEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         n_cells = self.grid_size[0] * self.grid_size[1]
#         self.observation_space = spaces.MultiDiscrete(np.ones((n_cells,))*3) # 3 possible states for each grid location (unvisited, visited, current)
#         self.visited = np.zeros(self.grid_size)
#
#     def _get_obs(self):
#         observation = self.visited.copy()
#         observation[tuple(self._agent_pos)] = 2
#         observation = observation.flatten()
#
#         return observation
#
#     def reset(self, seed=0):
#         """
#         This is a deterministic environment, so we don't use the seed."""
#         self.num_steps = 0
#         self._agent_pos = np.copy(self._starting_pos)
#         self.visited = np.zeros(self.grid_size)
#         self.map = np.copy(self.initial_map)
#         print("initial map:", self.map)
#         return self._get_obs(), self._get_info()
#
#     def step(self, action):
#         self.visited[tuple(self._agent_pos)] = 1
#         self._agent_pos, self.map, self.is_goal = update_location(agent_pos=self._agent_pos, action=action, map_array=self.map, goal=self.goal_pos)
#
#         observation = self._get_obs()
#         assert len(np.nonzero(observation == 2)) == 1
#         info = self._get_info()
#         if self.is_goal:
#             reward = 1
#         else:
#             reward = 0
#
#         self.num_steps += 1
#         terminated = self.num_steps >= self.episode_length or self.is_goal
#         print(f"step {self.num_steps}: {observation}")
#
#         return observation, reward, terminated, False, info