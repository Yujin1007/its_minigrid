import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

import imageio
from torchgen.native_function_generation import self_to_out_signature

from toy_envs.toy_env_utils import update_location, render_map_and_agent

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

    def __init__(self, map_array, goal_pos, starting_pos=None, render_mode=None, episode_length=20, full_observability=True):
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
        self.initial_states = [map_array]
        self.test_attribute = 1
        self.flag_bring_map = False
        self.initial_map, self._starting_pos, self.obj_candidate, self.agent_candidate = self._choose_initial_state()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    
    def step(self, action):
        # print("after changing attiribute:",self._agent_pos)
        self._new_agent_pos, self.map, is_goal = update_location(agent_pos=self._agent_pos, action=action, map_array=self.map,goal=self.goal_pos)
        self._history.append(self._agent_pos.tolist())

        observation = self._get_obs()
        info = self._get_info(is_goal)
        # reward = 0
        if is_goal:
            reward = 1.0
        else:
            reward = 0.0
        # if self._new_agent_pos.tolist() == self._agent_pos.tolist():
        #     reward = -1
        # else:
        #     reward = 0  # We are using sequence matching function to produce reward
        self._agent_pos = self._new_agent_pos

        self.num_steps += 1

        terminated = self.num_steps >= self.episode_length or is_goal
        return observation, reward, terminated,  False, info

    def _get_obs(self):
        # map = self.map.copy()
        # map[self._agent_pos[0], self._agent_pos[1]] = A
        # TODO make observation for partial observable student
        if self.full_observability:
            pass
            # return self._agent_pos
        # print(self.map)
        return self.map

    
    def _get_info(self, is_goal=False):
        return {"step": self.num_steps, "goal": is_goal}

    def _set_initial_states(self, new_initial_states) -> None:
        # Note: this value should be used only at the next reset
        flat_new_initial_states = [item for sublist in new_initial_states for item in sublist]
        self.initial_states = flat_new_initial_states
        # print("renew initial states", len(self.initial_states))
        # print(self.initial_states.shape)
    def _choose_initial_state(self):

        initial_map= random.choice(self.initial_states)
        # print("len initial_states:", len(self.initial_states), "\n",initial_map)

        # print("\n===========", self.test_attribute)
        # print("\n===========\n",self.initial_states)
        obj_candidate = np.argwhere(initial_map == B)
        agent_pos = np.argwhere(initial_map == A)
        agent_candidate = np.argwhere(initial_map == O)
        if len(agent_pos) == 0:
            starting_pos = random.choice(agent_candidate)
        else:
            starting_pos = agent_pos[0]
        return initial_map, starting_pos, obj_candidate, agent_candidate
    def reset(self, obj_idx=None, seed=0):
        """
        This is a deterministic environment, so we don't use the seed."""

        self.num_steps = 0
        self.initial_map, self._starting_pos, self.obj_candidate, self.agent_candidate = self._choose_initial_state()

        self._agent_pos = np.copy(self._starting_pos)

        self._history = [self._agent_pos.tolist()]
        if obj_idx is None:
            obj_idx = np.argwhere(self.initial_map== R)
            if len(obj_idx) == 0: # no red object in the map, choose random one
                obj_pos = self.obj_candidate[np.random.randint(0,len(self.obj_candidate))]
            else:
                obj_pos = obj_idx[0]
        else:
            obj_pos = self.obj_candidate[obj_idx]
        self.map = np.copy(self.initial_map)
        self.map[obj_pos[0], obj_pos[1]] = R
        self.map[self._starting_pos[0], self._starting_pos[1]] = A

        # print("initial map:", self.map)
        return self._get_obs(), self._get_info()

    def render(self):
        """
        Render the environment as an RGB image.
        
        The agent is represented by a yellow square, empty cells are white, and holes are blue."""
        if self.render_mode == "rgb_array":
            return render_map_and_agent(self.map, self._agent_pos)
        else:
            return
    
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
        np.array([
            [ 0,  0,  0, -1,  0,  0,  0,  0,  0],
            [ 0,  0,  0, -1,  0,  0,  B,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0, -1,  0,  0,  0,  0,  0],
            [-1,  0, -1, -1, -1, -1,  G,  B, -1],
            [ 0,  0,  0, -1,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  2,  0,  0,  B,  0,  0],
            [ 0,  0,  0, -1,  0,  0,  0,  0,  0]]),

        render_mode="rgb_array",)
    env.reset()
    env.render()
    
    # path = [RIGHT, RIGHT, DOWN, DOWN, LEFT, LEFT, UP, STAY]
    path = [UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, UP,UP, RIGHT, DOWN,DOWN,DOWN, UP, LEFT, LEFT, LEFT, LEFT, LEFT,DOWN,DOWN,DOWN,DOWN,RIGHT,RIGHT,RIGHT,RIGHT, UP, RIGHT, UP,UP, DOWN,DOWN, LEFT,DOWN,DOWN,DOWN, RIGHT,UP,UP,UP,STAY]
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
#         self._agent_pos, self.map, is_goal = update_location(agent_pos=self._agent_pos, action=action, map_array=self.map, goal=self.goal_pos)
#
#         observation = self._get_obs()
#         assert len(np.nonzero(observation == 2)) == 1
#         info = self._get_info()
#         if is_goal:
#             reward = 1
#         else:
#             reward = 0
#
#         self.num_steps += 1
#         terminated = self.num_steps >= self.episode_length or is_goal
#         print(f"step {self.num_steps}: {observation}")
#
#         return observation, reward, terminated, False, info