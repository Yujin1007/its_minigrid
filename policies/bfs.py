from toy_envs.toy_env_utils import bfs_shortest_path, update_location
import numpy as np
from collections import deque

O = 0     #open space
W = -1    # wall
G = 3     #goal
R = 1     #red object
B = 2     #blue object
A = 4     #agent


class BFS:
    def __init__(self, env):
        self.env = env
        self.goal_pos = tuple(self.env.goal_pos)  # Convert to tuple for easier handling

    def predict(self, obs, deterministic=True):
        agent_pos = tuple(np.argwhere(obs == A)[0])  # Convert to tuple for consistency
        path = bfs_shortest_path(obs, agent_pos, self.goal_pos)
        return path[0] if path else 4  # Return STAY (4) if no path is found

    # def bfs_shortest_path(self, map_array, agent_pos, goal_pos):
    #     """
    #     Optimized BFS to find the shortest path for the agent to move the red object to the goal.
    #     """
    #     queue = deque([(agent_pos, [], 0)])  # Queue stores (agent_pos, path, steps)
    #     visited = set()
    #
    #     # Encode state as agent position and the red object's position
    #     def encode_state(agent_pos):
    #         red_pos = tuple(np.argwhere(map_array == R)[0])  # Position of the red object
    #         return (agent_pos, red_pos)
    #
    #     while queue:
    #         curr_agent_pos, path, steps = queue.popleft()
    #
    #         # Mark the state as visited
    #         state = encode_state(curr_agent_pos)
    #         if state in visited:
    #             continue
    #         visited.add(state)
    #
    #         # Check if the red object is at the goal
    #         if map_array[goal_pos[0], goal_pos[1]] == R:
    #             return path  # Return the sequence of actions
    #
    #         # Explore all possible actions
    #         for action in range(5):  # 0 to 4
    #             new_agent_pos, new_map, is_goal = update_location(curr_agent_pos, action, map_array.copy(), goal_pos)
    #
    #             # If goal is reached, return the path
    #             if is_goal:
    #                 return path + [action]
    #
    #             # Add the new state to the queue if not visited
    #             new_state = encode_state(new_agent_pos)
    #             if new_state not in visited:
    #                 queue.append((new_agent_pos, path + [action], steps + 1))
    #
    #     return []  # No solution found




if __name__ == "__main__":


    map = np.array([
            [ 0,  0,  0, -1,  0,  0,  0,  0,  0],
           [ 0,  0,  0, -1,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 4,  0,  0,  0,  0,  0,  0,  0,  0],
           [-1,  0,  0,  0,  0,  0,  3, -1, -1],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  2,  0,  0],
           [ 0,  0,  0, -1,  0,  0,  0,  1,  0],
           [ 0,  0,  0, -1,  0,  0,  0,  0,  0]])
    start_pos = np.argwhere(map == A)[0]
    goal_pos = np.array([4,6])
    bfs_path = bfs_shortest_path(map, start_pos, goal_pos)
    print(bfs_path)