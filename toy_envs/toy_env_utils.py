import gymnasium as gym
import numpy as np

from collections import deque

O = 0     #open space
W = -1    # wall
G = 3     #goal
R = 1     #red object
B = 2     #blue object
A = 4     #agent

action_to_direction = {
            0: np.array([1, 0]), # Down
            1: np.array([0, 1]), # Right
            2: np.array([-1, 0]), # Up
            3: np.array([0, -1]), # Left
            4: np.array([0, 0]) # Stay in place
        }


def update_location(agent_pos, action, map_array, goal):
    """
    Update the agent's location and handle movable objects.
    """
    # print("map\n",map_array, "agent pos:",agent_pos)
    map_array[agent_pos[0],agent_pos[1]] = O
    direction = action_to_direction[action]
    new_pos = agent_pos + direction  # New position of the agent

    is_goal = False
    if not is_valid_location(new_pos, map_array):
        if map_array[goal[0], goal[1]] == O:
            map_array[goal[0], goal[1]] = G
        map_array[agent_pos[0], agent_pos[1]] = A
        return agent_pos, map_array, is_goal  # Agent stays in place if the new position is invalid

    # Check if there's a movable object at the new position
    obj_value = map_array[new_pos[0], new_pos[1]]
    if obj_value in (R, B):  # If the new position has a movable object
        obj_new_pos = new_pos + direction  # New position for the object
        if is_valid_location(obj_new_pos, map_array) and (map_array[obj_new_pos[0], obj_new_pos[1]] == O or map_array[obj_new_pos[0], obj_new_pos[1]] == G):
            # Move the object
            if obj_value == R:
                if map_array[obj_new_pos[0], obj_new_pos[1]] == G:
                    is_goal = True
            map_array[obj_new_pos[0], obj_new_pos[1]] = obj_value
            map_array[new_pos[0], new_pos[1]] = O
        else:
            if map_array[goal[0],goal[1]] == O:
                map_array[goal[0], goal[1]] = G
            map_array[agent_pos[0],agent_pos[1]] = A
            return agent_pos, map_array, is_goal  # Agent cannot push the object if the object can't move

    # Move the agent
    if map_array[goal[0], goal[1]] == O:
        map_array[goal[0], goal[1]] = G
    map_array[new_pos[0],new_pos[1]] = A
    return new_pos, map_array, is_goal


def is_valid_location(pos, map_array):
    x, y = pos
    if x < 0 or y < 0 or x >= map_array.shape[0] or y >= map_array.shape[1]:
        return False
    if map_array[x, y] == W:
        return False
    return True


def masking_obs(obs):
    map = obs.copy()
    obj_idx = np.argwhere(map == R)
    if len(obj_idx) > 0:
        map[obj_idx[0][0], obj_idx[0][1]] = B
    return map
    
def render_map_and_agent(map_array, agent_pos):
    """
       Render the map and agent as an RGB image.

       Map Legend:
       - Open space (O): Black [0, 0, 0]
       - Wall (W): White [255, 255, 255]
       - Red object (R): Red [255, 0, 0]
       - Blue object (B): Blue [0, 0, 255]
       - Goal (G): Green [0, 255, 0]
       - Agent: Yellow [255, 255, 0]
       """
    map_with_agent = map_array.copy()

    # Place the agent on the map
    # map_with_agent[agent_pos[0], agent_pos[1]] = 99  # Use a unique value for the agent

    # Convert the map to an RGB image
    map_with_agent_rgb = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    map_with_agent_rgb[map_with_agent == O] = [0, 0, 0]  # Open space - Black
    map_with_agent_rgb[map_with_agent == W] = [255, 255, 255]  # Wall - White
    map_with_agent_rgb[map_with_agent == R] = [255, 0, 0]  # Red object - Red
    map_with_agent_rgb[map_with_agent == B] = [0, 0, 255]  # Blue object - Blue
    map_with_agent_rgb[map_with_agent == G] = [0, 255, 0]  # Goal - Green
    map_with_agent_rgb[map_with_agent == A] = [255, 255, 0]  # Agent - Yellow

    # Increase the size of the image by a factor of 120
    map_with_agent_rgb = np.kron(map_with_agent_rgb, np.ones((120, 120, 1), dtype=np.uint8))

    return map_with_agent_rgb

# class BFS():
#     def __init__(self, env):
#         self.env = env
#         self.goal_pos = self.env.goal_pos
#
#     def predict(self, obs, deterministic=True):
#         agent_pos = np.argwhere(obs == A)[0]
#         path = self.bfs_shortest_path(obs, agent_pos, self.goal_pos)
#         return path[0]
#
#     def bfs_shortest_path(self, map_array, agent_pos, goal_pos, max_steps=100):
#         """
#         Find the shortest path to move the red object to the goal using BFS.
#         """
#         # BFS queue: (current_agent_pos, current_map, path, steps)
#         queue = deque([(agent_pos, map_array.copy(), [], 0)])
#         visited = set()
#
#         # Encode state for visited set
#         def encode_state(agent_pos, map_array):
#             return (tuple(agent_pos), tuple(map_array.flatten()))
#
#         while queue:
#             curr_agent_pos, curr_map, path, steps = queue.popleft()
#
#             # Check if we've reached the goal
#             if curr_map[goal_pos[0], goal_pos[1]] == R:
#                 return path  # Return the sequence of actions
#
#             # Stop if the maximum steps are exceeded
#             if steps >= max_steps:
#                 continue
#
#             # Mark state as visited
#             visited.add(encode_state(curr_agent_pos, curr_map))
#
#             # Explore all possible actions
#             for action in range(5):  # 0 to 4
#                 new_agent_pos, new_map, is_goal = update_location(curr_agent_pos, action, curr_map.copy(), goal_pos)
#
#                 # If goal is reached, return the path
#                 if is_goal:
#                     return path + [action]
#
#                 # Encode the new state
#                 new_state = encode_state(new_agent_pos, new_map)
#
#                 # Add new state to queue if not visited
#                 if new_state not in visited:
#                     queue.append((new_agent_pos, new_map, path + [action], steps + 1))
#
#         # Return an empty path if no solution is found
#         return []


def bfs_shortest_path(map_array, agent_pos, goal_pos, max_steps=100):
    """
    Find the shortest path to move the red object to the goal using BFS.
    """
    # BFS queue: (current_agent_pos, current_map, path, steps)
    queue = deque([(agent_pos, map_array.copy(), [], 0)])
    visited = set()
    cnt_search = 0
    max_search = 1_000_000
    # Encode state for visited set
    def encode_state(agent_pos, map_array):
        return (tuple(agent_pos), tuple(map_array.flatten()))

    while queue:
        cnt_search += 1
        if cnt_search >= max_search:
            print("failed in search")
            return []
        curr_agent_pos, curr_map, path, steps = queue.popleft()

        # Check if we've reached the goal
        if curr_map[goal_pos[0], goal_pos[1]] == R:
            return path  # Return the sequence of actions

        # Stop if the maximum steps are exceeded
        if steps >= max_steps:
            continue

        # Mark state as visited
        visited.add(encode_state(curr_agent_pos, curr_map))

        # Explore all possible actions
        for action in range(5):  # 0 to 4
            new_agent_pos, new_map, is_goal = update_location(curr_agent_pos, action, curr_map.copy(), goal_pos)

            # If goal is reached, return the path
            if is_goal:
                return path + [action]

            # Encode the new state
            new_state = encode_state(new_agent_pos, new_map)

            # Add new state to queue if not visited
            if new_state not in visited:
                queue.append((new_agent_pos, new_map, path + [action], steps + 1))

    # Return an empty path if no solution is found
    return []

class map_feasibility_check():
    def is_valid_location(self,pos, map_array):
        x, y = pos
        if x < 0 or y < 0 or x >= map_array.shape[0] or y >= map_array.shape[1]:
            return False
        return map_array[x, y] != W

    def bfs_reachable(self, start_pos, map_array, targets):
        """
        Perform BFS to check if the start_pos can reach any of the targets.
        """
        queue = deque([start_pos])
        visited = set()
        visited.add(tuple(start_pos))

        while queue:
            x, y = queue.popleft()

            if map_array[x, y] in targets:
                return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (x + dx, y + dy)
                if new_pos not in visited and is_valid_location(new_pos, map_array):
                    visited.add(new_pos)
                    queue.append(new_pos)

        return False

    def is_infeasible(self, map_array):
        """
        Check for infeasible situations in the given map.
        """
        # Locate key elements
        agent_pos = np.argwhere(map_array == A)[0]
        red_pos = np.argwhere(map_array == R)
        goal_pos = np.argwhere(map_array == G)

        # Check if the red object is trapped
        for pos in red_pos:
            accessible = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if self.is_valid_location(new_pos, map_array) and map_array[new_pos[0], new_pos[1]] in [O, G, B]:
                    accessible = True
                    break
            if not accessible:
                print(f"Infeasible: Red object at {pos} is trapped.")
                return True

        # Check if the goal is reachable
        if not self.bfs_reachable(agent_pos, map_array, [G, R]):
            print("Infeasible: Goal or red object is unreachable from the agent.")
            return True

        # Check if the goal is isolated
        if not any(self.bfs_reachable(goal, map_array, [R]) for goal in goal_pos):
            print("Infeasible: Goal is isolated and cannot be reached by the red object.")
            return True

        # All checks passed
        print("The map is feasible.")
        return False


class CustomObservationWrapper(gym.ObservationWrapper):
    """
    Custom wrapper to modify observations.
    Example: Add noise, normalize, or mask certain values.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space  # Keep the same observation space or modify if necessary

    def observation(self, obs):

        masked_obs = masking_obs(obs)
        return {"unmasked": obs, "masked": masked_obs}