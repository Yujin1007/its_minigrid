import gymnasium as gym
import numpy as np


O = 0     #open space
W = -1    # wall
G = 3   #goal
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

# def update_location(agent_pos, action, map_array, history):
#     """
#     Update the agent's location based on the action taken.
#
#     If the new location is invalid, the agent stays in place.
#         Invalid locations include:
#             - Locations outside the map
#             - Locations with a hole
#             - Locations where the location has been visited before
#     """
#     direction = action_to_direction[action]
#
#     new_pos = agent_pos + direction
#
#     if is_valid_location(new_pos, map_array, history):
#         return new_pos
#     else:
#         return agent_pos

# def is_valid_location(pos, map_array, history):
#     within_x_bounds = 0 <= pos[0] < map_array.shape[0]
#     within_y_bounds = 0 <= pos[1] < map_array.shape[1]
#
#     if within_x_bounds and within_y_bounds:
#         not_a_hole = map_array[pos[0], pos[1]] != -1
#         not_visited = pos.tolist() not in history
#
#         return  not_a_hole and not_visited
#     else:
#         return False
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

def masking_obs(obs):
    map = obs.copy()
    obj_idx = np.argwhere(map == R)
    if len(obj_idx) > 0:
        map[obj_idx[0][0], obj_idx[0][1]] = B
    return map

def is_valid_location(pos, map_array):
    x, y = pos
    if x < 0 or y < 0 or x >= map_array.shape[0] or y >= map_array.shape[1]:
        return False
    if map_array[x, y] == W:
        return False
    return True
    
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