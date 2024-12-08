import numpy as np
from collections import deque
from toy_envs.toy_env_utils import *
# Define constants
O = 0     # Open space
W = -1    # Wall
G = 3     # Goal
R = 1     # Red object
B = 2     # Blue object
A = 4     # Agent

action_to_direction = {
    0: np.array([1, 0]),  # Down
    1: np.array([0, 1]),  # Right
    2: np.array([-1, 0]), # Up
    3: np.array([0, -1]), # Left
    4: np.array([0, 0])   # Stay in place
}

def bfs_shortest_path(map_array, agent_pos, goal_pos, max_steps=100):
    """
    Find the shortest path to move the red object to the goal using BFS.
    """
    # BFS queue: (current_agent_pos, current_map, path, steps)
    queue = deque([(agent_pos, map_array.copy(), [], 0)])
    visited = set()

    # Encode state for visited set
    def encode_state(agent_pos, map_array):
        return (tuple(agent_pos), tuple(map_array.flatten()))

    while queue:
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

# Example map
map_array = np.array([
    [O, O, O, W, O, O, O, O, O],
    [O, O, O, W, O, O, O, O, O],
    [O, O, O, O, O, R, O, O, O],
    [O, O, O, W, O, O, B, O, O],
    [W, A, W, W, O, O, G, O, O],
    [O, O, O, W, O, O, O, O, O],
    [O, O, O, O, O, O, O, O, O],
    [O, O, O, W, O, O, B, O, O],
    [O, O, O, W, O, O, O, O, O]
])

# Initial agent and goal positions
agent_pos = np.array([4, 1])
goal_pos = np.array([4, 6])

# Find the shortest path
shortest_path = bfs_shortest_path(map_array, agent_pos, goal_pos)
action_to_direction_str = {
    0: "DOWN",
    1: "RIGHT",
    2: "UP",
    3: "LEFT",
    4: "STAY"
}

# Convert the shortest path to human-readable directions
human_readable_path = [action_to_direction_str[action] for action in shortest_path]

# Print the path
print("Shortest path:", human_readable_path)
print(shortest_path)