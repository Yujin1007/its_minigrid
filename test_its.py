from imitation.algorithms.bc import BC
from imitation.policies.base import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Define the environment
env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

# Path to the saved BC policy
model_path = "path/to/your_policy.pth"

# Load the policy
policy = BasePolicy.load(model_path)

# Create a dummy BC model for evaluation
bc_model = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    policy=policy,
)

# Test the policy
obs = env.reset()
for _ in range(1000):  # Run for 1000 timesteps
    action, _ = bc_model.policy.predict(obs, deterministic=True)  # Predict actions
    obs, reward, done, info = env.step(action)  # Step environment
    env.render()  # Optional: Render environment
    if done:
        obs = env.reset()  # Reset environment after an episode