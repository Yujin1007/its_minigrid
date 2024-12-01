from gymnasium.envs.registration import register

register(
     id="GridNav",
     entry_point="toy_envs.grid_nav:GridNavigationEnv"
)