import gymnasium as gym
gym.envs.register(
     id='Avalon-v0',
     entry_point='src.envs.avalon_env:AvalonEnv',
     max_episode_steps=100,
)