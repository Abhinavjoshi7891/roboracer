import gymnasium as gym
import numpy as np

# Create a basic F1TENTH environment
env = gym.make('f1tenth_gym:f1tenth-v0',
    config={'map': 'Spielberg', 'map_ext': '.png', 'num_agents': 1}
)

obs, info = env.reset()
print(f'Observation keys: {obs.keys()}')
print(f'LiDAR scan shape: {obs["scans"][0].shape}')
print('F1TENTH gym is working!')
env.close()
