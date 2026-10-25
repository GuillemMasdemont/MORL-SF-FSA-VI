""" import gym
import numpy as np

gym.envs.register(
     id='DeepSeaTreasure-v0',
     entry_point='envs.dst:DeepSeaTreasure',
     max_episode_steps=100
)

gym.envs.register(
     id='GridXYO8-v0',
     entry_point='envs.grid_xyo8:GridXYO8',
     max_episode_steps=20     
)

gym.envs.register(
     id='TerrainWorld-v0',
     entry_point='envs.terrainworld:TerrainWorld',
     max_episode_steps=100
)

gym.envs.register(
     id='ReacherMultiTask-v0',
     entry_point='envs.reacher:ReacherBulletEnv',
     max_episode_steps=100
)

gym.envs.register(
     id='FourRoom-v0',
     entry_point='envs.four_room:Shapes',
     max_episode_steps=200
)
 """
""" gym.envs.register(
    id='mo-highway-v0',
    entry_point='envs.highway_env:HighwayEnv',
    max_episode_steps=40
)

gym.envs.register(
    id='mo-highway-fast-v0',
    entry_point='envs.highway_env:HighwayEnvFast',
    max_episode_steps=30
) """