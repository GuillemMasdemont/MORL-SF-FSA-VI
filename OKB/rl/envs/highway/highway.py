import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from highway_env.envs import HighwayEnv, HighwayEnvFast


class MOHighwayEnv(HighwayEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HighwayEnv environment.

    See [highway-env](https://github.com/eleurent/highway-env) for more information.

    ## Reward Space
    The reward is 3-dimensional:
    - 0: high speed reward
    - 1: right lane reward
    - 2: collision reward
    """

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float32)
        self.reward_dim = 3

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                0.5 * rewards["high_speed_reward"],
                0.5 * rewards["right_lane_reward"],
                - float(action in [0, 2]),  # action is lane change
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"]
        vec_reward -= 10 * rewards["collision_reward"]
        return obs, vec_reward, terminated, truncated, info


class MOHighwayEnvFast(HighwayEnvFast):
    """A multi-objective version of the HighwayFastEnv environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 0.0]), shape=(3,), dtype=np.float32)
        self.reward_dim = 3

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"],
                rewards["right_lane_reward"],
                - float(action in [0, 2]),  # action is lane change
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"] 
        vec_reward -= rewards["collision_reward"]
        return obs, vec_reward, terminated, truncated, info
