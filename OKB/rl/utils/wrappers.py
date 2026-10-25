import itertools
import numpy as np
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gym.spaces import Box, Discrete


class NormalizedObservationWrapper(ObservationWrapper):
    """
    Normalizes observations such that the values are
    between 0.0 and 1.0, -1.0 and 0.0, or -1.0 and 1.0.
    """
    def __init__(self, env):
        super(NormalizedObservationWrapper, self).__init__(env)
        if not isinstance(self.env.observation_space, Box):
            raise AssertionError(
                "This wrapper can only be applied to environments with a continuous observation space.")
        if np.inf in self.env.observation_space.low or np.inf in self.env.observation_space.high:
            raise AssertionError(
                "This wrapper cannot be used for observation spaces with an infinite lower/upper bound.")

        self.a = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        self.b = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        for i in range(len(self.a)):
            if self.env.observation_space.low[i] < 0:
                self.a[i] = -1.0
            if self.env.observation_space.high[i] > 0:
                self.b[i] = 1.0

        self.observation_space = Box(low=self.a, high=self.b)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # x' = (b - a)*((x - min)/(max - min)) + a,  x' in [a, b]
        return ((self.b - self.a)*(observation - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)) + self.a


class NormalizedRewardWrapper(RewardWrapper):
    """
    Normalizes rewards such that the values are between 0.0 and 1.0.
    """
    def __init__(self, env, low=None, high=None):
        super(NormalizedRewardWrapper, self).__init__(env)
        self.low = low if low is not None else self.env.reward_range[0]
        self.high = high if high is not None else self.env.reward_range[1]
        self.reward_range = (0.0, 1.0)

    def reward(self, rew):
        return (rew - self.low) / (self.high - self.low)


class RandomAction(ActionWrapper):
    """
    Makes the environment stochastic by introducing a random_prob chance of executing a random action.
    """
    def __init__(self, env, random_prob=0.1):
        super().__init__(env)
        self.random_prob = random_prob
    
    def action(self, action):
        if self.env.unwrapped.np_random.random() < self.random_prob:
            action_env = self.env.action_space.sample()
        else:
            action_env = action
        return action_env
        

class BangBangAction(ActionWrapper):

    def __init__(self, env, bins=3):
        super().__init__(env)
        self.bins = bins
        self.action_space = Discrete(bins ** env.action_space.shape[0])
        self.action_bins = list(np.linspace(self.env.action_space.low, self.env.action_space.high, num=self.bins).transpose())
        self.action_list = itertools.product(*self.action_bins)
        self.action_list = [np.array(a) for a in self.action_list]
    
    def action(self, action):
        continuous_action = self.action_list[action]
        return continuous_action


class MaxAndSkipEnv(Wrapper):
    """
    From stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#NoopResetEnv
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MinAtarObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            0.0, 1.0, shape=np.array(self.env.game.state_shape())[[2, 0, 1]], dtype=np.float32
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)[np.newaxis,:,:,:]
