from abc import ABC, abstractmethod
from typing import Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter


class RLAlgorithm(ABC):
    def __init__(self, env: gym.Env, experiment_name: str = "test", project_name: str = "test", device: Union[th.device, str] = "auto") -> None:
        self.env = env
        self.observation_shape = self.env.observation_space.shape
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_space = env.action_space
        if isinstance(self.env.action_space, (spaces.Discrete, spaces.MultiBinary)):
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device

        self.project_name = project_name
        self.experiment_name = experiment_name

        self.num_timesteps = 0
        self.num_episodes = 0

        self.writer = None

    @abstractmethod
    def eval(self, obs: np.ndarray) -> Union[int, np.ndarray]:
        """Gives the best action for the given observation

        Args:
            obs (np.array): Observation

        Returns:
            np.array or int: Action
        """

    @abstractmethod
    def train(self):
        """Update algorithm's parameters"""

    @abstractmethod
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration

        Returns:
            dict: Config
        """

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Retrienve q_values for the given observation

        Args:
            obs (np.array): State

        Returns:
            np.array: [q(s,a_1),...,q(s,a_n)]
        """
        raise NotImplementedError

    def setup_wandb(self, project_name: str, experiment_name: str):
        self.experiment_name = experiment_name
        import wandb

        wandb.init(
            project=project_name,
            sync_tensorboard=True,
            config=self.get_config(),
            name=self.experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        self.writer = SummaryWriter(f"/tmp/{self.experiment_name}")
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self):
        if self.writer is None:
            return
        import wandb
        self.writer.close()
        wandb.finish()
