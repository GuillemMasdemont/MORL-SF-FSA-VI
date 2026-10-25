import numpy as np
import gymnasium as gym
import cv2

# Action keywords
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class ItemCollector(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 grid_length=10,
                 n=2,
                 penalty=True,
                 single_task=None,
                 initial_food_items=[5, 5]):
        """
            grid_length: Length of the square grid
            n: Number of food items
            penalty: Penalize other items when collecting an item
            decrease: Decrease in proportion of resources at each time step
            initial_food_items: Number of food items in the initial state
            spawn_prob: Probability of spawning a new food item of each type
                        at every time step
        """
        self.render_mode = render_mode

        self.grid = np.zeros((grid_length, grid_length, n))
        self.penalty = penalty
        self.resources = np.zeros(n)
        self.cumulants = np.zeros(n)
        self.m = n
        self.n = n

        obs_len = grid_length * grid_length * n
        self.observation_space = gym.spaces.Box(low=-np.inf, high=1, shape=(obs_len,))
        self.action_space = gym.spaces.Discrete(4)

        self.reward_dim = self.m
        self.reward_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.reward_dim,))

        self.n_steps = 0
        self.learning_options = False
        self.initial_food_items = initial_food_items

        self.single_task = single_task

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        grid_length, _, _ = self.grid.shape
        self.grid = np.zeros((grid_length, grid_length, self.n))

        # Populate the grid.
        for index, n_items in enumerate(self.initial_food_items):
            for _ in range(n_items):
                while True:
                    x, y = self.np_random.integers(grid_length, size=(2))
                    if (self.grid[x, y].sum() == 0 and
                            (x != int(grid_length / 2) or
                             y != int(grid_length / 2))):
                        break
                self.grid[x, y, index] = 1

        self.resources = np.zeros(self.m)
        self.cumulants = np.zeros(self.m)
        self.n_steps = 0

        if self.render_mode == "human":
            self._render_frame()

        return self.get_observation(), {}
    
    def balanced_reward(self, centre):
        if self.grid[centre].sum() == 0:
            return 0.0
        else:
            collected_item = np.flatnonzero(self.grid[centre])[0]
            counts = np.zeros(self.n)
            for i in range(self.n):
                counts[i] = np.count_nonzero(self.grid[:, :, i])

            if counts.max() == counts[collected_item]:
                return 1.0
            else:
                return -1.0

    def sequential_reward(self, centre):
        if self.grid[centre].sum() == 0:
            return 0.0
        else:
            collected_item = np.flatnonzero(self.grid[centre])[0]
            counts = np.zeros(self.n)
            for i in range(self.n):
                counts[i] = np.count_nonzero(self.grid[:, :, i])
                if collected_item == i:
                    return 1.0
                elif counts[i] > 0:
                    return -1.0
                elif counts[i] == 0:
                    continue

    def step(self, a):
        self.n_steps += 1
        self.update_grid(a)

        grid_length = self.grid.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))

        balanced_reward = self.balanced_reward(centre)
        sequential_reward = self.sequential_reward(centre)

        phi = np.zeros(self.n, dtype=np.float32)
        if self.grid[centre].sum() != 0:
            if self.penalty:
                phi += -1/self.m
            phi[np.flatnonzero(self.grid[centre])[0]] = 1.0
            self.grid[centre] = np.zeros(self.n)

        info = {'grid': self.grid, 'resources': self.resources}
        info['balanced_reward'] = balanced_reward
        info['sequential_reward'] = sequential_reward

        if self.render_mode == "human":
            self._render_frame()
        
        if self.single_task is not None:
            if self.single_task == "balanced":
                return self.get_observation(), balanced_reward, False, False, info
            elif self.single_task == "sequential":
                return self.get_observation(), sequential_reward, False, False, info

        return self.get_observation(), phi, False, False, info

    def get_observation(self):
        return np.concatenate(self.grid, axis=None)

    def _render_frame(self):
        grid_length = self.grid.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))
        self.grid[centre] = np.ones(self.n)
        scale = max(int(400 / self.grid.shape[1]), 1)
        modified_size = self.grid.shape[1] * scale
        img = cv2.resize(self.grid, (modified_size, modified_size),
                         interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                 value=[0.375, 0.375, 0.375])
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
        cv2.imshow('ForagingWorld', img)
        cv2.waitKey(100)
        self.grid[centre] = np.zeros(self.n)

    def close(self):
        cv2.destroyAllWindows()

    def update_grid(self, a):
        """
            The grid follows toroidal dynamics i.e. it wraps around, connecting
            cells on opposite edges.
        """
        if a == UP:
            self.grid = np.concatenate((self.grid[-1:, :, :],
                                        self.grid[:-1, :, :]),
                                       axis=0)

        elif a == DOWN:
            self.grid = np.concatenate((self.grid[1:, :, :],
                                        self.grid[:1, :, :]),
                                       axis=0)

        elif a == RIGHT:
            self.grid = np.concatenate((self.grid[:, 1:, :],
                                        self.grid[:, :1, :]),
                                       axis=1)

        elif a == LEFT:
            self.grid = np.concatenate((self.grid[:, -1:, :],
                                        self.grid[:, :-1, :]),
                                       axis=1)

    def spawn_new_item(self, food_type_index):
        """
            Each time a food item is consumed, a replacement item is spawned
            at a random position in the grid.
        """
        grid_length = self.grid.shape[0]
        while True:
            x, y = self._np_random.integers(grid_length, size=(2))
            if (self.grid[x, y].sum() == 0 and (x != int(grid_length / 2) or
                                                y != int(grid_length / 2))):
                break
        self.grid[x, y, food_type_index] = 1


if __name__ == "__main__":
    import mo_gymnasium as mo_gym
    import rl.envs.item_collector

    env = mo_gym.make("item-collector-v0", render_mode=None)
    env.reset()
    while True:
        action = int(input("Enter action: "))
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            env.reset()
        print(obs, r)
