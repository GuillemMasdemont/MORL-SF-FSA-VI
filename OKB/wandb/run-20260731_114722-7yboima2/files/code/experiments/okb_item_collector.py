import fire
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

import wandb as wb
from rl.successor_features.okb import OKB
from rl.utils.utils import seed_everything, equally_spaced_weights, incremental_weights
from rl.utils.eval import policy_evaluation

from rl.envs import item_collector


class SequentialRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, info['sequential_reward'], terminated, truncated, info


def main(
    g: int = 10,
    timesteps_per_iteration: int = 50000,
    num_iterations: int = 5,
    num_nets: int = 10,
    use_ok: bool = True,
    top_k_ok: int = 8,
    weight_selection: str = "okb",
    initial_ok_task: str = "one-hot",
    seed: int = None,
    save_dir: str = "./weights/",
    load_dir: str = None,
    single_task: str = None,
    ok_discrete: bool = False,
    eval_gpi_policy: bool = False,
    run_name: str = None,
    log: bool = True
):

    seed_everything(seed)

    if run_name is None:
        experiment_name = ("OKB" + (" (Random)" if weight_selection == "random" else "")) if use_ok else "SFOLS"
    else:
        experiment_name = run_name

    def make_env():
        env = mo_gym.make("item-collector-v0")
        return env

    env = make_env()
    eval_env = make_env() # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = OKB(
        env,
        gamma=0.95,
        learning_rate=3e-4,
        ok_learning_rate=1e-3,
        gradient_updates=g,
        ok_gradient_updates=g,
        num_nets=num_nets,
        use_ok=use_ok,
        weight_selection=weight_selection,
        initial_ok_task=initial_ok_task,
        top_k_ok=top_k_ok,
        top_k_base=1,
        num_ok_iterations=5,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        ok_net_arch=[256, 256, 256],
        buffer_size=int(1e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        ucb_exploration=0.0,
        epsilon_decay_steps=timesteps_per_iteration,
        learning_starts=1000,
        alpha_per=0.6,
        min_priority=0.1,
        per=True,
        batch_norm_momentum=0.99,
        drop_rate=0.01,
        layer_norm=True,
        gpi_type='gpi',
        target_net_update_freq=200,
        tau=1,
        log=log,
        project_name="OKB",
        experiment_name=experiment_name,
        seed=seed,
    )

    if load_dir is not None:
        agent.load(load_dir)

    if single_task is not None:
        env, eval_env = mo_gym.make(f"item-collector-v0", single_task=single_task), mo_gym.make(f"item-collector-v0", single_task=single_task)
        agent.gradient_updates = 1
        agent.ok_gradient_updates = 5
        agent.batch_size = 256

        if eval_gpi_policy:
            # python experiments/okb_item_collector.py --run_name="OKB Sequential Reward Task 2 g5" --use_ok=True --load_dir="./weights/okb-it5-item-collector-v0-23-01-2025-22:19:41/" --single_task="sequential" --eval_gpi_policy=True --log=False
            class GPI:
                def __init__(self, agent, w):
                    self.agent = agent
                    self.gamma = 0.95
                    self.agent.use_ok = False
                    self.agent.use_gpi = True
                    self.w = w
                def eval(self, obs):
                    return self.agent.eval(obs, self.w)
            w1 = np.array([1.0, -1.0])
            w2 = np.array([-1.0, 1.0])
            w3 = np.array([0.5, 0.5])
            print(w1, policy_evaluation(GPI(agent, w1), eval_env, rep=50))
            print(w2, policy_evaluation(GPI(agent, w1), eval_env, rep=50))
            print(w3, policy_evaluation(GPI(agent, w1), eval_env, rep=50))

        elif not ok_discrete:
            agent.ok_policy_noise = 0.05
            agent.learn_single_ok(env, eval_env, total_timesteps=100000, save_dir=save_dir)
        else:
            agent.epsilon_decay_steps = 1
            agent.learn_single_ok_discrete(env, eval_env, total_timesteps=100000)

    else:
        agent.learn(
            eval_env=eval_env,
            timesteps_per_iteration=timesteps_per_iteration * (2 if not use_ok else 1),  # SFOLS uses the meta-policy budget to train the base policy,
            num_iterations=num_iterations,
            test_tasks=incremental_weights(env.unwrapped.reward_dim, n_partitions=4),
            rep_eval=15,
            save_dir=save_dir,
        )

    agent.close_wandb()


if __name__ == "__main__":
    fire.Fire(main)
