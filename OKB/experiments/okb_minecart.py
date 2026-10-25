import fire
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

import wandb as wb
from rl.successor_features.okb import OKB
from rl.utils.utils import seed_everything, equally_spaced_weights, incremental_weights


def main(
    g: int = 20,
    timesteps_per_iteration: int = 25000,
    num_iterations: int = 8,
    num_nets: int = 10,
    use_ok: bool = True,
    top_k_ok: int = 8,
    weight_selection: str = "okb",
    seed: int = None,
    save_dir: str = "./weights/",
    run_name: str = None,
    log: bool = True
):

    seed_everything(seed)

    if run_name is None:
        experiment_name = ("OKB" + (" (Random)" if weight_selection == "random" else "")) if use_ok else "SFOLS"
    else:
        experiment_name = run_name

    def make_env():
        env = mo_gym.make("minecart-v0")
        return env

    env = make_env()
    eval_env = make_env()

    agent = OKB(
        env,
        gamma=0.98,
        learning_rate=3e-4,
        ok_learning_rate=1e-3,
        gradient_updates=g,
        ok_gradient_updates=g,
        num_nets=num_nets,
        crossq=True,
        normalize_obs=False,
        use_ok=use_ok,
        weight_selection=weight_selection,
        top_k_ok=top_k_ok,
        top_k_base=1,
        num_ok_iterations=5,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        ok_net_arch=[256, 256, 256],
        buffer_size=int(5e5),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        ucb_exploration=0.0,
        epsilon_decay_steps=timesteps_per_iteration,
        learning_starts=1000,
        alpha_per=0.6,
        min_priority=0.01,
        per=True,
        batch_norm_momentum=0.99,
        drop_rate=0.01,
        layer_norm=True,
        lcb_pessimism=0.0,
        target_net_update_freq=200,
        tau=1,
        log=log,
        project_name="OKB",
        experiment_name=experiment_name,
        seed=seed,
    )

    agent.learn(
        eval_env=eval_env,
        timesteps_per_iteration=timesteps_per_iteration * (2 if not use_ok else 1),  # SFOLS uses the meta-policy budget to train the base policy
        num_iterations=num_iterations,
        test_tasks=incremental_weights(env.unwrapped.reward_dim, n_partitions=4),
        rep_eval=15,
        save_dir=save_dir,
    )

    agent.close_wandb()


if __name__ == "__main__":
    fire.Fire(main)
