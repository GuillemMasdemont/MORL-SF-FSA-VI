import fire
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

import wandb as wb
from rl.successor_features.okb import OKB
from rl.utils.utils import seed_everything, equally_spaced_weights, incremental_weights
import rl.envs.mo_push
from rl.envs.mo_push.mo_push import FetchDiscreteAction, FetchDiscreteManhattanAction, FetchObservationWrapper



def main(
    g: int  = 10,
    timesteps_per_iteration: int = 50000,
    num_iterations: int = 10,
    num_nets: int = 10,
    use_ok: bool = True,
    top_k_ok: int = 8,
    reset_ok_nets: bool = False,
    weight_selection: str = "okb",
    initial_ok_task: str = "equal",
    double_timesteps: bool = False,
    seed: int = None,
    d: int = 4,
    run_name: str = None,
    save_dir: str = "./weights/",
    log: bool = True
):

    seed_everything(seed)

    if run_name is None:
        experiment_name = ("OKB" + (" (Random)" if weight_selection == "random" else "")) if use_ok else "SFOLS"
    else:
        experiment_name = run_name

    def make_env():
        env = mo_gym.make(f"mo-pickandplace-d{d}-v2")
        env = FetchDiscreteManhattanAction(env, use_gripper=True)
        env = FetchObservationWrapper(env)
        return env

    env = make_env()
    eval_env = make_env()

    agent = OKB(
        env,
        gamma=0.95,
        learning_rate=3e-4,
        ok_learning_rate=1e-3,
        gradient_updates=g,
        ok_gradient_updates=g,
        num_nets=num_nets,
        crossq=True,
        reset_ok_nets=reset_ok_nets,
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
        epsilon_decay_steps=timesteps_per_iteration,
        learning_starts=1000,
        alpha_per=0.6,
        min_priority=0.1,
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
        timesteps_per_iteration=timesteps_per_iteration * (2 if double_timesteps else 1),
        num_iterations=num_iterations,
        test_tasks=incremental_weights(d, n_partitions=3),
        rep_eval=5,
        save_dir=save_dir,
    )

    agent.close_wandb()


if __name__ == "__main__":
    fire.Fire(main)
