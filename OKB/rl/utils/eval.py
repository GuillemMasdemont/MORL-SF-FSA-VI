from typing import Iterable, List, Optional, Union, Callable

from functools import lru_cache
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
import torch.nn.functional as F
import jax.numpy as jnp
import flax.linen as nn
from gymnasium.spaces import Box, Discrete
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from scipy.spatial import ConvexHull
import wandb as wb
from pymoo.util.ref_dirs import get_reference_directions


def eval(agent, env, render=False):
    obs, _ = env.reset()
    done = False
    total_reward, discounted_return = 0.0, 0.0
    gamma = 1.0
    while not done:
        if render:
            env.render()
        obs, r, terminated, truncated, info = env.step(agent.eval(obs))
        done = terminated or truncated
        total_reward += r
        discounted_return += gamma * r
        gamma *= agent.gamma
    return total_reward, discounted_return

def policy_evaluation(agent, env, rep=5):
    """Returns the mean total reward and discounted return."""
    evals = [eval(agent, env) for _ in range(rep)]
    return np.mean([e[0] for e in evals]), np.mean([e[1] for e in evals])


def eval_phi(agent, env):
    """
    Returns:
        total vec reward, vec return
    """
    obs, _ = env.reset()
    done = False
    total_vec_reward, vec_return = np.zeros(env.reward_dim), np.zeros(env.reward_dim)
    gamma = 1.0
    while not done:
        obs, phi, terminated, truncated, info = env.step(agent.eval(obs, None))
        done = terminated or truncated
        total_vec_reward += phi
        vec_return += gamma * phi
        gamma *= agent.gamma
    return total_vec_reward, vec_return


def eval_mo(agent, env, w):
    """
    Returns:
        w.total_reward, w.return, total vec reward, vec return
    """
    obs, _ = env.reset()
    done = False
    total_vec_reward, vec_return = np.zeros_like(w), np.zeros_like(w)
    gamma = 1.0
    while not done:
        obs, r, terminated, truncated, info = env.step(agent.eval(obs, w))
        done = terminated or truncated
        total_vec_reward += info["vector_reward"]
        vec_return += gamma * info["vector_reward"]
        gamma *= agent.gamma
    return (
        np.dot(w, total_vec_reward),
        np.dot(w, vec_return),
        total_vec_reward,
        vec_return,
    )


def policy_evaluation_mo(agent, env, w, rep=5, return_undiscounted=False, return_scalarized_value=False):
    """Returns vectorized value of the policy (mean of the returns)"""
    if not return_undiscounted:
        if return_scalarized_value:
            returns = [eval_mo(agent, env, w)[1] for _ in range(rep)]
        else:
            returns = [eval_mo(agent, env, w)[3] for _ in range(rep)]
        return np.mean(returns, axis=0)

    else:
        evals = [eval_mo(agent, env, w) for _ in range(rep)]
        if return_scalarized_value:
            returns = [e[1] for e in evals]
            undiscounted_returns = [e[0] for e in evals]
        else:
            returns = [e[3] for e in evals]
            undiscounted_returns = [e[2] for e in evals]
        return (np.mean(returns, axis=0), np.mean(undiscounted_returns, axis=0))


def eval_test_tasks(agent, env, tasks, rep=10):
    """Returns mean scalar value of the policy"""
    returns = [policy_evaluation_mo(agent, env, w, rep=rep, return_scalarized_value=True) for w in tasks]
    return np.mean(returns, axis=0)


def best_vector(values, w):
    max_v = values[0]
    for i in range(1, len(values)):
        if values[i] @ w > max_v @ w:
            max_v = values[i]
    return max_v


def get_non_pareto_dominated_inds(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: The indices of the elements that should be kept to form the Pareto front or coverage set.
    """
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(candidates, return_index=True, return_inverse=True, return_counts=True, axis=0)

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def filter_pareto_dominated(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: A Pareto coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[get_non_pareto_dominated_inds(candidates, remove_duplicates=remove_duplicates)]


def log_pf(front, name="eval/front"):
    filtered_front = list(filter_pareto_dominated(front.copy()))
    reward_dim = filtered_front[0].shape[0]
    front = wb.Table(
        columns=[f"objective_{i}" for i in range(1, reward_dim + 1)],
        data=[p.tolist() for p in filtered_front],
    )
    wb.log({name: front})


def maximum_utility_loss(
    front: List[np.ndarray], reference_set: List[np.ndarray], weights_set: np.ndarray, utility: Callable = np.dot
) -> float:
    """Maximum Utility Loss Metric.

    Maximum utility loss of the policies on the PF for various weights.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the mul on
        reference_set: reference set (e.g. true Pareto front) to compute the mul on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: mul metric
    """
    max_scalarized_values_ref = [np.max([utility(weight, point) for point in reference_set]) for weight in weights_set]
    max_scalarized_values = [np.max([utility(weight, point) for point in front]) for weight in weights_set]
    utility_losses = [max_scalarized_values_ref[i] - max_scalarized_values[i] for i in range(len(max_scalarized_values))]
    return np.max(utility_losses)


def log_all_multi_policy_metrics(
    current_front: List[np.ndarray],
    reward_dim: int,
    global_step: int,
    iteration: int,
    test_tasks: List[np.ndarray],
    ref_front: Optional[List[np.ndarray]] = None,
    id: str = "",
):
    """Logs all metrics for multi-policy training.

    Logged metrics:
    - expected utility metric (EUM)
    If a reference front is provided, also logs:
    - Inverted generational distance (IGD)
    - Maximum utility loss (MUL)

    Args:
        current_front (List) : current Pareto front approximation, computed in an evaluation step
        reward_dim: number of objectives
        global_step: global step for logging
        test_tasks: test tasks. Warning: should be in the same order as the front
        ref_front: reference front, if known
    """
    if id != "":
        id = "_" + id

    returns = [np.dot(sf, wt) for sf, wt in zip(current_front, test_tasks)]
    logs = {}
    for i, wt in enumerate(test_tasks):
        logs[f"eval{id}/{wt}"] = returns[i]

    mean_return = np.mean(returns)
    filtered_front = list(filter_pareto_dominated(current_front))
    eum = expected_utility(filtered_front, weights_set=test_tasks)
    card = len(filtered_front)

    logs.update({
        f"eval{id}/eum": eum,
        f"eval{id}/mean return": mean_return,
        f"eval{id}/cardinality": card,
        "global_step": global_step,
        "iteration": iteration,
    })

    wb.log(logs, commit=False)

    front = wb.Table(
        columns=[f"objective_{i}" for i in range(1, reward_dim + 1)],
        data=[p.tolist() for p in filtered_front],
    )
    wb.log({f"eval{id}/front": front, "iteration": iteration})

    # If PF is known, log the additional metrics
    if ref_front is not None:
        generational_distance = igd(known_front=ref_front, current_estimate=filtered_front)
        mul = maximum_utility_loss(
            front=filtered_front,
            reference_set=ref_front,
            weights_set=test_tasks,
        )
        wb.log({f"eval{id}/igd": generational_distance, f"eval{id}/mul": mul, "iteration": iteration})


def visualize_eval_jax(agent, env, model=None, w=None, horizon=10, init_obs=None, compound=True, deterministic=True, show=False, filename=None):
    from rl.dynamics.util_jax import ModelEnv

    if init_obs is None:
        init_obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    actions = []
    real_obs = []
    real_rewards = []
    real_phis = []
    obs = init_obs.copy()
    for step in range(horizon):
        if w is not None:
            act = agent.eval(obs, w)
        else:
            act = agent.eval(obs)
        actions.append(act)
        obs, r, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        real_obs.append(obs.copy())
        real_rewards.append(r)
        if 'vector_reward' in info:
            real_phis.append(info['vector_reward'])
        if done: 
            break
    
    model_obs = []
    model_obs_stds = []
    model_rewards_stds = []
    model_rewards = []
    if model is not None:
        obs = init_obs.copy()
        model_env = ModelEnv(model, env_id=env.unwrapped.spec.id, rew_dim=1 if w is None else len(w))
        acts = actions
        if isinstance(env.action_space, Discrete):
            acts = nn.one_hot(acts, num_classes=env.action_space.n)
        for step in range(len(real_obs)):
            if compound or step == 0:
                obs, r, done, info = model_env.step(obs, acts[step], deterministic=deterministic)
            else:
                obs, r, done, info = model_env.step(real_obs[step - 1], acts[step], deterministic=deterministic)
            model_obs.append(obs.copy())
            model_obs_stds.append(np.sqrt(info['var_obs'].copy()))
            model_rewards_stds.append(np.sqrt(info['var_rewards'].copy()))
            model_rewards.append(r)
            #if done:
            #    break

    num_plots = obs_dim + (1 if w is None else len(w)) + 1
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))
    x = np.arange(0, len(real_obs))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axs = np.array(axs).reshape(-1)
    for i in range(num_plots):
        if i == num_plots - 1:
            axs[i].set_ylabel(f"Action")
            axs[i].grid(alpha=0.25)
            axs[i].plot(x, [actions[step] for step in x], label='Action', color='orange')
        elif i >= obs_dim:
            axs[i].set_ylabel(f"Reward {i - obs_dim}")
            axs[i].grid(alpha=0.25)
            if w is not None:
                axs[i].plot(x, [real_phis[step][i - obs_dim] for step in x], label='Environment', color='black')
            else:
                axs[i].plot(x, [real_rewards[step] for step in x], label='Environment', color='black')
            if model is not None:
                axs[i].plot(x, [model_rewards[step][i - obs_dim] for step in x], label='Model', color='blue')
                axs[i].fill_between(x, [model_rewards[step][i - obs_dim] + model_rewards_stds[step][i - obs_dim] for step in x], [model_rewards[step][i - obs_dim] - model_rewards_stds[step][i - obs_dim] for step in x], alpha=0.2, facecolor='blue')
        else:
            axs[i].set_ylabel(f"State {i}")
            axs[i].grid(alpha=0.25)
            axs[i].plot(x, [real_obs[step][i] for step in x], label='Environment', color='black')
            if model is not None:
                axs[i].plot(x, [model_obs[step][i] for step in x], label='Model', color='blue')
                axs[i].fill_between(x, [model_obs[step][i] + model_obs_stds[step][i] for step in x], [model_obs[step][i] - model_obs_stds[step][i] for step in x], alpha=0.2, facecolor='blue')
    sns.despine()
    if filename is not None:
        plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    return plt


def visualize_eval(agent, env, model=None, w=None, horizon=10, init_obs=None, compound=True, deterministic=False, show=False, filename=None):
    from rl.dynamics.util import ModelEnv

    if init_obs is None:
        init_obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    actions = []
    real_obs = []
    real_rewards = []
    real_phis = []
    obs = init_obs.copy()
    for step in range(horizon):
        if w is not None:
            act = agent.eval(obs, w)
        else:
            act = agent.eval(obs)
        actions.append(act)
        obs, r, terminated, truncated, info = env.step(act)
        done = terminated or truncated
        real_obs.append(obs.copy())
        real_rewards.append(r)
        if 'vector_reward' in info:
            real_phis.append(info['vector_reward'])
        if done: 
            break
    
    model_obs = []
    model_obs_stds = []
    model_rewards_stds = []
    model_rewards = []
    if model is not None:
        obs = init_obs.copy()
        model_env = ModelEnv(model, env_id=env.unwrapped.spec.id, rew_dim=1 if w is None else len(w))
        acts = th.tensor(actions).to(agent.device)
        if isinstance(env.action_space, Discrete):
            acts = F.one_hot(acts, num_classes=env.action_space.n).squeeze(1)
        for step in range(len(real_obs)):
            if compound or step == 0:
                obs, r, done, info = model_env.step(th.tensor(obs).to(agent.device), acts[step], deterministic=deterministic)
            else:
                obs, r, done, info = model_env.step(th.tensor(real_obs[step - 1]).to(agent.device), acts[step], deterministic=deterministic)
            model_obs.append(obs.copy())
            model_obs_stds.append(np.sqrt(info['var_obs'].copy()))
            model_rewards_stds.append(np.sqrt(info['var_rewards'].copy()))
            model_rewards.append(r)
            #if done:
            #    break

    num_plots = obs_dim + (1 if w is None else len(w)) + 1
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))
    x = np.arange(0, len(real_obs))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axs = np.array(axs).reshape(-1)
    for i in range(num_plots):
        if i == num_plots - 1:
            axs[i].set_ylabel(f"Action")
            axs[i].grid(alpha=0.25)
            axs[i].plot(x, [actions[step] for step in x], label='Action', color='orange')
        elif i >= obs_dim:
            axs[i].set_ylabel(f"Reward {i - obs_dim}")
            axs[i].grid(alpha=0.25)
            if w is not None:
                axs[i].plot(x, [real_phis[step][i - obs_dim] for step in x], label='Environment', color='black')
            else:
                axs[i].plot(x, [real_rewards[step] for step in x], label='Environment', color='black')
            if model is not None:
                axs[i].plot(x, [model_rewards[step][i - obs_dim] for step in x], label='Model', color='blue')
                axs[i].fill_between(x, [model_rewards[step][i - obs_dim] + model_rewards_stds[step][i - obs_dim] for step in x], [model_rewards[step][i - obs_dim] - model_rewards_stds[step][i - obs_dim] for step in x], alpha=0.2, facecolor='blue')
        else:
            axs[i].set_ylabel(f"State {i}")
            axs[i].grid(alpha=0.25)
            axs[i].plot(x, [real_obs[step][i] for step in x], label='Environment', color='black')
            if model is not None:
                axs[i].plot(x, [model_obs[step][i] for step in x], label='Model', color='blue')
                axs[i].fill_between(x, [model_obs[step][i] + model_obs_stds[step][i] for step in x], [model_obs[step][i] - model_obs_stds[step][i] for step in x], alpha=0.2, facecolor='blue')
    sns.despine()
    if filename is not None:
        plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight')
    if show:
        plt.show()
    return plt


@lru_cache
def equally_spaced_weights(dim: int, n: int, seed: int = 42) -> List[np.ndarray]:
    """Generate weight vectors that are equally spaced in the weight simplex.

    It uses the Riesz s-Energy method from pymoo: https://pymoo.org/misc/reference_directions.html

    Args:
        dim: size of the weight vector
        n: number of weight vectors to generate
        seed: random seed
    """
    return list(get_reference_directions("energy", dim, n, seed=seed))


def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    return HV(ref_point=ref_point * - 1)(np.array(points) * - 1)


def igd(known_front: List[np.ndarray], current_estimate: List[np.ndarray]) -> float:
    """Inverted generational distance metric. Requires to know the optimal front.

    Args:
        known_front: known pareto front for the problem
        current_estimate: current pareto front

    Return:
        a float stating the average distance between a point in current_estimate and its nearest point in known_front
    """
    ind = IGD(np.array(known_front))
    return ind(np.array(current_estimate))


def sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def expected_utility(front: List[np.ndarray], weights_set: List[np.ndarray], utility: Callable = np.dot) -> float:
    """Expected Utility Metric.

    Expected utility of the policies on the PF for various weights.
    Similar to R-Metrics in MOO. But only needs one PF approximation.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the eum on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: eum metric
    """
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs))


def make_gif(env, agent, weight: np.ndarray, fullpath: str, fps: int = 50, lenght: int = 300):
    frames = []
    state, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated) and len(frames) < lenght:
        frame = env.render().copy()
        frames.append(frame)
        action = agent.eval(state, weight)
        state, reward, terminated, truncated, info = env.step(action)

    # env.close()

    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(fullpath + '.gif', fps=fps)
    print("Saved gif at: " + fullpath + '.gif')
