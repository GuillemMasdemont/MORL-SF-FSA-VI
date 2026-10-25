import random
from typing import Callable, List, Optional
from functools import partial
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import cvxpy as cp
import gymnasium as gym
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import LinearReward
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints, orbax_utils
flax.config.update('flax_use_orbax_checkpointing', True)
import orbax.checkpoint as ocp
from scipy.optimize import nnls
from datetime import datetime

import wandb as wb
from rl.successor_features.ols import OLS
from rl.rl_algorithm import RLAlgorithm
from rl.utils.buffer import ReplayBuffer
from rl.utils.jax import BatchRenorm, ImpalaEncoder
from rl.utils.eval import eval_mo, eval_phi, policy_evaluation_mo, log_all_multi_policy_metrics, policy_evaluation
from rl.utils.prioritized_buffer import PrioritizedReplayBuffer, OKPrioritizedReplayBuffer
from rl.utils.utils import (linearly_decaying_epsilon, filter_from_list, unique_tol, random_weights, extrema_weights, ObservationNormalizer)


@jax.jit
def unit_norm(z, eps=1e-6):
    return z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), eps)

def unit_norm_np(z, eps=1e-6):
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), eps)

class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict

class OKTrainState(TrainState):
    batch_stats: flax.core.FrozenDict
    target_params: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict

class OKPolicyTrainState(TrainState):
    batch_stats: flax.core.FrozenDict

class OKPolicy(nn.Module):
    phi_dim: int
    batch_norm_momentum: float = 0.99
    num_hidden_layers: int = 2
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, w: jnp.ndarray, train):
        if self.image_obs:
            h_sz = ImpalaEncoder()(obs)
            dummy = BatchRenorm(use_running_average=not train)(h_sz)
        else:
            if self.batch_norm_momentum != 0.0:
                h_sz = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(obs)
            else:
                dummy = BatchRenorm(use_running_average=not train)(h_sz)

        h_sz = nn.Dense(self.hidden_dim)(h_sz)
        h_sz = nn.leaky_relu(h_sz)
        if self.batch_norm_momentum != 0.0:
            h_sz = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h_sz)

        h_sw = nn.Dense(self.hidden_dim)(w)
        h_sw = nn.leaky_relu(h_sw)
        if self.batch_norm_momentum != 0.0:
            h_sw = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h_sw)

        h = h_sz * h_sw
        for _ in range(self.num_hidden_layers - 1):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.leaky_relu(h)
            if self.batch_norm_momentum != 0.0:
                h = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h)

        action = nn.Dense(self.phi_dim)(h)
        action = unit_norm(action)

        return action

class OKSinglePolicy(nn.Module):
    phi_dim: int
    batch_norm_momentum: float = 0.99
    num_hidden_layers: int = 2
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, train: bool):
        if self.image_obs:
            h = ImpalaEncoder()(obs)
            dummy = BatchRenorm(use_running_average=not train)(h)
        else:
            if self.batch_norm_momentum != 0.0:
                h = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(obs)
            else:
                dummy = BatchRenorm(use_running_average=not train)(obs)

        for _ in range(self.num_hidden_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.leaky_relu(h)
            if self.batch_norm_momentum != 0.0:
                h = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h)

        action = nn.Dense(self.phi_dim)(h)
        action = unit_norm(action)

        return action
    

class OKCritic(nn.Module):
    phi_dim: int
    batch_norm_momentum: float = 0.99
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = 0.01
    ofn: bool = True
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray, w: jnp.ndarray, deterministic: bool, train: bool):
        if self.image_obs:
            x = ImpalaEncoder()(obs)
            dummy = BatchRenorm(use_running_average=not train)(x)

            if z.shape[0] != x.shape[0]:
                z = z.reshape((x.shape[0], -1))
            h_sz = jnp.concatenate([x, z], axis=-1)
        else:
            h_sz = jnp.concatenate([obs, z], axis=-1)
            if self.batch_norm_momentum != 0.0:
                h_sz = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h_sz)
            else:
                dummy1 = BatchRenorm(use_running_average=not train)(h_sz)

        h_sz = nn.Dense(self.hidden_dim)(h_sz)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            h_sz = nn.Dropout(rate=self.dropout_rate)(h_sz, deterministic=deterministic)
        if self.use_layer_norm:
            h_sz = nn.LayerNorm()(h_sz)
        h_sz = nn.leaky_relu(h_sz)
        if self.batch_norm_momentum != 0.0:
            h_sz = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h_sz)

        h_sw = nn.Dense(self.hidden_dim)(w)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            h_sw = nn.Dropout(rate=self.dropout_rate)(h_sw, deterministic=deterministic)
        if self.use_layer_norm:
            h_sw = nn.LayerNorm()(h_sw)
        h_sw = nn.leaky_relu(h_sw)
        if self.batch_norm_momentum != 0.0:
            h_sw = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h_sw)

        x = h_sw * h_sz
        for _ in range(self.num_hidden_layers - 1):
            x = nn.Dense(self.hidden_dim)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.leaky_relu(x)
            if self.batch_norm_momentum != 0.0:
                x = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(x)

        if self.ofn:
            x = unit_norm(x)

        x = nn.Dense(self.phi_dim)(x)
        return x

class OKSingleCritic(nn.Module):
    batch_norm_momentum: float = 0.99
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = 0.01
    ofn: bool = True
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray, deterministic: bool, train: bool):
        if self.image_obs:
            h_s = ImpalaEncoder()(obs)
            dummy = BatchRenorm(use_running_average=not train)(h_s)

            if z.shape[0] != h_s.shape[0]:
                z = z.reshape((h_s.shape[0], -1))
            h_sz = jnp.concatenate([h_s, z], axis=-1)
        else:
            h_sz = jnp.concatenate([obs, z], axis=-1)
            if self.batch_norm_momentum != 0.0:
                h_sz = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h_sz)
            else:
                dummy1 = BatchRenorm(use_running_average=not train)(h_sz)

        x = h_sz
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.leaky_relu(x)
            if self.batch_norm_momentum != 0.0:
                x = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(x)

        if self.ofn:
            x = unit_norm(x)

        x = nn.Dense(1)(x)
        return x

class VectorOKSingleCritic(nn.Module):
    batch_norm_momentum: float = 0.99
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = 0.01
    ofn: bool = True
    n_critics: int = 2
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray, deterministic: bool, train: bool):
        vmap_critic = nn.vmap(
            OKSingleCritic,
            variable_axes={"params": 0, "batch_stats": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "batch_stats": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            batch_norm_momentum=self.batch_norm_momentum,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            ofn=self.ofn,
            num_hidden_layers=self.num_hidden_layers,
            hidden_dim=self.hidden_dim,
            image_obs=self.image_obs,
            )(obs, z, deterministic, train)
        return q_values.reshape((self.n_critics, -1))

class VectorOKCritic(nn.Module):
    phi_dim: int
    batch_norm_momentum: float = 0.99
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = 0.01
    ofn: bool = True
    n_critics: int = 2
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray, w: jnp.ndarray, deterministic: bool, train: bool):
        vmap_critic = nn.vmap(
            OKCritic,
            variable_axes={"params": 0, "batch_stats": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "batch_stats": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            phi_dim=self.phi_dim,
            batch_norm_momentum=self.batch_norm_momentum,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            ofn=self.ofn,
            num_hidden_layers=self.num_hidden_layers,
            hidden_dim=self.hidden_dim,
            image_obs=self.image_obs,
            )(obs, z, w, deterministic, train)
        return q_values.reshape((self.n_critics, -1, self.phi_dim))


class Psi(nn.Module):
    action_dim: int
    rew_dim: int
    dropout_rate: Optional[float] = 0.01
    use_layer_norm: bool = True
    ofn: bool = True
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, w: jnp.ndarray, deterministic: bool):
        if self.image_obs:
            h_obs = ImpalaEncoder()(obs)
        else:
            h_obs = obs

        h_obs = nn.Dense(self.hidden_dim)(h_obs)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            h_obs = nn.Dropout(rate=self.dropout_rate)(h_obs, deterministic=deterministic)
        if self.use_layer_norm:
            h_obs = nn.LayerNorm()(h_obs)
        h_obs = nn.leaky_relu(h_obs)

        h_w = nn.Dense(self.hidden_dim)(w)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            h_w = nn.Dropout(rate=self.dropout_rate)(h_w, deterministic=deterministic)
        if self.use_layer_norm:
            h_w = nn.LayerNorm()(h_w)
        h_w = nn.leaky_relu(h_w)

        h = h_obs * h_w    # h = jnp.concatenate([h_obs, h_w], axis=-1)
        for _ in range(self.num_hidden_layers - 1):
            h = nn.Dense(self.hidden_dim)(h)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
            if self.use_layer_norm:
                h = nn.LayerNorm()(h)
            h = nn.relu(h)  # Leaky relu here leads to NaN due to very small negative values in unit_norm

        if self.ofn:
            h = unit_norm(h, eps=1e-4)

        psi = nn.Dense(self.action_dim * self.rew_dim)(h)
        return psi


class SingleQ(nn.Module):
    action_dim: int
    dropout_rate: Optional[float] = 0.01
    use_layer_norm: bool = True
    ofn: bool = True
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, deterministic: bool):
        if self.image_obs:
            h = ImpalaEncoder()(obs) # NatureCNN(use_layer_norm=self.use_layer_norm)(obs)
        else:
            h = obs

        for _ in range(self.num_hidden_layers - 1):
            h = nn.Dense(self.hidden_dim)(h)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
            if self.use_layer_norm:
                h = nn.LayerNorm()(h)
            h = nn.relu(h)  # Leaky relu here leads to NaN due to very small negative values in unit_norm

        if self.ofn:
            h = unit_norm(h, eps=1e-4)

        q = nn.Dense(self.action_dim)(h)
        return q

class VectorSingleQ(nn.Module):
    action_dim: int
    use_layer_norm: bool = True
    ofn: bool = True
    dropout_rate: Optional[float] = 0.01
    n_critics: int = 2
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, deterministic: bool):
        vmap_critic = nn.vmap(
            SingleQ,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            action_dim=self.action_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
            ofn=self.ofn,
            num_hidden_layers=self.num_hidden_layers,
            hidden_dim=self.hidden_dim,
            image_obs=self.image_obs,
            )(obs, deterministic)
        return q_values.reshape((self.n_critics, -1, self.action_dim))


class VectorPsi(nn.Module):
    action_dim: int
    rew_dim: int
    use_layer_norm: bool = True
    ofn: bool = True
    dropout_rate: Optional[float] = 0.01
    n_critics: int = 2
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, w: jnp.ndarray, deterministic: bool):
        vmap_critic = nn.vmap(
            Psi,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            action_dim=self.action_dim,
            rew_dim=self.rew_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
            ofn=self.ofn,
            num_hidden_layers=self.num_hidden_layers,
            hidden_dim=self.hidden_dim,
            image_obs=self.image_obs,
            )(obs, w, deterministic)
        return q_values.reshape((self.n_critics, -1, self.action_dim, self.rew_dim))


class OKB(RLAlgorithm):
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        ok_learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        ucb_exploration: float = 0.0,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        ok_policy_noise: float = 0.2,
        tau: float = 1.0,
        target_net_update_freq: int = 1000,  # ignored if tau != 1.0
        buffer_size: int = int(1e6),
        net_arch: List = [256, 256, 256],
        ok_net_arch: List = [256, 256, 256],
        num_nets: int = 2,
        batch_size: int = 256,
        learning_starts: int = 1000,
        phi_dim: int = 16,
        use_ok: bool = False,
        weight_selection: str = 'okb',
        gradient_updates: int = 10,
        ok_gradient_updates: int = 10,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = 1.0,
        use_gpi: bool = True,
        n_step: int = 1,
        top_k_ok: int = 32,
        top_k_base: int = 1,
        num_ok_iterations: int = 5,
        initial_ok_task: str = "one-hot", # or "equal"
        gpi_type: str = "gpi",
        lcb_pessimism: float = 0.0,
        per: bool = True,
        alpha_per: float = 0.6,
        min_priority: float = 1.0,
        drop_rate: float = 0.01,
        batch_norm_momentum: float = 0.99,
        layer_norm: bool = True,
        crossq: bool = True,
        reset_ok_nets: bool = False,
        normalize_obs: bool = False,
        seed: int = 0,
        project_name: str = "OLSO",
        experiment_name: str = "OLSO",
        log: bool = True,
        device = None
        ):
        super().__init__(env, experiment_name=experiment_name, project_name=project_name, device=device)
        self.use_ok = use_ok
        self.use_okgpi = False
        self.use_single_ok = False
        self.use_single_ok_discrete = False
        self.weight_selection = weight_selection
        self._ok_policy_updates = 0
        self.phi_dim = self.env.unwrapped.reward_dim
        self.learning_rate = learning_rate
        self.ok_learning_rate = ok_learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.offset_epsilon = 0
        self.ucb_exploration = ucb_exploration
        self.ok_policy_noise = ok_policy_noise
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.use_gpi = use_gpi
        self.gpi_type = gpi_type
        self.include_w = False
        self.lcb_pessimism = lcb_pessimism
        self.n_step = n_step
        self.top_k_ok = top_k_ok
        self.top_k_base = top_k_base
        self.num_ok_iterations = num_ok_iterations
        self.initial_ok_task = initial_ok_task
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.ok_net_arch = ok_net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.ok_gradient_updates = ok_gradient_updates
        self.num_nets = num_nets
        self.drop_rate = drop_rate
        self.batch_norm_momentum = batch_norm_momentum
        self.crossq = crossq
        self.normalize_obs = normalize_obs
        self.reset_ok_nets = reset_ok_nets
        self.layer_norm = layer_norm
        self.fitted_w = None

        if self.normalize_obs:
            self.obs_normalizer = ObservationNormalizer(shape=env.observation_space.shape, dtype=np.float32)

        if seed is None:
            self.seed = random.randint(0, int(1e6))
        else:
            self.seed = seed
        key = jax.random.PRNGKey(self.seed)
        self.key, psi_key, ok_key, bn_key, dropout_key = jax.random.split(key, 5)

        obs = env.observation_space.sample()
        w = np.zeros(self.phi_dim, dtype=np.float32)
        self.image_obs = len(obs.shape) > 2
        self.psi = VectorPsi(
            action_dim=self.action_dim, 
            rew_dim=self.phi_dim, 
            use_layer_norm=self.layer_norm, 
            dropout_rate=self.drop_rate if self.num_nets == 2 else 0.0,
            ofn=True,
            n_critics=self.num_nets, 
            num_hidden_layers=len(self.net_arch),
            hidden_dim=self.net_arch[0],
            image_obs=self.image_obs
        )
        self.psi_state = RLTrainState.create(
            apply_fn=self.psi.apply,
            params=self.psi.init(
                {"params": psi_key, "dropout": dropout_key},
                obs,
                w,
                deterministic=False,
            ),
            target_params=self.psi.init(
                {"params": psi_key, "dropout": dropout_key},
                obs,
                w,
                deterministic=False,
            ),
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.psi.apply = jax.jit(self.psi.apply, static_argnames=("dropout_rate", "use_layer_norm", "deterministic", "image_obs", "ofn"))

        if self.use_ok:
            self.ok_policy = OKPolicy(
                self.phi_dim,
                batch_norm_momentum=self.batch_norm_momentum if self.crossq else 0.0,
                num_hidden_layers=len(self.ok_net_arch), 
                hidden_dim=self.ok_net_arch[0],
                image_obs=self.image_obs
            )
            self.ok_critic = VectorOKCritic(
                self.phi_dim,
                batch_norm_momentum=self.batch_norm_momentum if self.crossq else 0.0,
                use_layer_norm=self.layer_norm and not self.crossq,
                num_hidden_layers=len(self.ok_net_arch),
                n_critics=2,
                ofn=True,
                dropout_rate=self.drop_rate,
                hidden_dim=self.ok_net_arch[0] * (2 if self.crossq else 1),
                image_obs=self.image_obs
            ) 
            ok_policy_init_variables = self.ok_policy.init({"params": ok_key, "batch_stats": bn_key}, obs, w, train=False)
            self.ok_policy_state = OKPolicyTrainState.create(
                apply_fn=self.ok_policy.apply,
                params=ok_policy_init_variables["params"],
                batch_stats=ok_policy_init_variables["batch_stats"],
                tx=optax.adam(learning_rate=self.ok_learning_rate, b1=0.5 if self.crossq else 0.9)
                #tx=optax.adamw(learning_rate=self.learning_rate, b1=0.5 if self.crossq else 0.9, weight_decay=1e-2),
                #tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=self.learning_rate, b1=0.5 if self.crossq else 0.9)),
            )

            ok_critic_init_variables = self.ok_critic.init({"params": ok_key, 
                                                            "batch_stats": bn_key,
                                                            "dropout": dropout_key},
                                                            obs, w, w, deterministic=False, train=False)
            ok_targetcritic_init_variables = self.ok_critic.init({"params": ok_key,
                                                            "batch_stats": bn_key,
                                                            "dropout": dropout_key},
                                                            obs, w, w, deterministic=False, train=False)
            self.ok_critic_state = OKTrainState.create(
                apply_fn=self.ok_critic.apply,
                params=ok_critic_init_variables["params"],
                target_params=ok_targetcritic_init_variables["params"],
                batch_stats=ok_critic_init_variables["batch_stats"],
                target_batch_stats=ok_targetcritic_init_variables["batch_stats"],
                tx=optax.adam(learning_rate=self.ok_learning_rate, b1=0.5 if self.crossq else 0.9),
                #tx=optax.adamw(learning_rate=self.learning_rate, b1=0.5 if self.crossq else 0.9, weight_decay=1e-2),
                #tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=self.learning_rate, b1=0.5 if self.crossq else 0.9)),
            )
            self.ok_policy.apply = jax.jit(self.ok_policy.apply, static_argnames=("image_obs", "batch_norm_momentum",))
            self.ok_critic.apply = jax.jit(self.ok_critic.apply, static_argnames=("image_obs", "batch_norm_momentum", "use_layer_norm", "dropout_rate", "ofn", "deterministic"))
        else:
            self.ok_policy = None

        @jax.jit
        def gpi_from_psi(psi, w):
            q = (psi * w.reshape(1, 1, w.shape[0])).sum(axis=2)
            max_q = q.max(axis=1)
            policy_index = max_q.argmax()
            action = q[policy_index].argmax()
            return action
        vgpipsi = jax.jit(jax.vmap(gpi_from_psi, in_axes=(None, 0)))
        self.vgpipsi = vgpipsi

        @partial(jax.jit, static_argnames=["psi"])
        def gpi(psi, psi_state, obs, w_ok, M):
            M = jnp.stack(M)
            obs_m = obs.reshape(1, *obs.shape).repeat(M.shape[0], axis=0)
            psi_values = psi.apply(psi_state.params, obs_m, M, deterministic=True)
            q_values = (psi_values * w_ok.reshape(1, 1, 1, w_ok.shape[0])).sum(axis=3)
            q_values = q_values.mean(axis=0)
            max_q = q_values.max(axis=1)
            policy_index = max_q.argmax()  # max_i max_a q(s,a,w_i)
            action = q_values[policy_index].argmax()
            return action, psi_values.mean(axis=0)
        self.vgpi = jax.jit(jax.vmap(gpi, in_axes=(None, None, 0, 0, None)), static_argnames=("psi"))

        def compute_value_gap(ok_policy_state, ok_critic_state, psi, psi_state, obs, phi, next_obs, done, w, M, gamma, key):
            action_ok, w_ok, _, _ = OKB.ok_action(ok_policy_state, psi, psi_state, obs, w, M, key, 0.0, add_noise=False)
            next_action_ok, next_w_ok, _, _ = OKB.ok_action(ok_policy_state, psi, psi_state, next_obs, w, M, key, 0.0, add_noise=False)

            sf = ok_critic_state.apply_fn({"params": ok_critic_state.params, "batch_stats": ok_critic_state.batch_stats}, obs, w_ok, w, deterministic=True, train=False)
            next_sf = ok_critic_state.apply_fn({"params": ok_critic_state.params, "batch_stats": ok_critic_state.batch_stats}, next_obs, next_w_ok, w, deterministic=True, train=False)

            sf = sf.mean(axis=0)
            value = (sf * w).sum(axis=1)
            next_sf = next_sf.mean(axis=0)
            next_sf = phi + gamma * (1 - done) * next_sf
            next_value = (next_sf * w).sum(axis=1)
            return (next_value - value) #, eq2
        self.vcompute_value_gap = jax.jit(jax.vmap(compute_value_gap, in_axes=(None, None, None, None, 0, 0, 0, 0, None, None, None, None)), static_argnames=("psi", "gamma"))

        def find_z(psi, psi_state, obs, w, M, key):
            psi_values = OKB.get_psis(psi, psi_state, obs, jnp.stack(M))
            # AQUI test_zs = jnp.array(equally_spaced_weights(self.phi_dim, 100))
            test_zs = jnp.concatenate([w.reshape(1, w.shape[0]), jnp.stack(M), jnp.eye(w.shape[0]), jax.random.normal(key, (1024, w.shape[0]))], axis=0)
            test_zs = unit_norm(test_zs)
            actions_ok = vgpipsi(psi_values, test_zs)
            return actions_ok, test_zs
        self.find_z = jax.jit(find_z, static_argnames=("psi"))

        self.per = per
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(self.observation_shape, 1, rew_dim=self.env.unwrapped.reward_dim, max_size=self.buffer_size, action_dtype=np.uint8)
        else:
            self.replay_buffer = ReplayBuffer(self.observation_shape, 1, rew_dim=self.env.unwrapped.reward_dim, max_size=self.buffer_size, action_dtype=np.uint8)
        if self.use_ok:
            self.ok_replay_buffer = OKPrioritizedReplayBuffer(self.observation_shape, 1, rew_dim=self.phi_dim, max_size=self.buffer_size, action_dtype=np.uint8, prioritized=False)
        self.min_priority = min_priority
        self.alpha = alpha_per
        self.M = []
        self.M_ok = []

        self.policy_inds = []
        self.meta_actions = []
        self.action_hist = []
        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name)

    def setup_single_ok(self):
        self.key, ok_key, bn_key, dropout_key = jax.random.split(self.key, 4)
        obs = self.env.observation_space.sample()
        z = np.zeros(self.phi_dim, dtype=np.float32)

        self.ok_single_policy = OKSinglePolicy(
                self.phi_dim,
                batch_norm_momentum=self.batch_norm_momentum if self.crossq else 0.0,
                num_hidden_layers=len(self.ok_net_arch), 
                hidden_dim=self.ok_net_arch[0],
                image_obs=self.image_obs
            )
        self.ok_single_critic = VectorOKSingleCritic(
                batch_norm_momentum=self.batch_norm_momentum if self.crossq else 0.0,
                use_layer_norm=self.layer_norm and not self.crossq,
                num_hidden_layers=len(self.ok_net_arch),
                n_critics=2,
                dropout_rate=self.drop_rate,
                hidden_dim=self.ok_net_arch[0] * (2 if self.crossq else 1), # wider critic, as in CrossQ?
                image_obs=self.image_obs
        ) 
        ok_policy_init_variables = self.ok_single_policy.init({"params": ok_key, "batch_stats": bn_key}, obs, train=False)
        self.ok_single_policy_state = OKPolicyTrainState.create(
            apply_fn=self.ok_single_policy.apply,
            params=ok_policy_init_variables["params"],
            batch_stats=ok_policy_init_variables["batch_stats"],
            tx=optax.adam(learning_rate=self.ok_learning_rate, b1=0.5 if self.crossq else 0.9),
        )

        ok_critic_init_variables = self.ok_single_critic.init({"params": ok_key, 
                                                        "batch_stats": bn_key,
                                                        "dropout": dropout_key},
                                                        obs, z, deterministic=False, train=False)
        ok_targetcritic_init_variables = self.ok_single_critic.init({"params": ok_key,
                                                        "batch_stats": bn_key,
                                                        "dropout": dropout_key},
                                                        obs, z, deterministic=False, train=False)
        self.ok_single_critic_state = OKTrainState.create(
            apply_fn=self.ok_single_critic.apply,
            params=ok_critic_init_variables["params"],
            target_params=ok_targetcritic_init_variables["params"],
            batch_stats=ok_critic_init_variables["batch_stats"],
            target_batch_stats=ok_targetcritic_init_variables["batch_stats"],
            tx=optax.adam(learning_rate=self.ok_learning_rate, b1=0.5 if self.crossq else 0.9),
            #tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=self.learning_rate, b1=0.5 if self.crossq else 0.9)),
        )
        self.ok_single_policy.apply = jax.jit(self.ok_single_policy.apply, static_argnames=("image_obs", "batch_norm_momentum",))
        self.ok_single_critic.apply = jax.jit(self.ok_single_critic.apply, static_argnames=("image_obs", "batch_norm_momentum", "use_layer_norm", "dropout_rate", "ofn", "deterministic"))

        self.ok_single_replay_buffer = OKPrioritizedReplayBuffer(self.observation_shape, 1, rew_dim=self.phi_dim, max_size=self.buffer_size, scalar_reward=True, action_dtype=np.uint8, prioritized=False)

    def setup_single_ok_discrete(self):
        self.key, q_key, dropout_key = jax.random.split(self.key, 3)
        obs = self.env.observation_space.sample()
        self.q_net = VectorSingleQ(
            action_dim=5,
            use_layer_norm=self.layer_norm,
            dropout_rate=self.drop_rate if self.num_nets == 2 else 0.0,
            ofn=True,
            n_critics=1,
            num_hidden_layers=len(self.net_arch),
            hidden_dim=self.net_arch[0],
            image_obs=self.image_obs
        )
        self.q_state = RLTrainState.create(
            apply_fn=self.q_net.apply,
            params=self.q_net.init(
                {"params": q_key, "dropout": dropout_key},
                obs,
                deterministic=False,
            ),
            target_params=self.q_net.init(
                {"params": q_key, "dropout": dropout_key},
                obs,
                deterministic=False,
            ),
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.q_net.apply = jax.jit(self.q_net.apply, static_argnames=("dropout_rate", "use_layer_norm", "deterministic", "image_obs", "ofn"))

        self.ok_single_replay_buffer = OKPrioritizedReplayBuffer(self.observation_shape, 1, rew_dim=1, max_size=self.buffer_size, scalar_reward=True, action_dtype=np.uint8, meta_action_dtype=np.uint8, prioritized=False)

    def get_config(self):
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "ok_learning_rate": self.ok_learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "ok_policy_noise": self.ok_policy_noise,
            "batch_size": self.batch_size,
            "per": self.per,
            "use_ok": self.use_ok,
            "weight_selection": self.weight_selection,
            "alpha_per": self.alpha,
            "min_priority": self.min_priority,
            "tau": self.tau,
            "num_nets": self.num_nets,
            "max_grad_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "ok_net_arch": self.ok_net_arch,
            "gradient_updates": self.gradient_updates,
            "ok_gradient_updates": self.ok_gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "top_k_ok": self.top_k_ok,
            "top_k_base": self.top_k_base,
            "initial_ok_task": self.initial_ok_task,
            "reset_ok_nets": self.reset_ok_nets,
            "num_ok_iterations": self.num_ok_iterations,
            "drop_rate": self.drop_rate,
            "layer_norm": self.layer_norm,
            "seed": self.seed,
        }

    def save(self, save_dir="weights/", filename=None):
        # orbax-checkpoint requires an absolute path; the CLI scripts default to "./weights/".
        save_dir = os.path.abspath(save_dir) + os.sep
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {}
        saved_params["psi_net_state"] = self.psi_state
        saved_params["M"] = self.M
        saved_params["M_ok"] = self.M_ok
        if self.ok_policy is not None:
            saved_params["ok_policy_state"] = self.ok_policy_state
            saved_params["ok_critic_state"] = self.ok_critic_state

        filename = self.experiment_name if filename is None else filename
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(saved_params)
        orbax_checkpointer.save(save_dir + filename, saved_params, save_args=save_args, force=True)

    def save_single_ok(self, save_dir="weights/", filename=None):
        save_dir = os.path.abspath(save_dir) + os.sep
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {}
        saved_params["ok_single_policy_state"] = self.ok_single_policy_state
        saved_params["ok_single_critic_state"] = self.ok_single_critic_state

        filename = self.experiment_name if filename is None else filename
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(saved_params)
        orbax_checkpointer.save(save_dir + filename, saved_params, save_args=save_args, force=True)

    def load(self, path, step=None):
        target = {"psi_net_state": self.psi_state, "M": self.M, "M_ok": self.M_ok}
        if self.use_ok:
            target["ok_policy_state"] = self.ok_policy_state
            target["ok_critic_state"] = self.ok_critic_state

        ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        restored = ckptr.restore(path, item=None)
        
        target['M'] = restored['M']
        target['M_ok'] = restored['M_ok']

        restored = ckptr.restore(path, item=target, restore_args=flax.training.orbax_utils.restore_args_from_target(target, mesh=None))

        self.psi_state = restored["psi_net_state"]
        if self.use_ok:
            self.ok_policy_state = restored["ok_policy_state"]
            self.ok_critic_state = restored["ok_critic_state"]
        self.M = [w for w in restored["M"]]
        if type(restored["M_ok"]) is dict:
            self.M_ok = [w for w in restored["M_ok"].values()]
        else:
            self.M_ok = [w for w in restored["M_ok"]]

    def sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=False, device=self.device)

    @staticmethod
    @partial(jax.jit, static_argnames=["psi", "pessimism", "return_q_values"])
    def batch_gpi(psi, psi_state, obs, w, pessimism, M, key, return_q_values=False):
        M_stack = jnp.stack(M)
        M_stack = M_stack.reshape(1, M_stack.shape[0], M_stack.shape[1]).repeat(len(obs), axis=0)
        obs_m = obs.reshape(obs.shape[0], 1, *obs.shape[1:]).repeat(M_stack.shape[1], axis=1)

        psi_values = psi.apply(psi_state.params, obs_m, M_stack, deterministic=True)
        q_values = (psi_values * w).sum(axis=3).reshape(psi_values.shape[0], obs.shape[0], len(M), -1)
        q_values = q_values.mean(axis=0)

        max_q = jnp.max(q_values, axis=2)
        pi = jnp.argmax(max_q, axis=1)
        best_q_values = q_values[jnp.arange(q_values.shape[0]), pi]
        acts = best_q_values.argmax(axis=1)

        if return_q_values:
            return acts, best_q_values[jnp.arange(q_values.shape[0]), acts], key

        return acts, key

    @staticmethod
    @jax.jit
    def update_ok_policy(ok_policy_state, ok_critic_state, w, obs, key):
        key, drop_key = jax.random.split(key, 2)

        def actor_loss(params, batch_stats, drop_key):
            actions, state_updates = ok_policy_state.apply_fn({"params": params, "batch_stats": batch_stats}, obs, w, mutable=["batch_stats"], train=True)
            psi = ok_critic_state.apply_fn({"params": ok_critic_state.params, "batch_stats": ok_critic_state.batch_stats}, obs, actions, w, rngs={"dropout": drop_key}, deterministic=False, train=False)
            q = (psi * w).sum(axis=2)
            loss = -q.mean()
            return loss, state_updates
        
        (actor_loss_value, state_updates), grads = jax.value_and_grad(actor_loss, has_aux=True)(ok_policy_state.params, ok_policy_state.batch_stats, drop_key)
        ok_policy_state = ok_policy_state.apply_gradients(grads=grads)
        ok_policy_state = ok_policy_state.replace(batch_stats=state_updates["batch_stats"])
        return ok_policy_state, actor_loss_value, key

    @staticmethod
    @jax.jit
    def update_ok_single_policy(ok_policy_state, ok_critic_state, obs, key):
        key, drop_key = jax.random.split(key, 2)

        def actor_loss(params, batch_stats, drop_key):
            actions, state_updates = ok_policy_state.apply_fn({"params": params, "batch_stats": batch_stats}, obs, mutable=["batch_stats"], train=True)
            q = ok_critic_state.apply_fn({"params": ok_critic_state.params, "batch_stats": ok_critic_state.batch_stats}, obs, actions, rngs={"dropout": drop_key}, deterministic=False, train=False)
            loss = -q.mean()
            return loss, state_updates
        
        (actor_loss_value, state_updates), grads = jax.value_and_grad(actor_loss, has_aux=True)(ok_policy_state.params, ok_policy_state.batch_stats, drop_key)
        ok_policy_state = ok_policy_state.apply_gradients(grads=grads)
        ok_policy_state = ok_policy_state.replace(batch_stats=state_updates["batch_stats"])
        return ok_policy_state, actor_loss_value, key

    @staticmethod
    @partial(jax.jit, static_argnames=["gamma", "kappa", "crossq"])
    def update_ok_critic(ok_policy_state, ok_critic_state, w, obs, meta_actions, rewards, next_obs, dones, gamma, kappa, crossq, key):
        key, noise_key, drop_key, drop_key_target = jax.random.split(key, 4)

        noise = jax.random.normal(noise_key, meta_actions.shape, dtype=jnp.float32) * 0.01
        noise = jnp.clip(noise, -0.05, 0.05)
        next_actions = ok_policy_state.apply_fn({"params": ok_policy_state.params, "batch_stats": ok_policy_state.batch_stats}, next_obs, w, train=False)
        next_actions = next_actions + noise # jnp.clip(next_actions + noise, -1.0, 1.0)
        next_actions = unit_norm(next_actions)

        def mse_loss(params, batch_stats, drop_key):
            if crossq:
                catted_q_values, state_updates = ok_critic_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([obs, next_obs], axis=0),
                    jnp.concatenate([meta_actions, next_actions], axis=0),
                    jnp.concatenate([w, w], axis=0),
                    mutable=["batch_stats"],
                    rngs={"dropout": drop_key},
                    deterministic=False,
                    train=True,
                )
                current_psi_values, next_psi_values = jnp.split(catted_q_values, 2, axis=1)
                next_q_values = (next_psi_values * w).sum(axis=2)
            else:
                current_psi_values, state_updates = ok_critic_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    obs,
                    meta_actions,
                    w,
                    mutable=["batch_stats"],
                    rngs={"dropout": drop_key},
                    deterministic=False,
                    train=True,
                )
                next_psi_values = ok_critic_state.apply_fn(
                    {"params": ok_critic_state.target_params, "batch_stats": ok_critic_state.target_batch_stats},
                    next_obs,
                    next_actions,
                    w,
                    rngs={"dropout": drop_key_target},
                    deterministic=False,
                    train=False,
                )
                next_q_values = (next_psi_values * w).sum(axis=2)
                
            min_ind = next_q_values.argmin(axis=0)
            next_psi_values = jnp.take_along_axis(next_psi_values, min_ind[None, ..., None], axis=0).squeeze(0)
            target = rewards + (1 - dones) * gamma * next_psi_values
            tds = current_psi_values - jax.lax.stop_gradient(target)
            loss = jnp.abs(tds)
            loss = jnp.where(loss < kappa, 0.5 * loss ** 2, loss * kappa).mean()
            return loss, (state_updates, tds)
        
        (loss_value, (state_updates, td_error)), grads = jax.value_and_grad(mse_loss, has_aux=True)(ok_critic_state.params, ok_critic_state.batch_stats, drop_key)
        ok_critic_state = ok_critic_state.apply_gradients(grads=grads)
        ok_critic_state = ok_critic_state.replace(batch_stats=state_updates["batch_stats"])

        if not crossq:
            ok_critic_state = ok_critic_state.replace(target_params=optax.incremental_update(ok_critic_state.params, ok_critic_state.target_params, 0.005))
            ok_critic_state = ok_critic_state.replace(target_batch_stats=optax.incremental_update(ok_critic_state.batch_stats, ok_critic_state.target_batch_stats, 0.005))

        return ok_critic_state, loss_value, td_error, key

    @staticmethod
    @partial(jax.jit, static_argnames=["gamma", "kappa", "crossq"])
    def update_ok_single_critic(ok_policy_state, ok_critic_state, obs, meta_actions, rewards, next_obs, dones, gamma, kappa, crossq, key):
        key, noise_key, drop_key, drop_key_target = jax.random.split(key, 4)

        noise = jax.random.normal(noise_key, meta_actions.shape, dtype=jnp.float32) * 0.01
        noise = jnp.clip(noise, -0.05, 0.05)
        next_actions = ok_policy_state.apply_fn({"params": ok_policy_state.params, "batch_stats": ok_policy_state.batch_stats}, next_obs, train=False)
        next_actions = next_actions + noise
        next_actions = unit_norm(next_actions)

        def mse_loss(params, batch_stats, drop_key):
            if crossq:
                catted_q_values, state_updates = ok_critic_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([obs, next_obs], axis=0),
                    jnp.concatenate([meta_actions, next_actions], axis=0),
                    mutable=["batch_stats"],
                    rngs={"dropout": drop_key},
                    deterministic=False,
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)
            else:
                current_q_values, state_updates = ok_critic_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    obs,
                    meta_actions,
                    mutable=["batch_stats"],
                    rngs={"dropout": drop_key},
                    deterministic=False,
                    train=True,
                )
                next_q_values = ok_critic_state.apply_fn(
                    {"params": ok_critic_state.target_params, "batch_stats": ok_critic_state.target_batch_stats},
                    next_obs,
                    next_actions,
                    rngs={"dropout": drop_key_target},
                    deterministic=False,
                    train=False,
                )
            min_next_q = next_q_values.min(axis=0).reshape(-1, 1)
            target = rewards + (1 - dones) * gamma * min_next_q
            target = target.reshape(1, -1)
            tds = current_q_values - jax.lax.stop_gradient(target)
            loss = jnp.abs(tds)
            loss = jnp.where(loss < kappa, 0.5 * loss ** 2, loss * kappa).mean()
            return loss, (state_updates, tds)
        
        (loss_value, (state_updates, td_error)), grads = jax.value_and_grad(mse_loss, has_aux=True)(ok_critic_state.params, ok_critic_state.batch_stats, drop_key)
        ok_critic_state = ok_critic_state.apply_gradients(grads=grads)
        ok_critic_state = ok_critic_state.replace(batch_stats=state_updates["batch_stats"])

        if not crossq:
            ok_critic_state = ok_critic_state.replace(target_params=optax.incremental_update(ok_critic_state.params, ok_critic_state.target_params, 0.005))
            ok_critic_state = ok_critic_state.replace(target_batch_stats=optax.incremental_update(ok_critic_state.batch_stats, ok_critic_state.target_batch_stats, 0.005))

        return ok_critic_state, loss_value, td_error, key

    @staticmethod
    @partial(jax.jit, static_argnames=["psi", "gamma", "min_priority"])
    def update(psi, psi_state, w, obs, actions, rewards, next_obs, dones, gamma, min_priority, key):
        key, inds_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)

        # DroQ update
        if psi.n_critics >= 2:
            psi_values_next = psi.apply(psi_state.target_params, next_obs, w, deterministic=False, rngs={"dropout": dropout_key_target})
            if psi_values_next.shape[0] > 2:
                inds = jax.random.randint(inds_key, (2,), 0, psi_values_next.shape[0])
                psi_values_next = psi_values_next[inds]
            q_values_next = (psi_values_next * w.reshape(w.shape[0], 1, w.shape[1])).sum(axis=3)
            min_inds = q_values_next.argmin(axis=0)
            min_psi_values = jnp.take_along_axis(psi_values_next, min_inds[None,...,None], 0).squeeze(0)
            
            max_q = (min_psi_values * w.reshape(w.shape[0], 1, w.shape[1])).sum(axis=2)
            max_acts = max_q.argmax(axis=1)
            target = min_psi_values[jnp.arange(min_psi_values.shape[0]), max_acts]

            def mse_loss(params, droptout_key):
                psi_values = psi.apply(params, obs, w, deterministic=False, rngs={"dropout": droptout_key})
                psi_values = psi_values[:, jnp.arange(psi_values.shape[1]), actions.squeeze()]
                tds = psi_values - target_psi
                loss = jnp.abs(tds)
                loss = jnp.where(loss < min_priority, 0.5 * loss ** 2, loss * min_priority)
                return loss.mean(), tds
        # DDQN update
        else:
            psi_values_next = psi.apply(psi_state.target_params, next_obs, w, deterministic=True)[0]
            psi_values_not_target = psi.apply(psi_state.params, next_obs, w, deterministic=True)
            q_values_next = (psi_values_not_target * w.reshape(w.shape[0], 1, w.shape[1])).sum(axis=3)[0]
            max_acts = q_values_next.argmax(axis=1)
            target = psi_values_next[jnp.arange(psi_values_next.shape[0]), max_acts]

            def mse_loss(params, droptout_key):
                psi_values = psi.apply(params, obs, w, deterministic=True)
                psi_values = psi_values[:, jnp.arange(psi_values.shape[1]), actions.squeeze()]
                tds = psi_values - target_psi
                loss = jnp.abs(tds)
                loss = jnp.where(loss < min_priority, 0.5 * loss ** 2, loss * min_priority)
                return loss.mean(), tds

        target_psi = rewards + (1 - dones) * gamma * target

        (loss_value, td_error), grads = jax.value_and_grad(mse_loss, has_aux=True)(psi_state.params, dropout_key_current)
        psi_state = psi_state.apply_gradients(grads=grads)

        return psi_state, loss_value, td_error, key
    
    def train_phi(self):
        for g in range(10):
            obs, _, _, next_obs, _ = self.replay_buffer.get_all_data(max_samples=self.batch_size)
            obs_u, _, _, _, _ = self.replay_buffer.get_all_data(max_samples=self.batch_size)
            obs_v, _, _, _, _ = self.replay_buffer.get_all_data(max_samples=self.batch_size)
            self.phi_state, pos_loss, neg_loss = self._update_phi(self.phi_state, obs, next_obs, obs_u, obs_v)
        return pos_loss, neg_loss

    @staticmethod
    @partial(jax.jit, static_argnames=["gamma", "gradient_steps", "min_priority", "batch_norm_momentum"])
    def _train_ok(ok_critic_state, ok_policy_state, s_obs, s_meta_actions, s_rewards, s_next_obs, s_dones, weight, M_ok, gamma, gradient_steps, min_priority, batch_norm_momentum, key):
        batch_size = s_obs.shape[0] // gradient_steps
        carry = {"ok_critic_state": ok_critic_state, "key": key, "loss_value": jnp.array(0.0)}

        def one_update(i, carry):
            ok_critic_state, key = carry["ok_critic_state"], carry["key"]
            s_obs2 = jax.lax.dynamic_slice_in_dim(s_obs, i * batch_size, batch_size)
            s_meta_actions2 = jax.lax.dynamic_slice_in_dim(s_meta_actions, i * batch_size, batch_size)
            s_rewards2 = jax.lax.dynamic_slice_in_dim(s_rewards, i * batch_size, batch_size)
            s_next_obs2 = jax.lax.dynamic_slice_in_dim(s_next_obs, i * batch_size, batch_size)
            s_dones2 = jax.lax.dynamic_slice_in_dim(s_dones, i * batch_size, batch_size)

            s_obs2 = jnp.tile(s_obs2, (2,) + (1,) * (s_obs2.ndim - 1))
            s_next_obs2 = jnp.tile(s_next_obs2, (2,) + (1,) * (s_obs2.ndim - 1))
            s_meta_actions2 = jnp.tile(s_meta_actions2, (2, 1))
            s_rewards2 = jnp.tile(s_rewards2, (2, 1))
            s_dones2 = jnp.tile(s_dones2, (2, 1))

            w_tile = jnp.tile(weight, (s_obs2.shape[0]//2, 1))
            sample_w = jax.random.choice(key, M_ok, (s_obs2.shape[0]//2,))
            # Add noise to sampled weight vectors
            key, noise_key = jax.random.split(key, 2)
            d = w_tile.shape[1]
            sample_w = jax.random.dirichlet(noise_key, 0.1 + d * sample_w, shape=(s_obs2.shape[0]//2,))

            w = jnp.vstack([w_tile, sample_w])

            ok_critic_state, loss_value, td_error, key = OKB.update_ok_critic(ok_policy_state, ok_critic_state, w, s_obs2, s_meta_actions2, s_rewards2, s_next_obs2, s_dones2, gamma, min_priority, batch_norm_momentum != 0, key)

            return {
                "ok_critic_state": ok_critic_state,
                "loss_value": loss_value,
                "key": key,
            }

        # update critic
        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        # update policy
        key = update_carry["key"]
        s_obs2 = jax.lax.dynamic_slice_in_dim(s_obs, 0, batch_size)
        s_obs2 = jnp.tile(s_obs2, (2,) + (1,) * (s_obs2.ndim - 1))
        w_tile = jnp.tile(weight, (s_obs2.shape[0]//2, 1))
        sample_w = jax.random.choice(key, M_ok, (s_obs2.shape[0]//2,))
        key, noise_key = jax.random.split(key, 2)
        d = w_tile.shape[1]
        sample_w = jax.random.dirichlet(noise_key, 0.1 + d * sample_w, shape=(s_obs2.shape[0]//2,))
        w = jnp.vstack([w_tile, sample_w])
        ok_policy_state, actor_loss, key = OKB.update_ok_policy(ok_policy_state, update_carry["ok_critic_state"], w, s_obs2, key)

        return (
            update_carry["ok_critic_state"],
            ok_policy_state,
            update_carry["loss_value"],
            actor_loss,
            key,
        )

    def train_ok(self, weight):
        if len(self.ok_replay_buffer) < self.learning_starts:
            return
        
        s_obs, s_actions, s_meta_actions, s_rewards, s_next_obs, s_dones = self.ok_replay_buffer.sample(self.batch_size * self.ok_gradient_updates)
        if self.normalize_obs:
            s_obs = self.obs_normalizer.normalize(s_obs)
            s_next_obs = self.obs_normalizer.normalize(s_next_obs)

        self.ok_critic_state, self.ok_policy_state, loss_value, actor_loss, self.key = OKB._train_ok(
            self.ok_critic_state,
            self.ok_policy_state,
            s_obs, s_meta_actions, s_rewards, s_next_obs, s_dones,
            weight,
            jnp.vstack(self.M_ok),
            self.gamma,
            self.ok_gradient_updates,
            self.min_priority,
            self.batch_norm_momentum,
            self.key,
        )

        if self.log and self.num_timesteps % 100 == 0:
            self.writer.add_scalar("ok/ok_critic_loss", loss_value.item(), self.num_timesteps)
            self.writer.add_scalar("ok/ok_actor_loss", actor_loss.item(), self.num_timesteps)

    @staticmethod
    @partial(jax.jit, static_argnames=["gamma", "gradient_steps", "min_priority", "batch_norm_momentum"])
    def _train_ok_single(ok_single_critic_state, ok_single_policy_state, s_obs, s_meta_actions, s_rewards, s_next_obs, s_dones, M_ok, gamma, gradient_steps, min_priority, batch_norm_momentum, key):
        batch_size = s_obs.shape[0] // gradient_steps
        carry = {"ok_critic_state": ok_single_critic_state, "key": key, "loss_value": jnp.array(0.0)}

        def one_update(i, carry):
            ok_critic_state, key = carry["ok_critic_state"], carry["key"]
            s_obs2 = jax.lax.dynamic_slice_in_dim(s_obs, i * batch_size, batch_size)
            s_meta_actions2 = jax.lax.dynamic_slice_in_dim(s_meta_actions, i * batch_size, batch_size)
            s_rewards2 = jax.lax.dynamic_slice_in_dim(s_rewards, i * batch_size, batch_size)
            s_next_obs2 = jax.lax.dynamic_slice_in_dim(s_next_obs, i * batch_size, batch_size)
            s_dones2 = jax.lax.dynamic_slice_in_dim(s_dones, i * batch_size, batch_size)

            ok_critic_state, loss_value, td_error, key = OKB.update_ok_single_critic(ok_single_policy_state, ok_critic_state, s_obs2, s_meta_actions2, s_rewards2, s_next_obs2, s_dones2, gamma, min_priority, batch_norm_momentum != 0, key)

            return {
                "ok_critic_state": ok_critic_state,
                "loss_value": loss_value,
                "key": key,
            }

        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        s_obs2 = jax.lax.dynamic_slice_in_dim(s_obs, 0, batch_size)
        ok_single_policy_state, actor_loss, key = OKB.update_ok_single_policy(ok_single_policy_state, update_carry['ok_critic_state'], s_obs2, update_carry['key'])

        return (
            update_carry["ok_critic_state"],
            ok_single_policy_state,
            update_carry["loss_value"],
            actor_loss,
            key,
        )

    def train_ok_single(self):
        if len(self.ok_single_replay_buffer) < self.learning_starts:
            return
        
        s_obs, s_actions, s_meta_actions, s_rewards, s_next_obs, s_dones = self.ok_single_replay_buffer.sample(self.batch_size * self.ok_gradient_updates)
        if self.normalize_obs:
            s_obs = self.obs_normalizer.normalize(s_obs)
            s_next_obs = self.obs_normalizer.normalize(s_next_obs)

        self.ok_single_critic_state, self.ok_single_policy_state, loss_value, actor_loss, self.key = OKB._train_ok_single(
            self.ok_single_critic_state,
            self.ok_single_policy_state,
            s_obs, s_meta_actions, s_rewards, s_next_obs, s_dones,
            jnp.vstack(self.M_ok),
            self.gamma,
            self.ok_gradient_updates,
            self.min_priority,
            self.batch_norm_momentum,
            self.key,
        )

        if self.log and self.num_timesteps % 100 == 0:
            self.writer.add_scalar("ok/ok_critic_loss", loss_value.item(), self.num_timesteps)
            self.writer.add_scalar("ok/ok_actor_loss", actor_loss.item(), self.num_timesteps)

    def train_ok_single_discrete(self):
        if len(self.ok_single_replay_buffer) < self.learning_starts:
                return

        critic_losses = []
        for g in range(self.ok_gradient_updates):
            s_obs, s_actions, s_meta_actions, s_rewards, s_next_obs, s_dones = self.ok_single_replay_buffer.sample(self.batch_size)
            
            if self.normalize_obs:
                s_obs = self.obs_normalizer.normalize(s_obs)
                s_next_obs = self.obs_normalizer.normalize(s_next_obs)

            self.q_state, loss, td_error, self.key = OKB.update_single_ok_discrete(self.q_net, self.q_state, s_obs, s_meta_actions, s_rewards, s_next_obs, s_dones, self.gamma, self.min_priority, self.key)
            critic_losses.append(loss.item())

        if self.num_timesteps % self.target_net_update_freq == 0:
            self.q_state = OKB.target_net_update(self.q_state)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_epsilon(self.initial_epsilon, self.epsilon_decay_steps, self.num_timesteps, self.offset_epsilon, self.final_epsilon)

        if self.log and self.num_timesteps % 100 == 0:
            self.writer.add_scalar("losses/critic_loss", np.mean(critic_losses), self.num_timesteps)
            self.writer.add_scalar("metrics/epsilon", self.epsilon, self.num_timesteps)

    @staticmethod
    @partial(jax.jit, static_argnames=["q", "gamma", "min_priority"])
    def update_single_ok_discrete(q, q_state, obs, actions, rewards, next_obs, dones, gamma, min_priority, key):
        key, inds_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)

        # DroQ update
        if q.n_critics >= 2:
            q_values_next = q.apply(q_state.target_params, next_obs, deterministic=False, rngs={"dropout": dropout_key_target})
            if q_values_next.shape[0] > 2:
                inds = jax.random.randint(inds_key, (2,), 0, q_values_next.shape[0])
                q_values_next = q_values_next[inds]
            min_q = q_values_next.min(axis=0)
            max_q = min_q.max(axis=1)
            target = max_q.reshape(-1, 1)

            def mse_loss(params, droptout_key):
                q_values = q.apply(params, obs, deterministic=False, rngs={"dropout": droptout_key})
                q_values = q_values[:, jnp.arange(q_values.shape[1]), actions.squeeze()]
                tds = q_values - target_q.reshape(1, -1)
                loss = jnp.abs(tds)
                loss = jnp.where(loss < min_priority, 0.5 * loss ** 2, loss * min_priority)
                return loss.mean(), tds
        # DDQN update
        else:
            q_values_next = q.apply(q_state.target_params, next_obs, deterministic=True)[0]
            q_values_not_target = q.apply(q_state.params, next_obs, deterministic=True)[0]
            max_acts = q_values_not_target.argmax(axis=1)
            target = q_values_next[jnp.arange(q_values_next.shape[0]), max_acts].reshape(-1, 1)

            def mse_loss(params, droptout_key):
                q_values = q.apply(params, obs,  deterministic=True)
                q_values = q_values[:, jnp.arange(q_values.shape[1]), actions.squeeze()]
                tds = q_values - target_q.reshape(1, -1)
                loss = jnp.abs(tds)
                loss = jnp.where(loss < min_priority, 0.5 * loss ** 2, loss * min_priority)
                return loss.mean(), tds

        target_q = rewards + (1 - dones) * gamma * target

        (loss_value, td_error), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params, dropout_key_current)
        q_state = q_state.apply_gradients(grads=grads)

        return q_state, loss_value, td_error, key

    def train(self, weight):
        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self.sample_batch_experiences()
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self.sample_batch_experiences()
            
            if self.normalize_obs:
                s_obs = self.obs_normalizer.normalize(s_obs)
                s_next_obs = self.obs_normalizer.normalize(s_next_obs)

            if len(self.M) > 1:
                s_obs2, s_actions2, s_rewards2, s_next_obs2, s_dones2 = np.vstack([s_obs]*2), np.vstack([s_actions]*2), np.vstack([s_rewards]*2), np.vstack([s_next_obs]*2), np.vstack([s_dones]*2)
                w = np.vstack([weight for _ in range(s_obs2.shape[0] // 2)] + random.choices(self.M, k=s_obs2.shape[0] // 2))
            else:
                s_obs2, s_actions2, s_rewards2, s_next_obs2, s_dones2 = s_obs, s_actions, s_rewards, s_next_obs, s_dones
                w = weight.reshape(1, -1).repeat(s_obs.shape[0], axis=0)

            self.psi_state, loss, td_error, self.key = OKB.update(self.psi, self.psi_state, w, s_obs2, s_actions2, s_rewards2, s_next_obs2, s_dones2, self.gamma, self.min_priority, self.key)
            critic_losses.append(loss.item())

            if self.per:
                td_error = jax.device_get(td_error)
                td_error = np.abs((td_error[:,: len(idxes)] * w[: len(idxes)]).sum(axis=2))
                per = np.max(td_error, axis=0)
                priority = per.clip(min=self.min_priority, max=100.0)**self.alpha
                self.replay_buffer.update_priorities(idxes, priority)

        if self.tau != 1 or self.num_timesteps % self.target_net_update_freq == 0:
            self.psi_state = OKB.target_net_update(self.psi_state)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_epsilon(self.initial_epsilon, self.epsilon_decay_steps, self.num_timesteps, self.offset_epsilon, self.final_epsilon)

        if self.log and self.num_timesteps % 100 == 0:
            if self.per:
                self.writer.add_scalar("metrics/mean_priority", np.mean(priority), self.num_timesteps)
                self.writer.add_scalar("metrics/max_priority", np.max(priority), self.num_timesteps)
                self.writer.add_scalar("metrics/mean_td_error_w", np.mean(per), self.num_timesteps)
            self.writer.add_scalar("losses/critic_loss", np.mean(critic_losses), self.num_timesteps)
            self.writer.add_scalar("metrics/epsilon", self.epsilon, self.num_timesteps)

    @staticmethod
    @jax.jit
    def target_net_update(psi_state):
        psi_state = psi_state.replace(target_params=optax.incremental_update(psi_state.params, psi_state.target_params, 1))
        return psi_state

    @staticmethod
    @partial(jax.jit, static_argnames=["psi"])
    def ok_gpi_action(ok_policy_state, ok_critic_state, psi, psi_state, obs, w, M_ok, M, key):
        M_ok = jnp.stack(M_ok)
        obs_m = obs.reshape(1, *obs.shape).repeat(M_ok.shape[0], axis=0)
        w_oks = ok_policy_state.apply_fn({"params": ok_policy_state.params, "batch_stats": ok_policy_state.batch_stats}, obs_m, M_ok, train=False)
        psis = ok_critic_state.apply_fn({"params": ok_critic_state.params, "batch_stats": ok_critic_state.batch_stats}, obs_m, w_oks, M_ok, deterministic=True, train=False)
        psis = psis.mean(axis=0)
        values = (psis * w).sum(axis=1)
        max_action_ind = values.argmax()
        best_w_ok = w_oks[max_action_ind]

        action, policy_ind, key = OKB.gpi_action(psi, psi_state, obs, best_w_ok, M, key, return_policy_index=True)
        return action, best_w_ok, policy_ind, key
    
    @staticmethod
    @partial(jax.jit, static_argnames=["psi", "noise", "add_noise"])
    def ok_action(ok_policy_state, psi, psi_state, obs, w, M, key, noise, add_noise=False):
        w_ok = ok_policy_state.apply_fn({"params": ok_policy_state.params, "batch_stats": ok_policy_state.batch_stats}, obs, w, train=False)
        w_ok = w_ok.reshape(w_ok.shape[-1])

        if add_noise:
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, w_ok.shape, dtype=jnp.float32) * noise
            noise = jnp.clip(noise, -0.5, 0.5)
            w_ok = w_ok + noise
            w_ok = unit_norm(w_ok)

        action, policy_ind, key = OKB.gpi_action(psi, psi_state, obs, w_ok, M, key, return_policy_index=True)
        return action, w_ok, policy_ind, key

    @staticmethod
    @partial(jax.jit, static_argnames=["psi", "return_policy_index"])
    def gpi_action(psi, psi_state, obs, w, M, key, return_policy_index=False):
        M = jnp.stack(M)
        
        obs_m = obs.reshape(1,*obs.shape).repeat(M.shape[0], axis=0)
        psi_values = psi.apply(psi_state.params, obs_m, M, deterministic=True)
        q_values = (psi_values * w.reshape(1, 1, 1, w.shape[0])).sum(axis=3)
        
        q_values = q_values.mean(axis=0)

        max_q = q_values.max(axis=1)
        policy_index = max_q.argmax()  # max_i max_a q(s,a,w_i)
        action = q_values[policy_index].argmax()

        if return_policy_index:
            return action, policy_index, key
        return action, key

    def eval(self, obs: np.ndarray, w: np.ndarray = None) -> int:
        if self.normalize_obs:
            obs = self.obs_normalizer.normalize(obs)

        if self.include_w:
            self.M.append(w)

        if self.use_single_ok:
            action, w_ok, pi_ind, self.key = OKB.ok_single_action(self.ok_single_policy_state, self.psi, self.psi_state, obs, self.M, self.key, 0.0, add_noise=False)
            self.policy_inds.append(pi_ind)
            self.meta_actions.append(w_ok)
            self.action_hist.append(action)
        elif self.use_single_ok_discrete:
            meta_action = OKB.ok_single_discrete_action(self.q_net, self.q_state, obs)
            ws = [np.array([-np.sqrt(0.5), np.sqrt(0.5)]), np.array([0.0, 1.0]), np.array([np.sqrt(0.5), np.sqrt(0.5)]), np.array([1.0, 0.0]), np.array([np.sqrt(0.5), -np.sqrt(0.5)])]
            w_ok = ws[meta_action] # self.M[meta_action]
            action, pi_ind, self.key = OKB.gpi_action(self.psi, self.psi_state, obs, w_ok, self.M, self.key, return_policy_index=True)
            self.policy_inds.append(pi_ind)
            self.meta_actions.append(w_ok)
            self.action_hist.append(action)
        elif self.use_ok:
            action, w_ok, pi_ind, self.key = OKB.ok_action(self.ok_policy_state, self.psi, self.psi_state, obs, w, self.M, self.key, 0.0, add_noise=False)
            self.policy_inds.append(pi_ind)
            self.meta_actions.append(w_ok)
            self.action_hist.append(action)
        elif self.use_gpi:
            action, policy_index, self.key = OKB.gpi_action(self.psi, self.psi_state, obs, w, self.M, self.key, return_policy_index=True)
            self.policy_inds.append(policy_index)
        else:
            action, self.key = OKB.max_action(self.psi, self.psi_state, obs, w, self.key)

        if self.include_w:
            self.M.pop(-1)

        action = jax.device_get(action)            
        return action

    @staticmethod
    @partial(jax.jit, static_argnames=["psi", "noise", "add_noise"])
    def ok_single_action(ok_single_policy_state, psi, psi_state, obs, M, key, noise, add_noise=False):
        w_ok = ok_single_policy_state.apply_fn({"params": ok_single_policy_state.params, "batch_stats": ok_single_policy_state.batch_stats}, obs, train=False)
        w_ok = w_ok.reshape(w_ok.shape[-1])

        if add_noise:
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, w_ok.shape, dtype=jnp.float32) * noise
            noise = jnp.clip(noise, -0.5, 0.5)
            w_ok = w_ok + noise
            w_ok = unit_norm(w_ok)

        action, policy_ind, key = OKB.gpi_action(psi, psi_state, obs, w_ok, M, key, return_policy_index=True)
        return action, w_ok, policy_ind, key
    
    @staticmethod
    @partial(jax.jit, static_argnames=["q_net"])
    def ok_single_discrete_action(q_net, q_state, obs):
        q_values = q_net.apply(q_state.params, obs, deterministic=True)
        q_values = q_values.mean(axis=0).squeeze(0)
        meta_action = q_values.argmax()
        return meta_action

    def act(self, obs, w=None) -> int:
        if self.normalize_obs:
            obs = self.obs_normalizer.normalize(obs)

        if np.random.random() < self.epsilon and not self.use_ok:
            return self.env.action_space.sample(), True
        else:            
            if self.use_single_ok:
                action, w_ok, pi_ind, self.key = OKB.ok_single_action(self.ok_single_policy_state, self.psi, self.psi_state, obs, self.M, self.key, self.ok_policy_noise, add_noise=True)
                return action, w_ok
            
            if self.use_single_ok_discrete:
                if np.random.random() < self.epsilon:
                    meta_action = np.random.randint(0, 5)
                else:
                    meta_action = OKB.ok_single_discrete_action(self.q_net, self.q_state, obs)
                ws = [np.array([-np.sqrt(0.5), np.sqrt(0.5)]), np.array([0.0, 1.0]), np.array([np.sqrt(0.5), np.sqrt(0.5)]), np.array([1.0, 0.0]), np.array([np.sqrt(0.5), -np.sqrt(0.5)])]
                w_ok = ws[meta_action]
                action, pi_ind, self.key = OKB.gpi_action(self.psi, self.psi_state, obs, w_ok, self.M, self.key, return_policy_index=True)
                return action, meta_action

            if self.use_ok:
                action, w_ok, pi_ind, self.key = OKB.ok_action(self.ok_policy_state, self.psi, self.psi_state, obs, w, self.M, self.key, self.ok_policy_noise, add_noise=True)
                action = jax.device_get(action)
                return action, w_ok
            
            if self.use_gpi:
                action, policy_index, self.key = OKB.gpi_action(self.psi, self.psi_state, obs, w, self.M, self.key, return_policy_index=True)
                action, policy_index = jax.device_get(action), jax.device_get(policy_index)
                self.policy_inds.append(policy_index)
            else:
                action, self.key = OKB.max_action(self.psi, self.psi_state, obs, w, self.key)
                action = jax.device_get(action)
            return action, False

    @staticmethod
    @partial(jax.jit, static_argnames=["psi"])
    def max_action(psi, psi_state, obs, w, key) -> int:
        psi_values = psi.apply(psi_state.params, obs, w, deterministic=True)
        q_values = (psi_values * w.reshape(1, w.shape[0])).sum(axis=3)
        q_values = q_values.mean(axis=0).squeeze(0)
        action = q_values.argmax()
        action = jax.device_get(action)
        return action, key
    
    @staticmethod
    @partial(jax.jit, static_argnames=["psi"])
    def get_psis(psi, psi_state, obs, M):
        M = jnp.stack(M)
        obs_m = obs.reshape(1, *obs.shape).repeat(M.shape[0], axis=0)
        psi_values = psi.apply(psi_state.params, obs_m, M, deterministic=True).mean(0)
        return psi_values
    
    def find_z_express_action(self, obs, action, w):
        actions_ok, test_zs = self.find_z(self.psi, self.psi_state, obs, w, self.M, self.key)
        idx = jnp.where(actions_ok == action)[0]
        if len(idx) > 0:
            idx = idx[0]
            return test_zs[idx]
        else:
            return None

    def construct_program_check_ok(self, psis, action_psi):
        z = cp.Variable(psis.shape[1])
        self.psis_param = cp.Parameter(psis.shape)
        self.action_psi_param = cp.Parameter(action_psi.shape)
        constraints = []
        for i in range(psis.shape[0]):
            constraints.append(self.psis_param[i] @ z + 1e-4 <= self.action_psi_param @ z)
        constraints.extend([cp.norm(z) <= 1])

        prob = cp.Problem(cp.Maximize(self.action_psi_param @ z), constraints)
        return prob

    def check_ok_can_express_action(self, obs, action):
        psis = OKB.get_psis(self.psi, self.psi_state, obs, self.M)
        psis = psis.reshape(psis.shape[0]*psis.shape[1], psis.shape[2])

        if self.ok_linear_program is None:
            self.ok_linear_program = self.construct_program_check_ok(psis[1:,:], psis[action].flatten())

        for i in range(len(self.M)):
            self.psis_param.value = np.delete(psis, (i * self.action_dim + action), axis=0)
            self.action_psi_param.value = np.array(psis[i * self.action_dim + action]).flatten()
            try:
                self.ok_linear_program.solve(solver=cp.SCIPY, warm_start=True)
            except:
                self.ok_linear_program.solve()

            if self.ok_linear_program.status not in ["infeasible", "unbounded"]:
                for variable in self.ok_linear_program.variables():
                    if np.linalg.norm(variable.value) < 0.9:
                        return 0.0
                return 1.0

        return 0.0
    
    def compute_ok_priorities(self, corner_weights):
        obs, actions, phis, next_obs, dones = self.replay_buffer.get_all_data(max_samples=50000)

        if self.normalize_obs:
            obs = self.obs_normalizer.normalize(obs)
            next_obs = self.obs_normalizer.normalize(next_obs)

        condition1 = -1 * np.ones(obs.shape[0])
    
        sum_per_w, mean_per_w, max_per_w, nonzero_per_w = [], [], [], []
        for w in corner_weights:
            gaps = np.zeros((obs.shape[0], 1), dtype=np.float64)

            window_len = 250
            num_b = int(np.ceil(obs.shape[0] / window_len))
            for b in range(num_b):
                start = b * window_len
                end = min((b + 1) * window_len, obs.shape[0])
                gaps[start:end] = np.asarray(self.vcompute_value_gap(self.ok_policy_state, self.ok_critic_state, self.psi, self.psi_state, obs[start:end], phis[start:end], next_obs[start:end], dones[start:end], w, self.M, self.gamma, self.key), np.float32)
            
            gaps = np.where(gaps <= 0.1, 0.0, gaps)
            num_positive_gaps = (gaps > 0.0).sum()

            gaps = gaps.reshape(-1)

            non_zero_inds = gaps.nonzero()[0]
            mean_gap = gaps[non_zero_inds].mean() if len(non_zero_inds) > 0 else 0.0
            max_gap = gaps.max()
            sum_gap = gaps.sum()
            print("w", w, "max gap", max_gap, "mean gap", mean_gap, "sum gap", sum_gap, "nonzero", len(non_zero_inds))
            sum_per_w.append(sum_gap)
            max_per_w.append(max_gap)
            mean_per_w.append(mean_gap)
            nonzero_per_w.append(len(non_zero_inds))

        print(f"Number of states s.t. gap is positive: {num_positive_gaps}/{condition1.shape[0]}")
        print(f"Number of states s.t. condition is True: {(condition1 > 0).sum()}/{condition1.shape[0]}")
        print("Next w for base policy priorities:", [(w, p) for w, p in zip(corner_weights, mean_per_w)])
        print("Largest sum, mean, max, nonzero gaps:", corner_weights[np.argmax(sum_per_w)], corner_weights[np.argmax(mean_per_w)], corner_weights[np.argmax(max_per_w)], corner_weights[np.argmax(nonzero_per_w)])

        candidates = [(p, w_c) for w_c, p in zip(corner_weights, mean_per_w)]
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates
    
    def update_ok_buffer(self):
        if len(self.ok_replay_buffer) == 0:
            return

        print("OK buffer size before:", len(self.ok_replay_buffer))

        buffer_size = len(self.ok_replay_buffer)
        self.ok_replay_buffer.clear()

        random_weights = jnp.concatenate([jnp.vstack(self.M), jnp.eye(self.phi_dim), jax.random.normal(self.key, (1000, self.phi_dim))], axis=0)
        random_weights /= jnp.maximum(jnp.linalg.norm(random_weights, axis=-1, keepdims=True), 1e-6)

        window_len = 100
        num_b = int(np.ceil(buffer_size / window_len))
        for b in range(num_b):
            start = b * window_len
            end = min((b + 1) * window_len, buffer_size)
            obs_b = self.ok_replay_buffer.obs[start:end]
            actions_b = self.ok_replay_buffer.actions[start:end]
            meta_actions_b = self.ok_replay_buffer.meta_actions[start:end]
            rewards_b = self.ok_replay_buffer.rewards[start:end]
            next_obs_b = self.ok_replay_buffer.next_obs[start:end]
            dones_b = self.ok_replay_buffer.dones[start:end]

            if self.normalize_obs:
                obs_b = self.obs_normalizer.normalize(obs_b)
                next_obs_b = self.obs_normalizer.normalize(next_obs_b)

            action_gpi, psis = self.vgpi(self.psi, self.psi_state, obs_b, meta_actions_b, self.M)
            for i in range(obs_b.shape[0]):
                if actions_b[i] == action_gpi[i]:
                    self.ok_replay_buffer.add_without_copy(obs_b[i], actions_b[i], meta_actions_b[i], rewards_b[i], next_obs_b[i], dones_b[i])
                else:
                    actions_ok = self.vgpipsi(psis[i], random_weights)
                    idx = np.where(actions_ok == actions_b[i])[0]
                    if len(idx) > 0:
                        idx = idx[0]
                        self.ok_replay_buffer.add_without_copy(obs_b[i], actions_b[i], random_weights[idx], rewards_b[i], next_obs_b[i], dones_b[i])

        print("OK buffer size after:", len(self.ok_replay_buffer))


    def learn_iteration(
        self,
        total_timesteps: int,
        w: np.ndarray,  
        M: List[np.ndarray],
        change_w_each_episode: bool = True,
        reset_num_timesteps: bool = True,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
        reset_exploration: bool = False,
    ):
        self.M = M.copy()

        self.policy_inds = []
        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_exploration:
            self.offset_epsilon = self.num_timesteps

        episode_vec_reward = np.zeros(self.env.unwrapped.reward_dim)
        num_episodes = 0
        (obs, info), done = self.env.reset(), False
        for _ in range(1, total_timesteps + 1):
            self.num_timesteps += 1

            if self.num_timesteps < self.learning_starts:
                action = self.env.action_space.sample()
                was_random = True
            else:
                action, was_random = self.act(obs, w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if self.normalize_obs:
                self.obs_normalizer.update(obs)

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)

            if self.ok_policy is not None:
                if not was_random:
                    z = unit_norm_np(w)
                    self.ok_replay_buffer.add(obs, action, z, vec_reward, next_obs, terminated)
                else:
                    z = self.find_z_express_action(obs, action, w)
                    if z is not None:
                        self.ok_replay_buffer.add(obs, action, z, vec_reward, next_obs, terminated)

            if self.num_timesteps >= self.learning_starts:
                self.train(w)


            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                _, _, total_vec_r, total_vec_return = eval_mo(self, eval_env, w)
                for i in range(episode_vec_reward.shape[0]):
                    self.writer.add_scalar(f"eval/total_reward_obj{i}", total_vec_r[i], self.num_timesteps)
                    self.writer.add_scalar(f"eval/return_obj{i}", total_vec_return[i], self.num_timesteps)

            episode_vec_reward += vec_reward
            if done:
                if self.normalize_obs:
                    self.obs_normalizer.update(next_obs)
                (obs, info), done = self.env.reset(), False
                num_episodes += 1
                self.num_episodes += 1

                if num_episodes % 100 == 0:
                    print(f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_vec_reward}")
                if self.log:
                    wb.log({"metrics/policy_index": np.array(self.policy_inds), "global_step": self.num_timesteps})
                    self.policy_inds = []
                    self.writer.add_scalar("metrics/episode", self.num_episodes, self.num_timesteps)
                    for i in range(episode_vec_reward.shape[0]):
                        self.writer.add_scalar(f"metrics/episode_reward_obj{i}", episode_vec_reward[i], self.num_timesteps)

                episode_vec_reward = np.zeros(self.env.unwrapped.reward_dim)

                if change_w_each_episode:
                    if self.top_k_base > 1:
                        w = random.choice(M)
                    else:
                        if random.random() < 0.5 or len(M) == 1:
                            w = M[0]
                        else:
                            w = random.choice(M[1:])
            else:
                obs = next_obs

    def learn_ok_iteration(
        self,
        total_timesteps: int,
        w: np.ndarray,
        M_ok: List[np.ndarray],
        change_w_each_episode: bool = True,
        reset_num_timesteps: bool = True,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
    ):
        self.M_ok = M_ok.copy()

        self.policy_inds = []
        self.meta_actions = []
        self.action_hist = []
        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        episode_vec_reward = np.zeros(self.env.unwrapped.reward_dim)
        num_episodes = 0
        (obs, info), done = self.env.reset(), False
        for _ in range(1, total_timesteps + 1):
            self.num_timesteps += 1

            action, w_ok = self.act(obs, w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if self.normalize_obs:
                self.obs_normalizer.update(obs)

            self.ok_replay_buffer.add(obs, action, w_ok, vec_reward, next_obs, terminated)

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)

            self.train_ok(w)

            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                _, _, total_vec_r, total_vec_return = eval_mo(self, eval_env, w)
                for i in range(episode_vec_reward.shape[0]):
                    self.writer.add_scalar(f"eval/total_reward_obj{i}", total_vec_r[i], self.num_timesteps)
                    self.writer.add_scalar(f"eval/return_obj{i}", total_vec_return[i], self.num_timesteps)

            episode_vec_reward += vec_reward
            if done:
                if self.normalize_obs:
                    self.obs_normalizer.update(next_obs)
                (obs, info), done = self.env.reset(), False
                num_episodes += 1
                self.num_episodes += 1

                if num_episodes % 100 == 0:
                    print(f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_vec_reward}")
                if self.log:
                    wb.log({"metrics/policy_index": np.array(self.policy_inds), "global_step": self.num_timesteps})
                    self.policy_inds = []
                    self.writer.add_scalar("metrics/episode", self.num_episodes, self.num_timesteps)
                    for i in range(episode_vec_reward.shape[0]):
                        self.writer.add_scalar(f"metrics/episode_reward_obj{i}", episode_vec_reward[i], self.num_timesteps)

                episode_vec_reward = np.zeros(self.env.unwrapped.reward_dim)

                if change_w_each_episode:
                    w = random.choice(M_ok)
            else:
                obs = next_obs

    def initial_weight_vector(self):
        if self.initial_ok_task == "one-hot":
            w = np.zeros(self.phi_dim, dtype=np.float32)
            w[0] = 1.0
        else:
            w = np.ones(self.phi_dim, dtype=np.float32) * 1 / self.phi_dim
        return w

    def next_w_ok_base_policy(self, iteration, corner_weights_ok):
        if self.weight_selection == 'okb':
            if iteration == 1:
                w = self.initial_weight_vector()
                M = unique_tol([w] + extrema_weights(self.phi_dim)[:self.top_k_base - 1])
                w = M[0]

            else:
                w_prime = self.ols.next_w()
                M_prime = [w_prime] + self.ols.get_corner_weights(top_k=32)
                M_prime = filter_from_list(M_prime, self.ols.W)
                corner_weights_base = unique_tol(M_prime)

                if len(corner_weights_base) == 0:
                    print("No new base policy needed.")
                    M = unique_tol(self.ols.get_ccs_weights())
                    w = M[0]
                    return w, M
                
                candidates = self.compute_ok_priorities(corner_weights_base)
                w = candidates[0][1]
                gap = candidates[0][0]
                M = unique_tol([w_ for (p_, w_) in candidates[:self.top_k_base]] + self.ols.get_ccs_weights())

                if gap == 0.0:
                    corner_weights_ok = unique_tol(corner_weights_ok)
                    corner_weights_ok = filter_from_list(corner_weights_ok, self.ols.W)
                    if len(corner_weights_ok) > 0:
                        candidates = self.compute_ok_priorities(corner_weights_ok)
                        w = candidates[0][1]
                        gap = candidates[0][0]
                        M = unique_tol([w_ for (p_, w_) in candidates[:self.top_k_base]] + self.ols.get_ccs_weights())
                        if gap == 0.0:
                            print("No new base policy needed.")
                            M = unique_tol(self.ols.get_ccs_weights())
                            w = M[0]
                            return w, M

        elif self.weight_selection == "random":
            random_ws = list(random_weights(dim=self.phi_dim, n=self.top_k_base + 1))[:self.top_k_base]
            M = unique_tol(random_ws + self.ols.get_ccs_weights())
            w = M[0]
        
        elif self.weight_selection == "sip":
            sip_weights = []
            for i in range(self.phi_dim):
                w = -1.0 * np.ones(self.phi_dim, dtype=np.float32)
                w[i] = 1.0
                sip_weights.append(w)
            w = sip_weights[int(iteration % len(sip_weights))]
            M = unique_tol([w] + self.ols.get_ccs_weights())

        else:
            raise NotImplementedError

        print("Next w base policies:", w)

        return w, M

    def learn(self, eval_env, timesteps_per_iteration, num_iterations, test_tasks, rep_eval=5, save_dir=None):
        self._ok_policy_updates = 0
        eval_env = LinearReward(eval_env)
        sample_k_corner_weights = 256 if self.phi_dim >= 4 else None

        self.ols = OLS(m=self.phi_dim, epsilon=None)
        self.ols_ok = OLS(m=self.phi_dim, epsilon=None, sample_k=sample_k_corner_weights)

        self.M_ok = self.ols_ok.get_corner_weights()
        corner_weights_ok = self.M_ok.copy()

        for iteration in range(1, num_iterations + 1):
            if self.ok_policy is not None and self.weight_selection != "ols":
                w, M = self.next_w_ok_base_policy(iteration, corner_weights_ok)
            else:  # SFOLS
                w = self.ols.next_w()
                M = unique_tol([w] + self.ols.get_ccs_weights() + self.ols.get_corner_weights(top_k=self.top_k_base - 1))

            self.use_gpi, self.use_ok = True, False
            self.learn_iteration(
                total_timesteps=timesteps_per_iteration,
                w=w,
                M=M,
                change_w_each_episode=True,
                eval_env=eval_env,
                eval_freq=1000,
                reset_num_timesteps=False,
                reset_exploration=True,
            )
            for w_cw in M:
                n_value = policy_evaluation_mo(self, eval_env, w_cw, rep=rep_eval)
                self.ols.add_solution(n_value, w_cw, add_not_improved=False)
            self.M = self.ols.get_ccs_weights()

            if self.log:
                wb.log({
                    "eval/Number of Base Weights": len(self.ols.get_ccs_weights()),
                    "global_step": self.num_timesteps,
                    "iteration": iteration,
                })

            ## OK Loop
            if self.ok_policy is not None:
                print("OK LOOP:")
                self.use_gpi, self.use_ok = True, True
                
                self.update_ok_buffer()

                W_prev_iteration = self.ols_ok.W.copy()
                self.ols_ok = OLS(m=self.phi_dim, epsilon=None, sample_k=sample_k_corner_weights)  # Reset OLS for OK
                for w in W_prev_iteration:  # Re-eval previous trained w's
                    n_value = policy_evaluation_mo(self, eval_env, w, rep=rep_eval)
                    self.ols_ok.add_solution(n_value, w, add_not_improved=False)

                for ok_iteration in range(self.num_ok_iterations):
                    w = self.ols_ok.next_w()
                    if w is None:
                        raise NotImplementedError
                    
                    M_ok = unique_tol([w] + self.ols_ok.get_ccs_weights() + self.ols_ok.get_corner_weights(top_k=self.top_k_ok))
                    self.learn_ok_iteration(
                        total_timesteps=timesteps_per_iteration // self.num_ok_iterations,
                        w=w,
                        M_ok=M_ok,
                        change_w_each_episode=True,
                        eval_env=eval_env,
                        eval_freq=1000,
                        reset_num_timesteps=False,
                    )
                    for w_cw in M_ok:
                        n_value = policy_evaluation_mo(self, eval_env, w_cw, rep=rep_eval)
                        self.ols_ok.add_solution(n_value, w_cw, add_not_improved=False)
                    corner_weights_ok = M_ok.copy()

                # Evaluation
                if self.log:
                    wb.log({
                        "eval/Number of OK Weights": len(self.ols_ok.get_ccs_weights()),
                        "global_step": self.num_timesteps,
                        "iteration": iteration,
                    })

                if self.log:
                    self.use_gpi, self.use_ok = True, True
                    ok_ccs = [policy_evaluation_mo(self, eval_env, wt, rep=rep_eval, return_undiscounted=True) for wt in test_tasks]
                    ok_undisc_ccs = [v[1] for v in ok_ccs]
                    ok_ccs = [v[0] for v in ok_ccs]
                    log_all_multi_policy_metrics(current_front=ok_ccs,
                                                reward_dim=eval_env.unwrapped.reward_dim,
                                                global_step=self.num_timesteps,
                                                iteration=iteration,
                                                test_tasks=test_tasks,
                                                id="OK CCS",
                                                )
                    log_all_multi_policy_metrics(current_front=ok_undisc_ccs,
                                                reward_dim=eval_env.unwrapped.reward_dim,
                                                global_step=self.num_timesteps,
                                                iteration=iteration,
                                                test_tasks=test_tasks,
                                                id="OK undiscounted CCS",
                                                )

            if self.log:
                self.use_gpi, self.use_ok = True, False
                gpi_ccs = [policy_evaluation_mo(self, eval_env, wt, rep=rep_eval, return_undiscounted=True) for wt in test_tasks]
                gpi_undisc_ccs = [v[1] for v in gpi_ccs]
                gpi_ccs = [v[0] for v in gpi_ccs]
                log_all_multi_policy_metrics(current_front=gpi_ccs,
                                             reward_dim=eval_env.unwrapped.reward_dim,
                                             global_step=self.num_timesteps,
                                             iteration=iteration,
                                             test_tasks=test_tasks,
                                             id="GPI CCS",
                                            )
                log_all_multi_policy_metrics(current_front=gpi_undisc_ccs,
                                             reward_dim=eval_env.unwrapped.reward_dim,
                                             global_step=self.num_timesteps,
                                             iteration=iteration,
                                             test_tasks=test_tasks,
                                             id="GPI undiscounted CCS",
                                            )

                self.use_gpi, self.use_ok = False, False
                nogpi_ccs = [policy_evaluation_mo(self, eval_env, wt, rep=rep_eval, return_undiscounted=True) for wt in test_tasks]
                nogpi_undisc_ccs = [v[1] for v in nogpi_ccs]
                nogpi_ccs = [v[0] for v in nogpi_ccs]
                log_all_multi_policy_metrics(current_front=nogpi_ccs,
                                             reward_dim=eval_env.unwrapped.reward_dim,
                                             global_step=self.num_timesteps,
                                             iteration=iteration,
                                             test_tasks=test_tasks,
                                             id="No GPI CCS",
                                            )
                log_all_multi_policy_metrics(current_front=nogpi_undisc_ccs,
                                             reward_dim=eval_env.unwrapped.reward_dim,
                                             global_step=self.num_timesteps,
                                             iteration=iteration,
                                             test_tasks=test_tasks,
                                             id="No GPI undiscounted CCS",
                                            )
            self.use_gpi, self.use_ok = True, True

            if save_dir is not None:
                timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                self.save(save_dir=save_dir, filename=f"{self.weight_selection}-it{iteration}-{self.env.unwrapped.spec.id}-{timestamp}")

    def learn_single_ok(self, env, eval_env, total_timesteps, reset_num_timesteps=True, eval_freq=1000, rep_eval=15, save_dir=None):
        self.setup_single_ok()

        self.policy_inds = []
        self.meta_actions = []
        self.action_hist = []
        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        self.use_single_ok = True

        episode_reward = 0.0
        num_episodes = 0
        (obs, info), done = env.reset(), False
        for _ in range(1, total_timesteps + 1):
            self.num_timesteps += 1

            action, w_ok = self.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if self.normalize_obs:
                self.obs_normalizer.update(obs)

            self.ok_single_replay_buffer.add(obs, action, w_ok, reward, next_obs, terminated)

            self.train_ok_single()

            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                total_r, total_return = policy_evaluation(self, eval_env, rep=rep_eval)
                self.writer.add_scalar("eval/total_reward", total_r, self.num_timesteps)
                self.writer.add_scalar("eval/discounted_return", total_return, self.num_timesteps)

            episode_reward += reward
            if done:
                if self.normalize_obs:
                    self.obs_normalizer.update(next_obs)

                (obs, info), done = env.reset(), False
                num_episodes += 1
                self.num_episodes += 1

                if num_episodes % 100 == 0:
                    print(f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_reward}")
                if self.log:
                    wb.log({"metrics/policy_index": np.array(self.policy_inds), "global_step": self.num_timesteps})
                    self.policy_inds = []
                    self.writer.add_scalar("metrics/episode", self.num_episodes, self.num_timesteps)
                    self.writer.add_scalar("metric/episode_reaward", episode_reward, self.num_timesteps)

                episode_reward = 0.0

            else:
                obs = next_obs

        if save_dir is not None:
            timestamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            self.save_single_ok(save_dir=save_dir, filename=f"single-ok-{self.weight_selection}-{self.env.unwrapped.spec.id}-{timestamp}")

    def learn_single_ok_discrete(self, env, eval_env, total_timesteps, reset_num_timesteps=True, eval_freq=1000, rep_eval=15):
        self.setup_single_ok_discrete()

        self.policy_inds = []
        self.meta_actions = []
        self.action_hist = []
        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        self.use_single_ok_discrete = True

        episode_reward = 0.0
        num_episodes = 0
        (obs, info), done = env.reset(), False
        for _ in range(1, total_timesteps + 1):
            self.num_timesteps += 1

            action, w_ok = self.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if self.normalize_obs:
                self.obs_normalizer.update(obs)

            self.ok_single_replay_buffer.add(obs, action, w_ok, reward, next_obs, terminated)

            self.train_ok_single_discrete()

            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                total_r, total_return = policy_evaluation(self, eval_env, rep=rep_eval)
                self.writer.add_scalar("eval/total_reward", total_r, self.num_timesteps)
                self.writer.add_scalar("eval/discounted_return", total_return, self.num_timesteps)

            episode_reward += reward
            if done:
                if self.normalize_obs:
                    self.obs_normalizer.update(next_obs)

                (obs, info), done = env.reset(), False
                num_episodes += 1
                self.num_episodes += 1

                if num_episodes % 100 == 0:
                    print(f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_reward}")
                if self.log:
                    wb.log({"metrics/policy_index": np.array(self.policy_inds), "global_step": self.num_timesteps})
                    self.policy_inds = []
                    self.writer.add_scalar("metrics/episode", self.num_episodes, self.num_timesteps)
                    self.writer.add_scalar("metric/episode_reaward", episode_reward, self.num_timesteps)

                episode_reward = 0.0

            else:
                obs = next_obs
