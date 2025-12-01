import copy
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional 

from .policies import CnnPolicy, MlpPolicy, MultiInputPolicy

SelfMDPO = TypeVar("SelfMDPO", bound="MDPO")


class MDPO(OnPolicyAlgorithm):
    """
    Mirror Descent Policy Optimization (MDPO)

    This implementation uses Mirror Descent with multi-step SGD updates
    and supports Tsallis entropy regularization.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for policy updates (outer learning rate)
    :param value_learning_rate: The learning rate for the value function
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size for both policy and value function
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param n_policy_updates: Number of SGD steps per policy update (sgd_steps in original MDPO)
    :param n_critic_updates: Number of critic updates per iteration
    :param inner_learning_rate: Inner loop learning rate for Mirror Descent (lr_now in original)
    :param kl_coeff: Coefficient for KL divergence regularization term
    :param clip_range_vf: Clipping range for value function (similar to PPO)
    :param tsallis_q: Tsallis entropy parameter (1.0 = standard KL divergence)
    :param method: Update method ('multistep-SGD', 'closedreverse-KL', 'closed-KL')
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param rollout_buffer_class: Rollout buffer class to use
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer
    :param normalize_advantage: Whether to normalize advantages
    :param stats_window_size: Window size for the rollout logging
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        value_learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.98,
        n_policy_updates: int = 5,
        n_critic_updates: int = 3,
        inner_learning_rate: Union[float, Schedule] = 1.0,
        kl_coeff: float = 0.1,
        clip_range_vf: Optional[Union[float, Schedule]] = 0.2,
        tsallis_q: float = 1.0,
        method: str = "multistep-SGD",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = True,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Validate method before calling super().__init__
        assert method in ["multistep-SGD", "closedreverse-KL", "closed-KL"], \
            f"Unknown method: {method}. Must be one of ['multistep-SGD', 'closedreverse-KL', 'closed-KL']"
        
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,  # entropy handled differently in MDPO
            vf_coef=0.0,   # value function optimized separately
            max_grad_norm=0.5,  # gradient clipping as in original MDPO
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage
        self.batch_size = batch_size
        self.n_policy_updates = n_policy_updates
        self.n_critic_updates = n_critic_updates
        self.inner_learning_rate = inner_learning_rate
        self.kl_coeff = kl_coeff
        self.clip_range_vf = clip_range_vf
        self.tsallis_q = tsallis_q
        self.method = method
        self.value_learning_rate = value_learning_rate
        
        # Will be populated in _setup_model
        self.policy_params = []
        self.value_params = []
        self.policy_optimizer = None
        self.value_optimizer = None

        # Sanity checks
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            if normalize_advantage:
                assert buffer_size > 1, (
                    "`n_steps * n_envs` must be greater than 1. "
                    f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                )
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Setup model and create separate optimizers for policy and value function."""
        super()._setup_model()
        
        # Separate policy and value function parameters
        self.policy_params = []
        self.value_params = []
        
        for name, param in self.policy.named_parameters():
            # Parameters for value function typically contain 'value' in their name
            if 'value' in name.lower():
                self.value_params.append(param)
            else:
                # Everything else is policy-related (action net, feature extractor, etc.)
                self.policy_params.append(param)
        
        # Debug: print parameter counts
        if self.verbose >= 1:
            print(f"Policy parameters: {sum(p.numel() for p in self.policy_params)}")
            print(f"Value parameters: {sum(p.numel() for p in self.value_params)}")
        
        # Create separate optimizers
        self.policy_optimizer = th.optim.Adam(
            self.policy_params,
            lr=self.learning_rate(1.0) if callable(self.learning_rate) else self.learning_rate,
            eps=1e-5
        )
        
        self.value_optimizer = th.optim.Adam(
            self.value_params,
            lr=self.value_learning_rate if not callable(self.value_learning_rate) else self.value_learning_rate(1.0),
            eps=1e-5
        )

    def _tsallis_kl_divergence(
        self,
        log_prob_new: th.Tensor,
        log_prob_old: th.Tensor,
    ) -> th.Tensor:
        """
        Compute Tsallis-q divergence between new and old policy.
        
        :param log_prob_new: Log probabilities from new policy
        :param log_prob_old: Log probabilities from old policy
        :return: Tsallis-q divergence
        """
        if self.tsallis_q == 1.0:
            # Standard KL divergence when q = 1
            return log_prob_new - log_prob_old
        else:
            # Tsallis-q divergence
            q_value = 2.0 - self.tsallis_q
            prob_new = th.exp(log_prob_new)
            prob_old = th.exp(log_prob_old)
            
            # Tsallis logarithm: log_q(x) = (x^(1-q) - 1) / (1-q)
            def tsallis_log(x: th.Tensor, q: float) -> th.Tensor:
                return (th.pow(x, 1.0 - q) - 1.0) / (1.0 - q + 1e-8)
            
            return tsallis_log(prob_new, q_value) - tsallis_log(prob_old, q_value)

    def _get_schedule_value(self, schedule: Union[float, Schedule], progress_remaining: float) -> float:
        """
        Get the value from a schedule (constant or callable).
        
        :param schedule: Schedule to evaluate
        :param progress_remaining: Progress remaining (1 at start, 0 at end)
        :return: Scheduled value
        """
        if callable(schedule):
            return schedule(progress_remaining)
        return schedule

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Calculate progress remaining for scheduling
        progress_remaining = 1.0 - (self.num_timesteps / self._total_timesteps)

        # Get scheduled values
        inner_lr = self._get_schedule_value(self.inner_learning_rate, progress_remaining)
        outer_lr = self._get_schedule_value(self.learning_rate, progress_remaining)
        clip_range_vf = self._get_schedule_value(self.clip_range_vf, progress_remaining) if self.clip_range_vf else None
        value_lr = self._get_schedule_value(self.value_learning_rate, progress_remaining)

        # Update learning rates
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = outer_lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = value_lr

        policy_losses = []
        value_losses = []
        kl_divergences = []
        
        # Get all data in one go (full batch)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()

            # Re-sample noise if using gSDE
            if self.use_sde:
                self.policy.reset_noise(actions.shape[0])

            # Get old distribution (frozen)
            with th.no_grad():
                old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))
                old_log_prob = old_distribution.log_prob(actions)

            # Normalize advantages
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ============ POLICY UPDATE (Multi-step SGD) ============
            for _ in range(self.n_policy_updates):
                distribution = self.policy.get_distribution(rollout_data.observations)
                log_prob = distribution.log_prob(actions)
                
                # Compute ratio
                ratio = th.exp(log_prob - old_log_prob)
                
                # Compute KL divergence
                if self.tsallis_q == 1.0:
                    kl_div = kl_divergence(distribution, old_distribution).mean()
                else:
                    kl_div_per_sample = self._tsallis_kl_divergence(log_prob, old_log_prob)
                    kl_div = kl_div_per_sample.mean()
                
                # Compute policy objective based on method
                if self.method == "multistep-SGD":
                    # Mirror Descent with KL regularization
                    policy_loss = -(ratio * advantages).mean() + kl_div / (inner_lr + 1e-8)
                
                elif self.method == "closedreverse-KL":
                    # Closed-form reverse KL
                    policy_loss = -(th.exp(advantages) * log_prob).mean()
                
                else:  # closed-KL
                    # Alternative closed-form approach
                    policy_loss = -(ratio * advantages).mean() + inner_lr * (ratio * log_prob).mean()
                
                # Gradient step
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy_params, self.max_grad_norm)
                self.policy_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                kl_divergences.append(kl_div.item())

            # ============ VALUE FUNCTION UPDATE (Clipped Loss) ============
            # Get old values from the full rollout buffer for clipping
            # Convert from numpy to torch tensor and move to device
            old_values = th.as_tensor(self.rollout_buffer.values.flatten(), device=self.device, dtype=th.float32)
            
            for _ in range(self.n_critic_updates):
                # Can use mini-batches for value function
                batch_start = 0
                for batch_rollout_data in self.rollout_buffer.get(self.batch_size):
                    batch_observations = batch_rollout_data.observations
                    batch_returns = batch_rollout_data.returns
                    
                    # Get corresponding old values for this batch
                    batch_size_actual = batch_observations.shape[0]
                    batch_old_values = old_values[batch_start:batch_start + batch_size_actual]
                    batch_start += batch_size_actual
                    
                    # Predict values
                    values_pred = self.policy.predict_values(batch_observations).flatten()
                    
                    # Clipped value loss (similar to PPO)
                    if clip_range_vf is not None:
                        # Compute value prediction error
                        values_diff = values_pred - batch_old_values
                        # Clip the difference
                        values_diff_clipped = th.clamp(
                            values_diff,
                            -clip_range_vf,
                            clip_range_vf
                        )
                        # Compute clipped values
                        values_pred_clipped = batch_old_values + values_diff_clipped
                        
                        # Compute both losses
                        vf_loss1 = functional.mse_loss(values_pred, batch_returns)
                        vf_loss2 = functional.mse_loss(values_pred_clipped, batch_returns)
                        value_loss = th.max(vf_loss1, vf_loss2)
                    else:
                        value_loss = functional.mse_loss(values_pred, batch_returns)
                    
                    value_losses.append(value_loss.item())
                    
                    # Gradient step
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    self.value_optimizer.step()
                
                # Reset batch_start for next critic update iteration
                batch_start = 0

        self._n_updates += 1
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), 
            self.rollout_buffer.returns.flatten()
        )

        # Logging
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/inner_learning_rate", inner_lr)
        self.logger.record("train/outer_learning_rate", outer_lr)
        self.logger.record("train/tsallis_q", self.tsallis_q)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self: SelfMDPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MDPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfMDPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
