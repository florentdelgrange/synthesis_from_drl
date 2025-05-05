import os
import sys
import gc
import threading
import time
import warnings

from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay, PolynomialDecay

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')

import random

import numpy as np
from tf_agents import specs
from tf_agents.trajectories import trajectory, PolicyStep
from tf_agents.typing import types
from tf_agents.typing.types import Int, Float
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import Progbar
from reinforcement_learning.goal_conditioned import ConditionalLabelingCombiner, HindsightExperienceReplay, \
    SamplingStrategy, HindsightExperienceReplayBuffer

import tf_agents
from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment, TimeLimit, TFPyEnvironment, PyEnvironment
from tf_agents.metrics import tf_metrics, py_metrics
from tf_agents.networks import q_network, categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer, reverb_replay_buffer, reverb_utils
from tf_agents.utils import common
from tf_agents.policies import policy_saver, q_policy, py_tf_eager_policy, greedy_policy, EpsilonGreedyPolicy, \
    categorical_q_policy
import tf_agents.trajectories.time_step as ts
from reinforcement_learning.environments import EnvironmentLoader
# specific to the pacman grid environment
from reinforcement_learning.environments.pacman.metric import LabelMetric

import reinforcement_learning
import wasserstein_mdp
from layers.encoders import EncodingType
from policies.latent_policy import LatentPolicyOverRealStateSpace
from policies.one_hot_categorical import OneHotTFPolicyWrapper
from reinforcement_learning.agents.wae_agent import WaeDqnAgent
from reinforcement_learning.environments.perturbed_env import PerturbedEnvironment
from util.io.dataset_generator import map_rl_trajectory_to_vae_input, ergodic_batched_labeling_function
from util.nn import ModelArchitecture
from policies.saved_policy import SavedTFPolicy
from wasserstein_mdp import WassersteinMarkovDecisionProcess, WassersteinRegularizerScaleFactor
from reinforcement_learning.flags import FLAGS, default_flags
from util.io.video import VideoEmbeddingObserverNumpy, VideoEmbeddingObserver

from typing import Tuple, Callable, Optional, Collection, Dict, Union
import functools
import datetime

tfd = tfp.distributions

try:
    import reverb
except ImportError as ie:
    print(ie, "Reverb is not installed on your system, "
              "meaning prioritized experience replay cannot be used.")
try:
    import wandb

    use_wandb = True
except ImportError as ie:
    print(ie, 'Wandb failed to load; tensorboard will be used instead to log training infos.')
    use_wandb = False

from absl import app

default_fc_architecture = ModelArchitecture(hidden_units=(256, 256), activation='relu')
default_cnn_architecture = ModelArchitecture(
    activation='relu',
    raw_last=False,
    filters=(32, 64, 32),
    kernel_size=(8, 4, 3),
    strides=(4, 2, 1),
    padding=("valid", "valid", "valid"),
)

time_limit = None

keep_last_model: bool = True

class WaeDqnLearner:
    def __init__(
            self,
            env_name: str,
            env_suite,
            labeling_fn: Callable[[Float], Float],
            latent_state_size: int,
            initial_collect_steps: Optional[int] = None,
            initial_collect_episodes: Optional[int] = None,
            num_iterations: int = int(2e5),
            collect_steps_per_iteration: Optional[int] = None,
            collect_episodes_per_iteration: Optional[int] = None,
            replay_buffer_capacity: int = int(1e6),
            network_fc_layer_params: ModelArchitecture = default_fc_architecture,
            network_conv_layer_params: ModelArchitecture = default_cnn_architecture,
            state_encoder_temperature: Float = 2. / 3,
            state_prior_temperature=1. / 2,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=20.,
                global_gradient_penalty_multiplier=10.),
            gamma: float = 0.99,
            minimizer_learning_rate: float = 3e-4,
            maximizer_learning_rate: float = 3e-4,
            encoder_learning_rate: Optional[float] = None,
            dqn_learning_rate: float = 3e-4,
            dqn_adam_epsilon: float = 1.5e-4,
            log_interval: int = 200,
            num_eval_episodes: int = 30,
            eval_interval: int = int(1e4),
            num_parallel_environments: int = 4,
            wae_batch_size: int = 128,
            dqn_batch_size: int = 64,
            n_wae_critic: Int = 5,
            n_wae_updates: Int = 5,
            save_directory_location: str = '.',
            hindsight_experience_replay: bool = False,
            her_sampling_strategy: str = 'future',
            prioritized_experience_replay: bool = False,
            priority_exponent: float = 0.6,
            importance_sampling_exponent: float = 1.,
            wae_eval_steps: Int = int(1e4),
            seed: Optional[int] = 42,
            epsilon_greedy: Optional[types.FloatOrReturningFloat] = 0.1,
            boltzmann_temperature: Optional[types.FloatOrReturningFloat] = None,
            final_exploration_step: Optional[int] = None,
            target_update_period: int = 20,
            target_update_tau: types.Float = 1.0,
            reward_scale_factor: types.Float = 1.0,
            gradient_clipping: Optional[types.Float] = None,
            env_time_limit: Optional[Int] = None,
            env_perturbation: Optional[Float] = .25,
            summarize_grads_and_vars: bool = False,
            state_components_concat_units: Tuple[int] = 32,
            pre_processing_net_raw_last: bool = False,
            log_name: Optional[str] = None,
            checkpoint: bool = True,
            save_best_model: bool = True,
            cost_fn: Optional[Dict[str, Union[str, Collection[str]]]] = None,
            cost_weights: Optional[Dict[str, float]] = None,
            use_batch_norm: bool = False,
            log_videos: bool = False,
            her_n_future_samples: int = 4,
            her_time_step_cost: float = 1.,
            her_state_penalty_multiplier: float = 1.,
            her_goal_reward_multiplier: float = 0.,
            her_reward_horizon: int = 100,
            her_epochs: Optional[int] = None,
            her_cycles: int = 50,
            her_optimization_steps: int = 40,
            eval_thread: bool = True,
            categorical: bool = False,
            num_atoms: int = 51,
            n_step_update: int = 1,
            min_q_value: float = -1.,
            max_q_value: float = 1.,
            concatenate_losses: bool = False,
            concatenate_alpha: float = 0.01,
            train_on_discrete_states: bool = False,
            wae_mdp_clip_by_global_norm: Optional[float] = None,
            steady_state_softclip: bool = False,
            straight_through: bool = False,
            softclip_fn: str = 'sofclip',
            use_total_variation: bool = False,
            anneal_temperature: bool = False,
            final_temperature_step: Optional[int] = None,
    ):
        self.parallel_envs = num_parallel_environments > 1 \
                             and not (prioritized_experience_replay or hindsight_experience_replay)
        assert sum(1 for p in (initial_collect_steps, initial_collect_episodes) if p is not None) == 1, \
            "either both or none of initial_collect_steps or initial collect episodes have been provided. "
        assert sum(1 for p in (collect_steps_per_iteration, collect_episodes_per_iteration) if p is not None) == 1, \
            "either both or none of collect_steps_per_iteration or collect_episodes_per_iteration have been provided."
        assert not hindsight_experience_replay or collect_episodes_per_iteration, \
            "collect episodes need be provided if hindsight experience replay is used"

        if collect_steps_per_iteration is not None and self.parallel_envs:
            replay_buffer_capacity = replay_buffer_capacity // num_parallel_environments
            collect_steps_per_iteration = max(1, collect_steps_per_iteration // num_parallel_environments)

        self.env_name = env_name
        self.env_suite = env_suite
        self.initial_collect_steps = initial_collect_steps \
            if initial_collect_episodes is None else initial_collect_episodes
        self._collect_episodes = initial_collect_episodes is not None
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_capacity = replay_buffer_capacity
        self.gamma = gamma
        self.log_interval = log_interval
        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval
        self.wae_eval_steps = wae_eval_steps
        self.num_parallel_environments = num_parallel_environments
        self.dqn_batch_size = dqn_batch_size
        self.wae_batch_size = wae_batch_size
        self.prioritized_experience_replay = prioritized_experience_replay
        self.hindsight_experience_replay = hindsight_experience_replay
        self.her_cycles = her_cycles
        self.her_optimization_steps = her_optimization_steps
        self.n_wae_critic = n_wae_critic
        self.n_wae_updates = n_wae_updates if not concatenate_losses else 1
        self.labeling_fn = labeling_fn
        self.save_best_model = save_best_model
        self.num_iterations = num_iterations
        self.concatenate_losses = concatenate_losses
        self.concatenate_alpha = concatenate_alpha
        if self.concatenate_losses:
            self.num_iterations *= self.n_wae_critic
            self.num_wae_updates = 1
        self.her_epochs = her_epochs if her_epochs is not None else max(
            1, int(num_iterations / (her_cycles * her_optimization_steps * n_wae_updates * n_wae_critic)))
        self.is_exponent = importance_sampling_exponent
        self._log_videos = log_videos
        self._eval_thread = eval_thread
        self.categorical = categorical
        self.train_on_discrete_states = train_on_discrete_states

        # step counters
        self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name="global_step")
        self.dqn_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name="dqn_step")
        self.wae_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64, name="wae_step")
        if reward_scale_factor:
            min_q_value *= reward_scale_factor
            max_q_value *= reward_scale_factor

        if not train_on_discrete_states and straight_through:
            warnings.warn("Straight through estimator is only available for discrete states. "
                          "The training will be performed on discrete states.")
            train_on_discrete_states = True

        if final_exploration_step is not None:
            if epsilon_greedy is not None:
                class EpsilonGreedy:
                    step = self.dqn_step
                    schedule = tf.optimizers.schedules.PolynomialDecay(
                        1.,
                        decay_steps=final_exploration_step,
                        end_learning_rate=tf.cast(epsilon_greedy, tf.float32))

                    def __call__(self, *args, **kwargs):
                        return self.schedule(self.step)

                epsilon_greedy = EpsilonGreedy()
            if boltzmann_temperature is not None:
                class BoltzmannTemperature:
                    step = self.dqn_step
                    schedule = tf.optimizers.schedules.PolynomialDecay(
                        tf.cast(boltzmann_temperature, tf.float32),
                        decay_steps=final_exploration_step,
                        end_learning_rate=1e-1)

                    def __call__(self, *args, **kwargs):
                        return self.schedule(self.step)

                boltzmann_temperature = BoltzmannTemperature()

        self.anneal_temperature = anneal_temperature
        self._anneal_temp_with_time_limit = False
        if final_temperature_step is None and time_limit is not None:
            final_temperature_step = time_limit
            self._anneal_temp_with_time_limit = True
        else:
            final_temperature_step = num_iterations // (n_wae_updates * n_wae_critic)
        if anneal_temperature:
            self._state_encoder_temperature_schedule = tf.optimizers.schedules.PolynomialDecay(
                state_encoder_temperature,
                decay_steps=final_temperature_step,
                end_learning_rate=5e-2,
            )
            self._state_prior_temperature_schedule = tf.optimizers.schedules.PolynomialDecay(
                state_prior_temperature,
                decay_steps=final_temperature_step,
                end_learning_rate=5e-2,
            )
            state_encoder_temperature = tf.Variable(
                initial_value=state_encoder_temperature,
                trainable=False,
                name='state_encoder_temperature',
                dtype=tf.float32,
            )
            self._state_encoder_temperature = state_encoder_temperature
            state_prior_temperature = tf.Variable(
                initial_value=state_prior_temperature,
                trainable=False,
                name='state_prior_temperature',
                dtype=tf.float32
            )
            self._state_prior_temperature = state_prior_temperature

        # set the wae network components to the same architecture
        state_encoder_network = transition_network = reward_network = decoder_network = \
            steady_state_lipschitz_network = transition_loss_lipschitz_network = network_fc_layer_params

        # set up the environment loader
        env_loader = EnvironmentLoader(env_suite, seed=seed)
        env_wrappers = []
        if env_time_limit is not None:
            env_wrappers.append(
                lambda env: TimeLimit(env, env_time_limit))
        # recursive perturbation trick to enforce ergodicity
        if env_perturbation > 0.:
            env_wrappers.append(
                lambda env: PerturbedEnvironment(
                    env,
                    perturbation=env_perturbation,
                    recursive_perturbation=True))

        # load the environment
        if self.parallel_envs:
            self.tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_loader.load(env_name, env_wrappers)] * num_parallel_environments))
            _obs = self.tf_env.reset().observation
            self.py_env = env_suite.load(env_name)
            self.py_env.reset()
            # self.eval_env = tf_py_environment.TFPyEnvironment(self.py_env)
        else:
            self.py_env = env_loader.load(env_name, env_wrappers)
            self.py_env.reset()
            self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
            _obs = self.tf_env.reset().observation
            # self.eval_env = tf_py_environment.TFPyEnvironment(env_suite.load(env_name))

        if hasattr(self.py_env, 'gamma'):
            self.py_env.gamma = self.gamma

        self.observation_spec = self.tf_env.observation_spec()
        self.latent_observation_spec = specs.BoundedTensorSpec(
            shape=(latent_state_size,),
            dtype=tf.float32,
            minimum=0.,
            maximum=1.,
            name='latent_state')
        self.action_spec = self.tf_env.action_spec()

        if categorical:
            self.q_network = categorical_q_network.CategoricalQNetwork(
                # Q-network inputs are latent states
                self.latent_observation_spec,
                self.tf_env.action_spec(),
                num_atoms=num_atoms,
                fc_layer_params=network_fc_layer_params.hidden_units, )
            policy = categorical_q_policy.CategoricalQPolicy(
                time_step_spec=ts.time_step_spec(self.latent_observation_spec),
                action_spec=self.tf_env.action_spec(),
                q_network=self.q_network,
                min_q_value=min_q_value,
                max_q_value=max_q_value,)
        else:
            self.q_network = q_network.QNetwork(
                # Q-network inputs are latent states
                self.latent_observation_spec,
                self.tf_env.action_spec(),
                fc_layer_params=network_fc_layer_params.hidden_units, )

            # Q-policy is a Categorical distribution policy with logits inferred based on the Q-network
            policy = q_policy.QPolicy(
                time_step_spec=ts.time_step_spec(self.latent_observation_spec),
                action_spec=self.tf_env.action_spec(),
                q_network=self.q_network, )

        # policy that can be fed as input of the WAE-MDP
        wae_policy = greedy_policy.GreedyPolicy(
            OneHotTFPolicyWrapper(
                policy,
                time_step_spec=policy.time_step_spec,
                action_spec=specs.BoundedTensorSpec(
                    shape=policy.action_spec.shape,
                    dtype=tf.float32,
                    minimum=policy.action_spec.minimum,
                    maximum=policy.action_spec.maximum)))

        # DQN optimizer
        dqn_optimizer = tf.keras.optimizers.Adam(learning_rate=dqn_learning_rate, epsilon=dqn_adam_epsilon)
        # WAE-MDP optimizers
        if self.concatenate_losses:
            wae_mdp_minimizer = dqn_optimizer
            encoder_optimizer = dqn_optimizer
        else:
            wae_mdp_minimizer = tf.keras.optimizers.Adam(learning_rate=minimizer_learning_rate)
            if encoder_learning_rate is not None:
                encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_learning_rate)
            else:
                encoder_optimizer = None
        wae_mdp_maximizer = tf.keras.optimizers.Adam(learning_rate=maximizer_learning_rate)

        # initialize WAE-MDP
        print("Observation Spec")
        print("================")
        for _spec in tf.nest.flatten(self.tf_env.observation_spec()):
            print(f'> {_spec.name}: shape={_spec.shape}, {_spec.dtype}, bounds=[{_spec.minimum}, {_spec.maximum}]')

        reward_shape = self.tf_env.time_step_spec().reward.shape
        if reward_shape == ():
            reward_shape = (1,)
        state_shape = [tuple(obs_spec.shape) for obs_spec in tf.nest.flatten(self.tf_env.observation_spec())]
        if len(state_shape) == 1:
            state_shape = state_shape[0]

        if hasattr(self.tf_env.observation_spec(), 'keys'):
            input_names = tuple(self.tf_env.observation_spec().keys())
        else:
            input_names = None

        state_pre_proc_nets = [
            network_conv_layer_params
            if len(_state_spec.shape) >= 2
            else None
            for i, _state_spec in enumerate(tf.nest.flatten(self.tf_env.observation_spec()))
        ]
        if len(state_pre_proc_nets) == 1:
            state_pre_proc_nets = state_pre_proc_nets[0]

        if cost_fn is not None:
            n_states = len(tf.nest.flatten(self.tf_env.observation_spec()))
            if n_states > 1 and type(cost_fn['state']) == str:
                cost_fn['state'] = tuple([cost_fn['state']] * n_states)
                cost_weights['state'] = tuple([cost_weights['state']] * n_states)
            elif n_states > 1 and type(cost_fn['state']) in (tuple, list):
                assert len(cost_fn['state']) == n_states, \
                    "One cost_fn by state space component should be provided."
                cost_fn['state'] = tuple(cost_fn['state'])
                cost_weights['state'] = tuple(cost_weights['state'])
            elif n_states > 1:
                warnings.warn("The cost function provided should be either a str or a collection of str.")
                cost_fn = None
        self.wae_mdp = WassersteinMarkovDecisionProcess(
            state_shape=state_shape,
            action_shape=(self.tf_env.action_spec().maximum + 1,),
            reward_shape=reward_shape,
            label_shape=labeling_fn(_obs).shape[1:],
            discretize_action_space=False,
            state_encoder_network=state_encoder_network._replace(raw_last=pre_processing_net_raw_last),
            state_encoder_temperature=state_encoder_temperature,
            state_prior_temperature=state_prior_temperature,
            latent_policy_network=None,
            action_decoder_network=None,
            latent_state_size=latent_state_size,
            transition_network=transition_network,
            reward_network=reward_network,
            decoder_network=decoder_network,
            steady_state_lipschitz_network=steady_state_lipschitz_network,
            transition_loss_lipschitz_network=transition_loss_lipschitz_network,
            n_critic=n_wae_critic,
            external_latent_policy=wae_policy,
            minimizer=wae_mdp_minimizer,
            maximizer=wae_mdp_maximizer,
            encoder_optimizer=encoder_optimizer,
            wasserstein_regularizer_scale_factor=wasserstein_regularizer_scale_factor,
            reset_state_label=env_perturbation > 0.,
            state_encoder_type=EncodingType.DETERMINISTIC,
            deterministic_state_embedding=True,
            trainable_prior=False,
            state_encoder_pre_processing_network=state_pre_proc_nets,
            input_name=input_names,
            input_state_component_concat_units=state_components_concat_units,
            cost_fn=cost_fn,
            cost_weights=cost_weights,
            use_batch_norm=use_batch_norm,
            clip_by_global_norm=wae_mdp_clip_by_global_norm,
            steady_state_net_softclipping=steady_state_softclip,
            straight_through=straight_through,
            softclip_fn=softclip_fn,
            use_total_variation=use_total_variation,
            summary=True)

        # initialize WAE-DQN Agent
        self.tf_agent = WaeDqnAgent(
            time_step_spec=self.tf_env.time_step_spec(),
            latent_time_step_spec=ts.time_step_spec(self.latent_observation_spec),
            action_spec=self.tf_env.action_spec(),
            label_spec=tf.TensorSpec(self.wae_mdp.label_shape),
            q_network=self.q_network,
            optimizer=dqn_optimizer,
            encoder_optimizer=encoder_optimizer,
            epsilon_greedy=epsilon_greedy,
            boltzmann_temperature=boltzmann_temperature,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.dqn_step,
            target_update_period=target_update_period,
            target_update_tau=target_update_tau,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            emit_log_probability=True,
            summarize_grads_and_vars=summarize_grads_and_vars,
            labeling_fn=labeling_fn if env_perturbation <= 0. else ergodic_batched_labeling_function(labeling_fn),
            wae_mdp=self.wae_mdp,
            categorical=categorical,
            n_step_update=n_step_update,
            min_q_value=min_q_value,
            max_q_value=max_q_value,
            concatenate_wae_loss=self.concatenate_losses,
            concatenate_alpha=self.concatenate_alpha,
            straight_through=straight_through
        )
        self.tf_agent.initialize()

        # The collect policy first embeds the original observation to the latent space,
        # then execute the action based on the tf_agent collect policy
        self.collect_policy = LatentPolicyOverRealStateSpace(
            time_step_spec=self.tf_env.time_step_spec(),
            labeling_function=labeling_fn,
            latent_policy=self.tf_agent.collect_policy,
            state_embedding_function=lambda _state, _label: self.wae_mdp.state_embedding_function(
                state=_state,
                label=_label,
                dtype=tf.float32,
                use_discrete_states=self.train_on_discrete_states)
        )

        # Experience Replay
        self.max_priority = tf.Variable(0., trainable=False, name='max_priority', dtype=tf.float64)
        trajectory_spec = trajectory.from_transition(
            time_step=self.tf_env.time_step_spec(),
            action_step=self.collect_policy.policy_step_spec,
            next_time_step=self.tf_env.time_step_spec())
        if self.prioritized_experience_replay:
            if checkpoint:
                checkpoint_path = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn', 'reverb')
                reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(checkpoint_path)
            else:
                reverb_checkpointer = None

            table_name = 'prioritized_replay_buffer'
            table = reverb.Table(
                table_name,
                max_size=replay_buffer_capacity,
                sampler=reverb.selectors.Prioritized(priority_exponent=priority_exponent),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1))
            if checkpoint:
                reverb_server = reverb.Server([table], checkpointer=reverb_checkpointer)
            else:
                reverb_server = reverb.Server([table])

            self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                data_spec=trajectory_spec,
                sequence_length=2,
                table_name=table_name,
                local_server=reverb_server)

            _add_trajectory = reverb_utils.ReverbAddTrajectoryObserver(
                py_client=self.replay_buffer.py_client,
                table_name=table_name,
                sequence_length=2,
                stride_length=1,
                priority=self.max_priority)

            self.num_episodes = py_metrics.NumberOfEpisodes()
            self.env_steps = py_metrics.EnvironmentSteps()
            self.avg_return = py_metrics.AverageReturnMetric(buffer_size=50)
            observers = [self.num_episodes, self.env_steps, self.avg_return, _add_trajectory]

            self.driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=observers,
                max_steps=collect_steps_per_iteration,
                max_episodes=collect_episodes_per_iteration)
            self.initial_collect_driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=[_add_trajectory],
                max_steps=initial_collect_steps,
                max_episodes=initial_collect_episodes)
        else:
            # Hindsight experience replay
            if self.hindsight_experience_replay:
                self.replay_buffer = HindsightExperienceReplayBuffer(
                    data_spec=trajectory_spec,
                    batch_size=self.tf_env.batch_size,
                    max_length=replay_buffer_capacity,
                    goal_augmented_env=self.py_env,
                    sampling_strategy={
                        'final': SamplingStrategy.FINAL,
                        'future': SamplingStrategy.FUTURE
                    }[her_sampling_strategy.lower()],
                    n_future_samples=her_n_future_samples,
                    gamma=gamma,
                    time_step_cost=her_time_step_cost,
                    state_penalty_multiplier=her_state_penalty_multiplier,
                    goal_reward_multiplier=her_goal_reward_multiplier,
                    reward_horizon=her_reward_horizon)
            else:
                self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                    data_spec=trajectory_spec,
                    batch_size=self.tf_env.batch_size,
                    max_length=replay_buffer_capacity)

            self.num_episodes = tf_metrics.NumberOfEpisodes()
            self.env_steps = tf_metrics.EnvironmentSteps()
            self.avg_return = tf_metrics.AverageReturnMetric(batch_size=self.tf_env.batch_size, buffer_size=50)

            observers = [self.num_episodes, self.env_steps] if not self.parallel_envs else []
            observers += [self.avg_return]

            observers += [self.replay_buffer.add_batch]

            # specific to the pacman grid environment
            if 'Pacman' in self.env_name or 'Doom' in self.env_name:
                self.label_metric = LabelMetric(batch_size=self.tf_env.batch_size)
                observers.append(self.label_metric)
            else:
                self.label_metric = lambda _: None

            if initial_collect_steps:
                self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                    self.tf_env,
                    self.collect_policy,
                    observers=[self.replay_buffer.add_batch],
                    num_steps=initial_collect_steps)
            else:
                self.initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                    self.tf_env,
                    self.collect_policy,
                    observers=[self.replay_buffer.add_batch],
                    num_episodes=initial_collect_episodes)
            if collect_steps_per_iteration:
                self.driver = dynamic_step_driver.DynamicStepDriver(
                    self.tf_env, self.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
            else:
                self.driver = dynamic_episode_driver.DynamicEpisodeDriver(
                    self.tf_env, self.collect_policy, observers=observers, num_episodes=collect_episodes_per_iteration)

        # Dataset for WAE-DQN
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=num_parallel_environments,
            sample_batch_size=self.dqn_batch_size,
            num_steps=n_step_update + 1).prefetch(3)
        self.iterator = iter(self.dataset)

        # Dataset for WAE-MDP
        def dataset_generator(generator_fn):
            ds = self.replay_buffer.as_dataset(
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                num_steps=2
            )
            if hasattr(ds, 'grid'):
                map_fn = ds.grid
            else:
                map_fn = ds.map
            return map_fn(
                map_func=generator_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                #  deterministic=False  # TF version >= 2.2.0
            )

        self.wae_dataset = dataset_generator(
            lambda trajectory, buffer_info: map_rl_trajectory_to_vae_input(
                trajectory=trajectory,
                labeling_function=ergodic_batched_labeling_function(labeling_fn),
                discrete_action=True,
                num_discrete_actions=self.tf_env.action_spec().maximum + 1,
                sample_info=buffer_info if self.prioritized_experience_replay else None))
        self.wae_iterator = iter(
            self.wae_dataset.batch(
                batch_size=self.wae_batch_size,
                drop_remainder=True
            ).prefetch(tf.data.experimental.AUTOTUNE))

        # logs
        if log_name is None:
            log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_name += f'-seed={seed}'
        train_log_dir = os.path.join(
            save_directory_location, 'logs', env_name, 'wae_dqn', log_name)
        if use_wandb:
            self.train_summary_writer = None
        else:
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_directory_location = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn')

        # checkpointing
        if checkpoint:
            self.checkpoint_dir = os.path.join(
                save_directory_location, 'saves', env_name, 'wae_dqn', 'training_checkpoint', log_name)
            self.train_checkpointer = common.Checkpointer(
                ckpt_dir=self.checkpoint_dir,
                max_to_keep=1,
                agent=self.tf_agent,
                policy=self.collect_policy,
                # replay_buffer=self.replay_buffer,
                global_step=self.global_step,
                dqn_step=self.dqn_step,
                wae_step=self.wae_step, )
        else:
            self.checkpoint_dir = None
            self.train_checkpointer = None

        self.wae_mdp_save_dir = os.path.join(
            save_directory_location, 'saves', env_name, 'wae_dqn', 'wae_mdp', log_name)
        self.policy_dir = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn', 'policy', log_name)
        self.q_policy_dir = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn', 'q_policy', log_name)
        self.policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)
        self.q_policy_saver = policy_saver.PolicySaver(self.tf_agent._policy)
        self.best_model_dir = os.path.join(save_directory_location, 'saves', env_name, 'wae_dqn', 'best_model')

        if self.checkpoint_dir and os.path.exists(self.checkpoint_dir):
            self.train_checkpointer.initialize_or_restore()
            print("Checkpoint loaded! global_step={:d}; dqn_step={:d}; wae_step={:d}".format(
                self.global_step.numpy(), self.dqn_step.numpy(), self.wae_step.numpy()))
        if not os.path.exists(self.policy_dir):
            os.makedirs(self.policy_dir)

        self.wae_mdp.is_exponent = self.is_exponent

        self._eval_window = []

    @property
    def is_exponent(self):
        return self._is_exponent(self.global_step)

    @is_exponent.setter
    def is_exponent(self, value):
        self._is_exponent = PolynomialDecay(
            initial_learning_rate=value,
            end_learning_rate=1.,
            decay_steps=self.num_iterations, )

    @property
    def eval_window(self):
        return np.array(self._eval_window)

    def update_progress_bar(self, progressbar, wae_mdp_loss, dqn_loss, num_steps=1):
        log_values = [
            ('dqn_loss', dqn_loss),
            ('replay_buffer_frames', self.replay_buffer.num_frames()),
            ('training_avg_return', self.avg_return.result()),
        ]
        # specific to the pacman grid environment
        if 'Pacman' in self.env_name or 'Doom' in self.env_name:
            log_values += [
                ('goal_reached', self.label_metric.result()['goal']),
                ('unsafe', self.label_metric.result()['unsafe']),
            ]
        if not self.parallel_envs:
            log_values += [
                ('num_episodes', self.num_episodes.result()),
                ('env_steps', self.env_steps.result())
            ]

        log_values += \
            [('wae_step', self.wae_step.numpy()), ('dqn_step', self.dqn_step.numpy())] + \
            [(key, value) for key, value in wae_mdp_loss.items()] + \
            [(key, value.result()) for key, value in self.wae_mdp.loss_metrics.items()] + \
            [(key, value) for key, value in self.wae_mdp.temperature_metrics.items()]

        progressbar.add(num_steps, log_values)

    def _train_eval(
            self,
            env: Union[PyEnvironment, TFPyEnvironment],
            display_progressbar: bool = False,
            progressbar: Optional = None,
    ):
        dqn_loss = 0.
        start = time.time()
        global time_limit

        for _ in range(self.global_step.numpy(), self.num_iterations):
            if time_limit is not None:
                if time.time() - start > time_limit:
                    break

            if self.anneal_temperature:
                if self._anneal_temp_with_time_limit:
                    anneal_step = time.time() - start
                else:
                    anneal_step = self.dqn_step
                self._state_encoder_temperature.assign(self._state_encoder_temperature_schedule(anneal_step))
                self._state_prior_temperature.assign(self._state_prior_temperature_schedule(anneal_step))

            # WAE model update
            if not self.concatenate_losses or self.concatenate_alpha > 0.:
                wae_mdp_loss = self.wae_mdp.training_step(
                    dataset=None,
                    dataset_iterator=self.wae_iterator,
                    batch_size=self.wae_batch_size,
                    annealing_period=1,
                    global_step=self.wae_step,
                    display_progressbar=False,
                    progressbar=None,
                    eval_and_save_model_interval=np.inf,
                    eval_steps=self.wae_eval_steps,
                    save_directory=None,
                    log_name='wae_mdp',
                    train_summary_writer=None,
                    log_interval=np.inf,
                    start_annealing_step=0,
                    optimization_direction='max' if self.concatenate_losses else None)
            else:
                wae_mdp_loss = dict()

            if self.global_step.numpy() % (self.n_wae_updates * self.n_wae_critic) == 0:

                # Collect a few steps using collect_policy and save to the replay buffer.
                self.driver.run(env.current_time_step())

                # Use data from the buffer and update the agent's network.
                experience, info = next(self.iterator)
                if self.prioritized_experience_replay:
                    is_weights = tf.cast(
                        tf.reduce_min(info.probability[:, 0, ...]) / info.probability[:, 0, ...],
                        dtype=tf.float32
                    ) ** self.is_exponent
                    loss_info = self.tf_agent.train(experience, weights=is_weights)
                    dqn_loss = loss_info.loss

                    priorities = tf.cast(tf.abs(loss_info.extra.td_error), tf.float64)
                    self.replay_buffer.update_priorities(keys=info.key[:, 0, ...], priorities=priorities)
                    if tf.reduce_max(priorities) > self.max_priority:
                        self.max_priority.assign(tf.reduce_max(priorities))
                    self.wae_mdp.is_exponent = self.is_exponent
                else:
                    loss_info = self.tf_agent.train(experience)
                    dqn_loss = loss_info.loss

            if display_progressbar:
                self.update_progress_bar(progressbar, wae_mdp_loss=wae_mdp_loss, dqn_loss=dqn_loss)

            if self.global_step.numpy() % self.log_interval == 0:
                self.log(extra_logs={'dqn_loss': dqn_loss})

            if self.global_step.numpy() % self.eval_interval == 0 and self.global_step.numpy() != 0:
                if self.train_checkpointer:
                    print("Model checkpointed")
                    self.train_checkpointer.save(self.global_step)
                    if self.prioritized_experience_replay:
                        self.replay_buffer.py_client.checkpoint()
                step = self.dqn_step.numpy() if self.concatenate_losses else self.wae_step.numpy()
                if self._eval_thread:
                    self.policy_saver.save(self.policy_dir)
                    self.q_policy_saver.save(self.q_policy_dir)
                    self.wae_mdp.save(self.wae_mdp_save_dir, 'model')
                    eval_thread = threading.Thread(
                        target=self.eval,
                        args=(step, progressbar),
                        daemon=True,
                        name='eval')
                    eval_thread.start()
                else:
                    self.eval(step, progressbar, load_model=False)

            if self.global_step.numpy() % 100000 == 0 and self.checkpoint_dir:
                self.policy_saver.save(self.policy_dir)
                self.q_policy_saver.save(self.q_policy_dir)
                self.wae_mdp.save(self.wae_mdp_save_dir, 'model')

            self.global_step.assign_add(1)

    def _train_eval_her(
            self,
            env: Union[PyEnvironment, TFPyEnvironment],
            display_progressbar: bool = False,
            progressbar: Optional = None,
    ):
        dqn_loss = 0.
        wae_mdp_loss = 0.

        # global step maintain an epoch counter
        for epoch in range(self.global_step.numpy(), self.her_epochs):
            for cycle in range(self.her_cycles):

                # collect new episodes
                self.driver.run(env.current_time_step())

                for opt_step in range(self.her_optimization_steps):
                    for _ in range(self.n_wae_updates * self.n_wae_critic):

                        # WAE model update
                        wae_mdp_loss = self.wae_mdp.training_step(
                            dataset=None,
                            dataset_iterator=self.wae_iterator,
                            batch_size=self.wae_batch_size,
                            annealing_period=1,
                            global_step=self.wae_step,
                            display_progressbar=False,
                            progressbar=None,
                            eval_and_save_model_interval=np.inf,
                            eval_steps=self.wae_eval_steps,
                            save_directory=None,
                            log_name='wae_mdp',
                            train_summary_writer=None,
                            log_interval=np.inf,
                            start_annealing_step=0, )

                        if self.wae_step.numpy() % self.log_interval == 0:
                            self.log(
                                extra_logs={
                                    'dqn_loss': dqn_loss,
                                    'her_cycle': cycle,
                                    'her_epoch': epoch,
                                    'her_optimization_step': opt_step},
                                step=self.wae_step.numpy())
                            if self.train_checkpointer:
                                print("Model checkpointed")
                                self.train_checkpointer.save(self.wae_step)

                        if display_progressbar:
                            self.update_progress_bar(
                                progressbar,
                                wae_mdp_loss=wae_mdp_loss,
                                dqn_loss=dqn_loss, )

                    # DQN update
                    experience, info = next(self.iterator)
                    loss_info = self.tf_agent.train(experience)
                    dqn_loss = loss_info.loss

            self.eval(self.wae_step.numpy(), progressbar, load_model=False)

            self.global_step.assign_add(1)

    def log(self, extra_logs: Optional[Dict] = None, step: Optional[int] = None):
        if step is None:
            if self.concatenate_losses:
                step = self.dqn_step.numpy()
            else:
                step = self.wae_step.numpy()
        if self.train_summary_writer is not None:
            with self.train_summary_writer.as_default():
                if extra_logs is not None and 'dqn_loss' in extra_logs:
                    tf.summary.scalar('dqn_loss', extra_logs['dqn_loss'], step=self.dqn_step)
                tf.summary.scalar('training average returns', self.avg_return.result(), step=self.dqn_step)
                for key, value in self.wae_mdp.loss_metrics.items():
                    tf.summary.scalar(key, value.result(), step=self.wae_step)
        elif use_wandb:
            logs = {
                **extra_logs,
                **{'training_average_returns': self.avg_return.result(),
                   'wae_step': self.wae_step,
                   'dqn_step': self.dqn_step,
                   'replay_buffer_frames': self.replay_buffer.num_frames(),
                   },
                **({'is_exponent': self.is_exponent,
                    'max_priority': self.max_priority}
                   if self.prioritized_experience_replay
                   else dict()),
                **{key: value.result() for key, value in self.wae_mdp.loss_metrics.items()},
                **({key: value for key, value in self.wae_mdp.temperature_metrics.items()}
                   if self.anneal_temperature else dict()),
            }
            # specific to the pacman grid environment
            if 'Pacman' in self.env_name or 'Doom' in self.env_name:
                logs['goal_reached'] = self.label_metric.result()['goal']
                logs['unsafe'] = self.label_metric.result()['unsafe']
            if self.tf_agent._epsilon_greedy is not None:
                epsilon = self.tf_agent._epsilon_greedy
                logs['epsilon_greedy'] = epsilon() if callable(epsilon) else epsilon
            if self.tf_agent._boltzmann_temperature is not None:
                temperature = self.tf_agent._boltzmann_temperature
                logs['boltzmann_temperature'] = temperature() if callable(temperature) else temperature
            wandb.log(logs, step=step)
        # reset accumulators after logging
        self.wae_mdp.reset_metrics()

    def train_and_eval(self, display_progressbar: bool = True, display_interval: float = 0.1):

        # Optimize by wrapping some of the code in a graph using TF function.
        self.tf_agent.train = common.function(self.tf_agent.train)
        if not self.prioritized_experience_replay and not self.hindsight_experience_replay:
            self.driver.run = common.function(self.driver.run)

        metrics = ['eval_avg_returns', 'avg_eval_episode_length', 'replay_buffer_frames',
                   'training_avg_returns', 'wae_step', 'dqn_step', 't_1', 't_2',
                   "num_episodes", "env_steps", 'dynamic_reward_scaling'
                   ] + list(self.wae_mdp.loss_metrics.keys())

        if not self.parallel_envs:
            metrics += ['num_episodes', 'env_steps']

        if display_progressbar:
            progressbar = Progbar(target=self.num_iterations, interval=display_interval, stateful_metrics=metrics)
            progressbar.update(self.global_step.numpy())
            print('\n')
        else:
            progressbar = None

        env = self.tf_env if not self.prioritized_experience_replay else self.py_env

        if (
                (self._collect_episodes and self.num_episodes.result() <= self.initial_collect_steps) or
                self.replay_buffer.num_frames() <= self.initial_collect_steps
        ):
            print("Initialize replay buffer...")
            self.initial_collect_driver.run(env.current_time_step())

        print("Start training...")

        if self.hindsight_experience_replay:
            self._train_eval_her(env, display_progressbar, progressbar)
        else:
            self._train_eval(env, display_progressbar, progressbar)

        if keep_last_model:
            self.policy_saver.save(self.policy_dir)
            self.q_policy_saver.save(self.q_policy_dir)
            self.wae_mdp.save(self.wae_mdp_save_dir, 'model')
        else:
            # clean files if keeping the last model has not been requested
            def _rm_dir(_path):
                import shutil
                # check if folder exists
                if os.path.exists(_path):
                    # remove if exists
                    shutil.rmtree(_path)
                else:
                    warnings.warn(f"The directory {_path} does not exist.")

            _rm_dir(self.policy_dir)
            _rm_dir(self.wae_mdp_save_dir)

    def eval(self, step: int = 0, progressbar: Optional = None, load_model=True):
        avg_eval_return = tf_metrics.AverageReturnMetric(buffer_size=50)
        avg_eval_episode_length = tf_metrics.AverageEpisodeLengthMetric(buffer_size=50)
        metrics = [avg_eval_return, avg_eval_episode_length]
        # specific to the pacman grid environment
        if 'Pacman' in self.env_name or 'Doom' in self.env_name:
            label_metric = LabelMetric(buffer_size=50)
            metrics.append(label_metric)

        for policy_dir in [self.policy_dir, self.q_policy_dir]:
            for metric in metrics:
                metric.reset()

            if load_model:
                policy = SavedTFPolicy(policy_dir)
                wae_mdp = wasserstein_mdp.load(
                    model_path=os.path.join(self.wae_mdp_save_dir, 'model'),
                    summary=False)
                wae_mdp.external_latent_policy = OneHotTFPolicyWrapper(
                    policy,
                    time_step_spec=policy.time_step_spec,
                    action_spec=policy.action_spec)
            else:
                if policy_dir == self.policy_dir:
                    policy = self.tf_agent.policy
                else:
                    policy = self.tf_agent._policy
                wae_mdp = self.wae_mdp

            if not self.categorical and policy_dir == self.policy_dir:
                policy = EpsilonGreedyPolicy(policy, epsilon=.05)

                def epsilon_greedy_distribution(time_step, policy_state):
                    step = policy.action(time_step, policy_state)
                    return step._replace(action=tfd.Deterministic(step.action))

                policy._distribution = epsilon_greedy_distribution

            py_env = EnvironmentLoader(self.env_suite).load(self.env_name)
            eval_env = wae_mdp.wrap_tf_environment(
                tf_env=tf_py_environment.TFPyEnvironment(py_env),
                labeling_function=self.labeling_fn,
                use_discrete_states=self.train_on_discrete_states)
            latent_policy = eval_env.wrap_latent_policy(
                policy,
                observation_dtype=policy.time_step_spec.observation.dtype)

            eval_env.reset()

            if self._log_videos:
                video_observer = VideoEmbeddingObserverNumpy(py_env)
            else:
                video_observer = lambda _: None
                video_observer.data = None

            driver = dynamic_episode_driver.DynamicEpisodeDriver(
                eval_env,
                latent_policy,
                metrics + [video_observer],
                num_episodes=self.num_eval_episodes)
            driver.run()
            # video_observer.finalize()

            q = '' if policy_dir == self.policy_dir else '_q'
            log_values = [
                (f'eval_avg_return{q}', avg_eval_return.result()),
                (f'avg_eval_episode_length{q}', avg_eval_episode_length.result()),
            ]
            if 'Pacman' in self.env_name or 'Doom' in self.env_name:
                log_values += [
                    (f'eval_goal_reached{q}', label_metric.result()['goal']),
                    (f'eval_unsafe{q}', label_metric.result()['unsafe']),
                ]
            if progressbar is not None:
                progressbar.add(0, log_values)
            else:
                print('Evaluation')
                for key, value in log_values:
                    print(key, '=', value.numpy())

            self._eval_window.append(avg_eval_return.result().numpy())

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Average return', avg_eval_return.result(), step=step)
                    tf.summary.scalar('Average episode length', avg_eval_episode_length.result(), step=step)
            elif use_wandb:
                logs = {
                    f'eval_avg_return{q}': avg_eval_return.result(),
                    f'eval_avg_episode_length{q}': avg_eval_episode_length.result(),
                    f'eval_avg_return_max': max(self._eval_window),
                }
                if 'Pacman' in self.env_name or 'Doom' in self.env_name:
                        logs = {**logs, **{
                        # specific to the pacman grid environment
                        f'eval_goal_reached{q}': label_metric.result()['goal'],
                        f'eval_unsafe{q}': label_metric.result()['unsafe'],
                    }}
                wandb.log(logs, step=self.dqn_step.numpy() if self.concatenate_losses else self.wae_step.numpy())
                if self._log_videos and avg_eval_return.result().numpy() > max(self._eval_window[:-1] + [-np.inf]):
                    wandb.log({
                        "eval_policy": wandb.Video(
                            # os.path.join(self.save_directory_location, 'video.mp4'),
                            np.moveaxis(video_observer.data, -1, 1),
                            caption=f"avg_return={avg_eval_return.result().numpy():.3g}_step={step:d}",
                            fps=12,
                            format='mp4'),
                    }, step=step)

            if self.save_best_model and wae_mdp.assign_score(
                    score={'eval_policy': avg_eval_return.result().numpy()},
                    #  score={'eval_policy': label_metric.result()['goal'].numpy()},
                    model_name='wae_mdp',
                    checkpoint_model=True,
                    training_step=step,
                    save_directory=self.best_model_dir,
            ):
                policy_saver.PolicySaver(policy).save(
                    os.path.join(self.best_model_dir, f'policy'))
                print("policy saved in:", os.path.join(self.best_model_dir, 'policy'))

            del video_observer
            del eval_env
            del py_env
            del driver

            if load_model:
                del wae_mdp
                del policy

        gc.collect()


def load(model_path: str, policy_path: Optional = None):
    wae_mdp = wasserstein_mdp.load(model_path)
    if policy_path is None and os.path.exists(os.path.join(model_path, 'policy')):
        policy_path = os.path.join(model_path, 'policy')
    if policy_path:
        saved_policy = SavedTFPolicy(policy_path)
        wae_mdp.external_latent_policy = OneHotTFPolicyWrapper(
            saved_policy,
            time_step_spec=saved_policy.time_step_spec,
            action_spec=saved_policy.action_spec)
    return wae_mdp


def main(params):
    _params = {key: value for key, value in FLAGS.flag_values_dict().items() if key not in default_flags}
    _params.update(params)
    params = _params
    # set seed
    seed = params['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.random.set_seed(params['seed'])

    global time_limit
    global keep_last_model
    time_limit = params['time_limit']
    keep_last_model = params['keep_last_model']

    try:
        import importlib
        for module in params['import']:
            importlib.import_module(module)
    except BaseException as err:
        serr = str(err)
        print("Error to load module: " + serr)
        return -1

    if use_wandb:
        run = wandb.init(project='wae_dqn', entity=params['wandb_entity'])

        if params['rerun']:
            run_id = params['rerun']

            api = wandb.Api(timeout=120)
            run = api.run(f'{run_id}')
            run_params = run.config
            for p in params.keys():
                if not FLAGS[p].present and p in run_params.keys():
                    params[p] = run_params[p]

            params['save_dir'] = os.path.join(params['save_dir'], f'run_{run.id}', f'seed_{params["seed"]}')

            if 'num_units_per_hidden_layer' in run_params.keys() and 'num_hidden_layers' not in run_params.keys():
                params['network_layers'] = [params['num_units_per_hidden_layer']] * params['num_hidden_layers']

        run.tags += tuple(params['tags'])
        wandb.config.update(params)
    
    labeling_fn = reinforcement_learning.labeling_functions.get(params['env_name'],
        lambda observation: tf.zeros(shape=(tf.shape(tf.nest.flatten(observation)[0])[0], 0), dtype=tf.bool)
    )
    if params['log_name'] is None and use_wandb:
        log_name = wandb.run.name
    else:
        log_name = params['log_name']

    learner = WaeDqnLearner(
        env_name=params['env_name'],
        env_suite=importlib.import_module('tf_agents.environments.' + params['env_suite']),
        labeling_fn=labeling_fn,
        latent_state_size=params['latent_state_size'],
        num_iterations=params['steps'],
        initial_collect_steps=params['initial_collect_steps'],
        initial_collect_episodes=params['initial_collect_episodes'],
        collect_steps_per_iteration=params['collect_steps_per_iteration'],
        collect_episodes_per_iteration=params['collect_episodes_per_iteration'],
        replay_buffer_capacity=params['replay_buffer_size'],
        network_fc_layer_params=ModelArchitecture(
            hidden_units=params['network_layers'],
            activation=params['activation']),
        network_conv_layer_params=ModelArchitecture(
            filters=params['cnn_filters'],
            kernel_size=params['cnn_kernel_size'],
            strides=params['cnn_strides'],
            padding=params['cnn_padding'],
            activation=params['cnn_layers_activation'],
            max_pooling=params['use_max_pooling']),
        state_encoder_temperature=params['state_encoder_temperature'],
        state_prior_temperature=params['state_prior_temperature'],
        wasserstein_regularizer_scale_factor=WassersteinRegularizerScaleFactor(
            global_gradient_penalty_multiplier=params['gradient_penalty_scale_factor'],
            steady_state_scaling=params['steady_state_regularizer_scale_factor'],
            local_transition_loss_scaling=params['transition_regularizer_scale_factor'], ),
        gamma=params['gamma'],
        minimizer_learning_rate=params['wae_minimizer_learning_rate'],
        maximizer_learning_rate=params['wae_maximizer_learning_rate'],
        encoder_learning_rate=params['encoder_learning_rate'],
        dqn_learning_rate=params['policy_learning_rate'],
        log_interval=params['log_interval'],
        num_eval_episodes=params['n_eval_episodes'],
        eval_interval=1 if params['her'] else params['n_eval_interval'],
        num_parallel_environments=params['n_parallel_envs'],
        wae_batch_size=params['wae_batch_size'],
        dqn_batch_size=params['policy_batch_size'],
        n_wae_critic=params['n_wae_critic'],
        n_wae_updates=params['n_wae_updates'],
        save_directory_location=params['save_dir'],
        prioritized_experience_replay=params['prioritized_experience_replay'],
        priority_exponent=params['priority_exponent'],
        wae_eval_steps=params['wae_eval_steps'],
        seed=params['seed'],
        epsilon_greedy=params['epsilon_greedy'] if not params['boltzmann_temperature'] else None,
        final_exploration_step=params['final_exploration_step'],
        boltzmann_temperature=params['boltzmann_temperature'],
        target_update_period=params['target_update_period'],
        target_update_tau=params['target_update_scale'],
        reward_scale_factor=params['reward_scaling'],
        gradient_clipping=params['policy_gradient_clipping'],
        env_time_limit=params['env_time_limit'],
        env_perturbation=params['env_perturbation'],
        summarize_grads_and_vars=params['log_grads_and_vars'],
        state_components_concat_units=params['state_components_concat_units'],
        cost_fn={
            'state': params['state_cost_fn'][0] if len(params['state_cost_fn']) == 1 else tuple(
                params['state_cost_fn']),
            'reward': params['reward_cost_fn'],
        },
        cost_weights={
            'state': params['state_cost_weights'][0] if len(params['state_cost_weights']) == 1 else tuple(
                params['state_cost_weights']),
            'reward': params['reward_cost_weight'],
        },
        pre_processing_net_raw_last=params['pre_processing_net_raw_last'],
        log_name=log_name,
        use_batch_norm=params['use_batch_norm'],
        checkpoint=params['checkpoint'],
        importance_sampling_exponent=params['importance_sampling_exponent'],
        log_videos=params['log_videos'],
        hindsight_experience_replay=params['her'],
        her_sampling_strategy=params['her_sampling_strategy'],
        her_reward_horizon=params['her_reward_horizon'],
        her_time_step_cost=params['her_time_step_cost'],
        her_n_future_samples=params['her_n_future_samples'],
        her_goal_reward_multiplier=params['her_goal_reward_multiplier'],
        her_state_penalty_multiplier=params['her_state_penalty_multiplier'],
        her_cycles=params['her_cycles'],
        her_optimization_steps=params['her_optimization_steps'],
        her_epochs=params['her_epochs'],
        num_atoms=params['num_atoms'],
        categorical=params['categorical'],
        n_step_update=params['n_step_update'],
        min_q_value=params['min_q_value'],
        max_q_value=params['max_q_value'],
        eval_thread=params['eval_thread'],
        concatenate_losses=params['concatenate_losses'],
        concatenate_alpha=params['concatenate_alpha'],
        train_on_discrete_states=params['train_on_discrete_states'],
        wae_mdp_clip_by_global_norm=params['wae_mdp_clip_by_global_norm'],
        steady_state_softclip=params['steady_state_softclip'],
        softclip_fn=params['softclip_fn'],
        straight_through=params['straight_through'],
        use_total_variation=params['use_total_variation'],
        anneal_temperature=params['anneal_temperature'],
        final_temperature_step=params['final_temperature_step'],
        dqn_adam_epsilon=params['policy_adam_epsilon'],
    )

    learner.train_and_eval(display_progressbar=params['display_progressbar'])

    if use_wandb:
        wandb.finish()

    return learner.eval_window


def app_main(argv):
    del argv
    return main(dict())


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, app_main))
    # app.run(main)
