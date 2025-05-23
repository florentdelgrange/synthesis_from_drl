import gc
import json
import os.path
from collections import namedtuple

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Callable, NamedTuple, List, Union, Dict, Collection
from collections.abc import Collection as Collec
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from scipy.optimize import minimize_scalar
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras.metrics import Mean, MeanSquaredError
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd

import tf_agents
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.policies import TFPolicy, tf_policy
from tf_agents.typing import types
from tf_agents.typing.types import Float, Int
from tf_agents.environments import tf_py_environment, tf_environment

from layers.autoregressive_bernoulli import AutoRegressiveBernoulliNetwork
from layers.latent_policy import LatentPolicyNetwork
from layers.decoders import RewardNetwork, ActionReconstructionNetwork, StateReconstructionNetwork
from layers.encoders import StateEncoderNetwork, ActionEncoderNetwork, AutoRegressiveStateEncoderNetwork, EncodingType, \
    DeterministicStateEncoderNetwork
from layers.lipschitz_functions import SteadyStateLipschitzFunction, TransitionLossLipschitzFunction
from layers.steady_state_network import SteadyStateNetwork
from tf_agents.trajectories import time_step as ts, policy_step, PolicyStep
from util.nn import get_activation_fn, ModelArchitecture, generate_sequential_model, get_model, dSiLU
from variational_mdp import VariationalMarkovDecisionProcess, EvaluationCriterion, debug_gradients, debug, epsilon
from verification.local_losses import estimate_local_losses_from_samples


class WassersteinRegularizerScaleFactor(NamedTuple):
    global_scaling: Optional[Float] = None
    global_gradient_penalty_multiplier: Optional[Float] = None
    steady_state_scaling: Optional[Float] = None
    steady_state_gradient_penalty_multiplier: Optional[Float] = None
    local_transition_loss_scaling: Optional[Float] = None
    local_transition_loss_gradient_penalty_multiplier: Optional[Float] = None

    values = namedtuple('WassersteinRegularizer', ['scaling', 'gradient_penalty_multiplier'])

    def sanity_check(self):
        if self.global_scaling is None and (self.steady_state_scaling is None or
                                            self.local_transition_loss_scaling is None):
            raise ValueError("Either a global scaling value or a unique scaling value for"
                             "each Wasserstein regularizer should be provided.")

        if self.global_gradient_penalty_multiplier is None and (
                self.steady_state_gradient_penalty_multiplier is None or
                self.local_transition_loss_gradient_penalty_multiplier is None):
            raise ValueError("Either a global gradient penalty multiplier or a unique multiplier for"
                             "each Wasserstein regularizer should be provided.")

    @property
    def stationary(self):
        self.sanity_check()
        return self.values(
            scaling=self.steady_state_scaling if self.steady_state_scaling is not None else self.global_scaling,
            gradient_penalty_multiplier=(self.steady_state_gradient_penalty_multiplier
                                         if self.steady_state_gradient_penalty_multiplier is not None else
                                         self.global_gradient_penalty_multiplier))

    @property
    def local_transition_loss(self):
        self.sanity_check()
        return self.values(
            scaling=(self.local_transition_loss_scaling
                     if self.local_transition_loss_scaling is not None else
                     self.global_scaling),
            gradient_penalty_multiplier=(self.local_transition_loss_gradient_penalty_multiplier
                                         if self.local_transition_loss_gradient_penalty_multiplier is not None else
                                         self.global_gradient_penalty_multiplier))


class BaseModelArchitecture(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(name="base_model")
        for key, value in kwargs.items():
            setattr(self, key, value)


class WassersteinMarkovDecisionProcess(VariationalMarkovDecisionProcess):
    def __init__(
            self,
            state_shape: Union[Collection[Tuple[int, ...]], Tuple[int, ...]],
            action_shape: Tuple[int, ...],
            reward_shape: Tuple[int, ...],
            label_shape: Tuple[int, ...],
            discretize_action_space: bool,
            state_encoder_network: ModelArchitecture,
            action_decoder_network: ModelArchitecture,
            transition_network: ModelArchitecture,
            reward_network: ModelArchitecture,
            decoder_network: ModelArchitecture,
            latent_policy_network: ModelArchitecture,
            steady_state_lipschitz_network: ModelArchitecture,
            transition_loss_lipschitz_network: ModelArchitecture,
            latent_state_size: int,
            number_of_discrete_actions: Optional[int] = None,
            action_encoder_network: Optional[ModelArchitecture] = None,
            state_encoder_pre_processing_network: Optional[ModelArchitecture] = None,
            state_decoder_post_processing_network: Optional[ModelArchitecture] = None,
            time_stacked_states: bool = False,
            state_encoder_temperature: float = 2. / 3,
            state_prior_temperature: float = 1. / 2,
            action_encoder_temperature: Optional[Float] = None,
            latent_policy_temperature: Optional[Float] = None,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=1., global_gradient_penalty_multiplier=1.),
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            reset_state_label: bool = True,
            minimizer: Optional[tf.optimizers.Optimizer] = None,
            maximizer: Optional[tf.optimizers.Optimizer] = None,
            encoder_optimizer: Optional[tf.optimizers.Optimizer] = None,
            entropy_regularizer_scale_factor: float = 0.,
            entropy_regularizer_decay_rate: float = 0.,
            entropy_regularizer_scale_factor_min_value: float = 0.,
            importance_sampling_exponent: Optional[Float] = 1.,
            importance_sampling_exponent_growth_rate: Optional[Float] = 0.,
            time_stacked_lstm_units: int = 128,
            reward_bounds: Optional[Tuple[float, float]] = None,
            latent_stationary_network: Optional[tfk.Model] = None,
            action_entropy_regularizer_scaling: float = 0.,
            enforce_upper_bound: bool = False,
            squared_wasserstein: bool = False,
            n_critic: int = 5,
            trainable_prior: bool = False,
            state_encoder_type: EncodingType = EncodingType.AUTOREGRESSIVE,
            policy_based_decoding: bool = False,
            deterministic_state_embedding: bool = True,
            state_encoder_softclipping: bool = False,
            steady_state_net_softclipping: bool = False,
            external_latent_policy: Optional[TFPolicy] = None,
            input_name: Optional[Union[Tuple[str, ...], str]] = None,
            input_state_component_concat_units: Union[int, Tuple[int, ...]] = 32,
            cost_fn: Optional[Dict[str, str]] = None,
            cost_weights: Optional[Dict[str, float]] = None,
            use_batch_norm: bool = False,
            summary: bool = False,
            clip_by_global_norm: Optional[float] = None,
            # approximate total variation instead of approximating wasserstein distance
            # in that case, applying gradient penalty to enforce the one Lipschitzness is unnecessary
            use_total_variation: bool = False,
            softclip_fn: str = 'softclip',
            straight_through: bool = False,
            *args, **kwargs
    ):
        super(WassersteinMarkovDecisionProcess, self).__init__(
            state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape, label_shape=label_shape,
            encoder_network=None, transition_network=None, reward_network=None, decoder_network=None,
            time_stacked_states=time_stacked_states, latent_state_size=latent_state_size,
            encoder_temperature=state_encoder_temperature, prior_temperature=state_prior_temperature,
            encoder_temperature_decay_rate=encoder_temperature_decay_rate,
            prior_temperature_decay_rate=prior_temperature_decay_rate,
            pre_loaded_model=True, optimizer=None,
            reset_state_label=reset_state_label,
            evaluation_window_size=0,
            evaluation_criterion=EvaluationCriterion.MEAN,
            importance_sampling_exponent=importance_sampling_exponent,
            importance_sampling_exponent_growth_rate=importance_sampling_exponent_growth_rate,
            time_stacked_lstm_units=time_stacked_lstm_units,
            reward_bounds=reward_bounds,
            entropy_regularizer_scale_factor=entropy_regularizer_scale_factor,
            entropy_regularizer_scale_factor_min_value=entropy_regularizer_scale_factor_min_value,
            entropy_regularizer_decay_rate=entropy_regularizer_decay_rate,
            deterministic_state_embedding=deterministic_state_embedding)

        # for saving the model
        _params = list(locals().items())
        self._params = {key: str(value) for key, value in _params}
        self.wasserstein_regularizer_scale_factor = wasserstein_regularizer_scale_factor
        self.mixture_components = None
        self._minimizer = minimizer
        self._maximizer = maximizer
        self._encoder_optimizer = encoder_optimizer
        self.action_discretizer = discretize_action_space
        self.policy_based_decoding = policy_based_decoding
        self.action_entropy_regularizer_scaling = action_entropy_regularizer_scaling
        self.enforce_upper_bound = enforce_upper_bound
        self.squared_wasserstein = squared_wasserstein
        self.n_critic = n_critic
        self.trainable_prior = trainable_prior
        self.include_state_encoder_entropy = not (
                entropy_regularizer_scale_factor < epsilon
                or state_encoder_type is EncodingType.DETERMINISTIC)
        self.include_action_encoder_entropy = not (action_entropy_regularizer_scaling < epsilon)
        self._state_encoder_type = state_encoder_type
        self.external_latent_policy = external_latent_policy
        self._cost_fn = {
            'state': 'l2',
            'action': 'l2',
            'reward': 'l2',
        }
        self._cost_weights = {
            'state': 1.,
            'action': 1.,
            'reward': 1.,
        }
        if cost_fn is not None:
            self._cost_fn.update(cost_fn)
        if cost_weights is not None:
            self._cost_weights.update(cost_weights)
        if isinstance(self._cost_weights['state'], Collec) and all(
                weight == 0. for weight in self._cost_weights['state']
        ):
            self._cost_weights['state'] = 0.

        if self.action_discretizer:
            self.number_of_discrete_actions = number_of_discrete_actions
        else:
            assert len(action_shape) == 1
            self.number_of_discrete_actions = self.action_shape[0]

        if isinstance(state_encoder_temperature, tf.Variable):
            self._encoder_temperature = state_encoder_temperature
            self.temperature_metrics['t_1'] = self.encoder_temperature
        if isinstance(state_prior_temperature, tf.Variable):
            self._prior_temperature = state_prior_temperature
            self.temperature_metrics['t_2'] = self.prior_temperature

        self._action_encoder_temperature = None
        if action_encoder_temperature is None:
            self.action_encoder_temperature = 1. / (self.number_of_discrete_actions - 1)
        else:
            self.action_encoder_temperature = action_encoder_temperature

        self._latent_policy_temperature = None
        if latent_policy_temperature is None:
            self.latent_policy_temperature = self.action_encoder_temperature / 1.5
        else:
            self.latent_policy_temperature = latent_policy_temperature

        self.clip_by_global_norm = clip_by_global_norm

        self._sample_additional_transition = False

        self._dTV_clip_method = softclip_fn
        if softclip_fn == 'dsilu':
            # Find the maximum value of dSiLU
            dSiLU_np = lambda _x: dSiLU(_x).numpy()
            max_result = minimize_scalar(lambda x: -dSiLU_np(x), bounds=(-10, 10), method='bounded')
            dSiLU_max = -max_result.fun

            # Find the minimum value of dSiLU
            min_result = minimize_scalar(dSiLU_np, bounds=(-10, 10), method='bounded')
            dSiLU_min = min_result.fun
        else:
            dSiLU_max = dSiLU_min = 0.

        _k = .25
        hinge_softness = 2.
        softclip_fn = {
            'softclip': tfb.SoftClip(low=-1., high=1., hinge_softness=hinge_softness),
            'dsilu': lambda x: 2. * (dSiLU(_k * x) - dSiLU_min) / (dSiLU_max - dSiLU_min) - 1.,
            'scaled_tanh': lambda x: _k * x * tf.pow((1. + tf.square(_k * x)), -.5),
            'tanh': tf.math.tanh,
            'sigmoid': lambda x: 2 * tf.math.sigmoid(hinge_softness * x) - 1.,
        }.get(softclip_fn, 'softclip')

        if state_encoder_softclipping:
            self.softclip = softclip_fn
        else:
            self.softclip = tfb.Identity()

        self.use_total_variation = use_total_variation
        self.straight_through = straight_through

        try:
            n_states = len(state_shape)
            state = []
            assert input_name is None or len(input_name) == len(state_shape), "The number of elements in input_name " \
                                                                              f"should be the same as the number of state components. Length of input_name: {len(input_name):d}. " \
                                                                              f"Expected: {len(state_shape):d}."
            for i, shape in enumerate(state_shape):
                state.append(
                    tfkl.Input(shape=shape, name=f"state_{i:d}" if input_name is None else input_name[i]))
            n_state_components = len(state)
        except TypeError:
            n_states = 1
            state = tfkl.Input(shape=state_shape, name="state" if input_name is None else input_name)
            n_state_components = 1
        action = tfkl.Input(shape=action_shape, name="action")
        latent_state = tfkl.Input(shape=(self.latent_state_size,), name="latent_state")
        latent_action = tfkl.Input(shape=(self.number_of_discrete_actions,), name="latent_action")
        next_latent_state = tfkl.Input(shape=(self.latent_state_size,), name='next_latent_state')

        # pre-processing
        def _get_pre_processing_net(
                state_encoder_pre_processing_network: ModelArchitecture,
                prefix: Optional[str] = ''
        ):
            if state_encoder_pre_processing_network is None:
                return None
            state_pre_proc_nets = []
            if type(state_encoder_pre_processing_network) is ModelArchitecture:
                state_encoder_pre_processing_network = [state_encoder_pre_processing_network]
            for i, net in enumerate(state_encoder_pre_processing_network):
                _state_shape = state_shape if n_state_components == 1 else state_shape[i]
                if net is not None and net.is_cnn:
                    state_pre_proc_nets.append(
                        EncodingNetwork(
                            input_tensor_spec=tf.TensorSpec(shape=_state_shape, name=f'state_{i:d}'),
                            activation_fn=get_activation_fn(net.activation),
                            fc_layer_params=(
                                net.hidden_units
                                if net.hidden_units is not None and len(net.hidden_units) > 0
                                else None),
                            conv_layer_params=list(zip(
                                net.filters,
                                net.kernel_size,
                                net.strides)),
                            name=net.name if net.name is not None else f'{prefix}_pre_proc_conv_net_for_input_{i:d}'))
                elif net is not None:
                    state_pre_proc_nets.append(
                        EncodingNetwork(
                            input_tensor_spec=tf.TensorSpec(shape=_state_shape, name=f'state_{i:d}'),
                            activation_fn=get_activation_fn(net.activation),
                            fc_layer_params=net.hidden_units,
                            name=net.name if net.name is not None else f'{prefix}_pre_proc_fc_net_for_input_{i:d}'))
                else:
                    state_pre_proc_nets.append(None)
            if n_state_components == 1:
                state_pre_proc_nets = state_pre_proc_nets[0]
            return state_pre_proc_nets

        # state encoder network
        state_pre_proc_nets = _get_pre_processing_net(state_encoder_pre_processing_network, 'state_encoder')
        if state_encoder_type is EncodingType.AUTOREGRESSIVE:
            hidden_units, activation = (state_encoder_network.hidden_units,
                                        get_activation_fn(state_encoder_network.activation))
            self.state_encoder_network = AutoRegressiveStateEncoderNetwork(
                state_shape=state_shape,
                activation=activation,
                hidden_units=hidden_units,
                latent_state_size=self.latent_state_size,
                atomic_prop_dims=self.atomic_prop_dims,
                time_stacked_states=self.time_stacked_states,
                temperature=self.state_encoder_temperature,
                time_stacked_lstm_units=self.time_stacked_lstm_units,
                pre_proc_net=state_pre_proc_nets,
                output_softclip=self.softclip, )
        elif state_encoder_type is EncodingType.DETERMINISTIC:
            self.state_encoder_network = DeterministicStateEncoderNetwork(
                state=state,
                activation=get_activation_fn(state_encoder_network.activation),
                hidden_units=state_encoder_network.hidden_units,
                latent_state_size=latent_state_size,
                atomic_prop_dims=self.atomic_prop_dims,
                time_stacked_states=time_stacked_states,
                output_softclip=self.softclip,
                pre_proc_net=state_pre_proc_nets,
                concat_units=input_state_component_concat_units,
                raw_last=state_encoder_network.raw_last,
                use_batch_norm=use_batch_norm)
        else:
            self.state_encoder_network = StateEncoderNetwork(
                state=state,
                activation=get_activation_fn(state_encoder_network.activation),
                hidden_units=state_encoder_network.hidden_units,
                latent_state_size=self.latent_state_size,
                atomic_prop_dims=self.atomic_prop_dims,
                time_stacked_states=self.time_stacked_states,
                time_stacked_lstm_units=self.time_stacked_lstm_units,
                pre_proc_net=state_pre_proc_nets,
                output_softclip=self.softclip,
                lstm_output=state_encoder_type is EncodingType.LSTM,
                concat_units=input_state_component_concat_units,
                raw_last=state_encoder_network.raw_last)
        # action encoder network
        if self.action_discretizer and not self.policy_based_decoding:
            self.action_encoder_network = ActionEncoderNetwork(
                latent_state=latent_state,
                action=action,
                number_of_discrete_actions=self.number_of_discrete_actions,
                action_encoder_network=generate_sequential_model(action_encoder_network), )
        else:
            self.action_encoder_network = None
        # transition network
        self.transition_network = AutoRegressiveBernoulliNetwork(
            event_shape=(self.latent_state_size,),
            activation=get_activation_fn(transition_network.activation),
            hidden_units=transition_network.hidden_units,
            conditional_event_shape=(self.latent_state_size + self.number_of_discrete_actions,),
            temperature=self.state_prior_temperature,
            name='autoregressive_transition_network')
        # stationary distribution over latent states
        self.latent_stationary_network: AutoRegressiveBernoulliNetwork = SteadyStateNetwork(
            atomic_prop_dims=self.atomic_prop_dims,
            latent_state_size=latent_state_size,
            activation=get_activation_fn(transition_network.activation),
            hidden_units=transition_network.hidden_units,
            trainable_prior=trainable_prior,
            temperature=self.state_prior_temperature,
            output_softclip=tfb.SoftClip(low=-7., high=7.) if steady_state_net_softclipping else tfb.Identity(),
            name='latent_stationary_network')
        # latent policy
        if self.external_latent_policy is None and self.latent_policy_network is not None:
            self.latent_policy_network = LatentPolicyNetwork(
                latent_state=latent_state,
                latent_policy_network=generate_sequential_model(latent_policy_network),
                number_of_discrete_actions=self.number_of_discrete_actions, )
        else:
            self.latent_policy_network = None
        # reward function
        self.reward_network = RewardNetwork(
            latent_state=latent_state,
            latent_action=latent_action,
            next_latent_state=next_latent_state,
            reward_network=generate_sequential_model(reward_network._replace(batch_norm=use_batch_norm)),
            reward_shape=self.reward_shape)
        # state reconstruction function
        # post-processing
        if self._cost_weights['state'] != 0. and (
                state_decoder_post_processing_network is not None or state_pre_proc_nets is not None
        ):
            state_post_proc_nets = []
            nets = state_decoder_post_processing_network
            if nets is None:
                if state_encoder_pre_processing_network is not None:
                    nets = state_encoder_pre_processing_network
                    if isinstance(nets, ModelArchitecture):
                        nets = [state_encoder_pre_processing_network]
                else:
                    nets = []
            for i, net in enumerate(nets):
                _state_shape = state_shape if n_state_components == 1 else state_shape[i]
                if net is not None:
                    state_post_proc_nets.append(
                        get_model(
                            model_arch=net._replace(
                                hidden_units=None if net.is_cnn else net.hidden_units,
                                input_dim=_state_shape if state_decoder_post_processing_network is None else None,
                                output_dim=((self.latent_state_size - self.atomic_prop_dims,)
                                            if not net.is_cnn else None),
                                name=f'state_post_proc_net_{i:d}',
                            ),
                            invert=state_decoder_post_processing_network is None,
                            as_model=True,
                        ))
                else:
                    state_post_proc_nets.append(None)
            if n_state_components == 1:
                state_post_proc_nets = state_post_proc_nets[0]
        else:
            state_post_proc_nets = None

        if self._cost_weights['state'] != 0.:
            self.reconstruction_network = StateReconstructionNetwork(
                latent_state=next_latent_state,
                decoder_network=generate_sequential_model(decoder_network._replace(batch_norm=use_batch_norm)),
                state_shape=self.state_shape,
                time_stacked_states=self.time_stacked_states,
                time_stacked_lstm_units=self.time_stacked_lstm_units,
                post_processing_net=state_post_proc_nets)
        else:
            self.reconstruction_network = None
        # action reconstruction function
        if self.action_discretizer and not self.policy_based_decoding:
            self.action_reconstruction_network = ActionReconstructionNetwork(
                latent_state=latent_state,
                latent_action=latent_action,
                action_decoder_network=generate_sequential_model(action_decoder_network),
                action_shape=self.action_shape)
        else:
            self.action_reconstruction_network = None
        # steady state Lipschitz function
        self.steady_state_lipschitz_network = SteadyStateLipschitzFunction(
            latent_state=latent_state,
            latent_action=latent_action if not self.policy_based_decoding else None,
            next_latent_state=next_latent_state,
            steady_state_lipschitz_network=generate_sequential_model(
                steady_state_lipschitz_network._replace(
                    activation={
                        'elu': 'smooth_elu',
                    }.get(
                        steady_state_lipschitz_network.activation,
                        steady_state_lipschitz_network.activation))),
            output_softclip=(
                softclip_fn
                if self.use_total_variation and self._dTV_clip_method != 'penalty'
                else tfb.Identity())
        )
        # transition loss Lipschitz function
        self.transition_loss_lipschitz_network = TransitionLossLipschitzFunction(
            state=state,
            action=action,
            latent_state=latent_state,
            latent_action=latent_action if self.action_discretizer else None,
            next_latent_state=next_latent_state,
            transition_loss_lipschitz_network=generate_sequential_model(
                transition_loss_lipschitz_network._replace(
                    activation={
                        'elu': 'smooth_elu',
                    }.get(
                        transition_loss_lipschitz_network.activation,
                        transition_loss_lipschitz_network.activation))),
            pre_proc_net=_get_pre_processing_net(state_encoder_pre_processing_network, 'transition_lipschitz'),
            output_softclip=(
                softclip_fn
                if self.use_total_variation and self._dTV_clip_method != 'penalty'
                else tfb.Identity())
        )

        if debug or summary:
            self.state_encoder_network.summary()
            if self.action_discretizer and not self.policy_based_decoding:
                self.action_encoder_network.summary()
            else:
                print("No action encoder")
            self.transition_network.summary()
            self.latent_stationary_network.summary()
            if self.latent_policy_network is not None:
                self.latent_policy_network.summary()
            self.reward_network.summary()
            if self.reconstruction_network is not None:
                self.reconstruction_network.summary()
            if self.action_discretizer:
                self.action_reconstruction_network.summary()
            self.steady_state_lipschitz_network.summary()
            self.transition_loss_lipschitz_network.summary()

        self.encoder_network = self.state_encoder_network

        self._base_architecture = BaseModelArchitecture(
            state_encoder_network=state_encoder_network,
            action_decoder_network=action_decoder_network,
            reward_network=reward_network,
            decoder_network=decoder_network,
            latent_policy_network=latent_policy_network,
            steady_state_lipschitz_network=steady_state_lipschitz_network,
            transition_loss_lipschitz_network=transition_loss_lipschitz_network,
            action_encoder_network=action_encoder_network,
            state_encoder_pre_processing_network=state_encoder_pre_processing_network,
            state_decoder_pre_processing_network=state_decoder_post_processing_network)

        self.loss_metrics = {
            'reconstruction_loss': Mean(name='reconstruction_loss'),
            'action_mse': MeanSquaredError(name='action_mse'),
            'reward_mse': MeanSquaredError(name='reward_loss'),
            'transition_loss': Mean('transition_loss'),
            'latent_policy_entropy': Mean('latent_policy_entropy'),
            'steady_state_regularizer': Mean('steady_state_wasserstein_regularizer'),
            'gradient_penalty': Mean('gradient_penalty'),
            'marginal_state_encoder_entropy': Mean('marginal_state_encoder_entropy'),
            'gradients_max': Mean('gradients_max'),
            'gradients_min': Mean('gradients_min'),
        }
        if self.reconstruct_state:
            if n_states == 1:
                self.loss_metrics['state_mse'] = MeanSquaredError(name='state_mse')
            else:
                names = [_input.name for _input in self.state_encoder_network.inputs]
                for i in range(n_states):
                    self.loss_metrics[f'{names[i]}_mse'] = MeanSquaredError(name=f'state_{i:d}_mse')
        if self.include_state_encoder_entropy or self.include_action_encoder_entropy:
            self.loss_metrics['entropy_regularizer'] = Mean('entropy_regularizer')
        if state_encoder_type is not EncodingType.DETERMINISTIC:
            self.loss_metrics.update({
                'binary_encoding_log_probs': Mean('binary_encoding_log_probs'),
                'state_encoder_entropy': Mean('state_encoder_entropy'),
            })
        if self.policy_based_decoding:
            self.loss_metrics['marginal_variance'] = Mean(name='marginal_variance')
        elif self.action_discretizer:
            self.loss_metrics.update({
                'marginal_action_encoder_entropy': Mean('marginal_action_encoder_entropy'),
                'action_encoder_entropy': Mean('action_encoder_entropy'),
            })
            self.temperature_metrics.update({
                't_1_action': self.action_encoder_temperature,
                't_2_action': self.latent_policy_temperature,
            })

        self._score = Mean("wae_score")
        self._last_score = None

    @property
    def evaluation_window(self):
        return tf.expand_dims(self._score.result(), 0)

    def anneal(self):
        super().anneal()
        for var, decay_rate in [
            (self._action_encoder_temperature, self.encoder_temperature_decay_rate),
            (self._latent_policy_temperature, self.prior_temperature_decay_rate),
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

    def attach_optimizer(
            self,
            optimizers: Optional[Union[Tuple[tf.optimizers.Optimizer, ...], List[tf.optimizers.Optimizer]]] = None,
            minimizer: Optional[tf.optimizers.Optimizer] = None,
            maximizer: Optional[tf.optimizers.Optimizer] = None,
            encoder_optimizer: Optional[tf.optimizers.Optimizer] = None,
    ):
        assert optimizers is not None or (
                minimizer is not None and maximizer is not None)
        if optimizers is not None:
            assert 2 <= len(optimizers) <= 3
            minimizer, maximizer = optimizers[:2]
            if len(optimizers) == 3 and encoder_optimizer is None:
                encoder_optimizer = optimizers[-1]
        self._minimizer = minimizer
        self._maximizer = maximizer
        self._encoder_optimizer = encoder_optimizer

    def detach_optimizer(self):
        minimizer = self._minimizer
        maximizer = self._maximizer
        encoder_optimizer = self._encoder_optimizer
        optimizers = [minimizer, maximizer]
        self._minimizer = None
        self._maximizer = None
        self._encoder_optimizer = None
        if encoder_optimizer is not None:
            optimizers.append(encoder_optimizer)
        return tuple(optimizers)

    def binary_encode_state(self, state: Float, label: Optional[Float] = None) -> tfd.Distribution:
        return self.state_encoder_network.discrete_distribution(
            state=state, label=label)

    def relaxed_state_encoding(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            training: bool = False,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.state_encoder_network.relaxed_distribution(
            state=state, temperature=temperature, label=label, training=training)

    def discrete_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_encoder_network.discrete_distribution(
                latent_state=latent_state, action=action)
        else:
            return tfd.Deterministic(loc=action)

    def relaxed_action_encoding(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
            temperature
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_encoder_network.relaxed_distribution(
                latent_state=latent_state, action=action, temperature=temperature)
        else:
            return tfd.Deterministic(loc=action)

    def decode_state(self, latent_state: tf.Tensor, training: bool = False) -> tfd.Distribution:
        if self.reconstruct_state:
            return self.reconstruction_network.distribution(latent_state=latent_state, training=training)
        else:
            return tfd.Deterministic(loc=latent_state)

    def decode_action(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
            *args, **kwargs
    ) -> tfd.Distribution:
        if self.action_discretizer:
            return self.action_reconstruction_network.distribution(
                latent_state=latent_state, latent_action=latent_action)
        else:
            return tfd.Deterministic(loc=latent_action)

    def action_generator(
            self,
            latent_state: Float
    ) -> tfd.Distribution:
        if self.action_discretizer:
            batch_size = tf.shape(latent_state)[0]
            loc = self.action_reconstruction_network([
                tf.repeat(latent_state, self.number_of_discrete_actions, axis=0),
                tf.tile(tf.eye(self.number_of_discrete_actions), [batch_size, 1])
            ])
            loc = tf.reshape(
                loc,
                tf.concat([[batch_size], [self.number_of_discrete_actions], self.action_shape], axis=-1))
            return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    logits=self.discrete_latent_policy(latent_state).logits_parameter()),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=loc,
                    scale_diag=tf.ones(tf.shape(loc)) * 1e-6))
        else:
            return self.discrete_latent_policy(latent_state)

    def relaxed_latent_transition(
            self,
            latent_state: Float,
            latent_action: Float,
            temperature: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.transition_network.relaxed_distribution(
            conditional_input=tf.concat([latent_state, latent_action], axis=-1))

    def discrete_latent_transition(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, *args, **kwargs
    ) -> tfd.Distribution:
        return self.transition_network.discrete_distribution(
            conditional_input=tf.concat([latent_state, latent_action], axis=-1))

    def relaxed_markov_chain_latent_transition(
            self, latent_state: tf.Tensor, temperature: float = 1e-5, reparamaterize: bool = True
    ) -> tfd.Distribution:
        return NotImplemented

    def discrete_markov_chain_latent_transition(
            self, latent_state: tf.Tensor
    ) -> tfd.Distribution:
        return NotImplemented

    def relaxed_latent_policy(
            self,
            latent_state: tf.Tensor,
            temperature: Float = 1e-5,
    ) -> tfd.Distribution:
        if self.external_latent_policy is not None:
            return self.external_latent_policy.distribution(
                ts.TimeStep(
                    observation=latent_state,
                    reward=0.,
                    discount=1.,
                    step_type=ts.StepType.MID
                )
            ).action
        else:
            return self.latent_policy_network.relaxed_distribution(
                latent_state=latent_state, temperature=temperature)

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        if self.external_latent_policy is not None:
            return self.external_latent_policy.distribution(
                ts.TimeStep(
                    observation=latent_state,
                    reward=0.,
                    discount=1.,
                    step_type=ts.StepType.MID
                )
            ).action
        else:
            return self.latent_policy_network.discrete_distribution(latent_state=latent_state)

    def reward_distribution(
            self,
            latent_state: Float,
            latent_action: Float,
            next_latent_state: Float,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self.reward_network.distribution(
            latent_state=latent_state,
            latent_action=latent_action,
            next_latent_state=next_latent_state)

    def markov_chain_reward_distribution(
            self,
            latent_state: Float,
            next_latent_state: Float,
    ) -> tfd.Distribution:
        batch_size = tf.shape(latent_state)[0]
        loc = self.reward_network([
            tf.repeat(latent_state, self.number_of_discrete_actions, axis=0),
            tf.tile(tf.eye(self.number_of_discrete_actions), [batch_size, 1]),
            tf.repeat(next_latent_state, self.number_of_discrete_actions, axis=0),
        ])
        loc = tf.reshape(loc, tf.concat([[batch_size], [self.number_of_discrete_actions], self.reward_shape], axis=-1))
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=self.discrete_latent_policy(latent_state).logits_parameter()),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=loc,
                scale_diag=tf.ones(tf.shape(loc)) * 1e-6))

    def discrete_latent_steady_state_distribution(
            self,
            batch_size: Optional[int] = None,
            *args, **kwargs) -> tfd.Distribution:
        if batch_size is None:
            return self.latent_stationary_network.discrete_distribution(*args, **kwargs)
        else:
            return tfd.BatchBroadcast(
                self.latent_stationary_network.discrete_distribution(*args, **kwargs),
                with_shape=[batch_size])

    def relaxed_latent_steady_state_distribution(
            self,
            batch_size: Optional[int] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        if batch_size is None:
            return self.latent_stationary_network.relaxed_distribution(*args, **kwargs)
        else:
            return tfd.BatchBroadcast(
                self.latent_stationary_network.relaxed_distribution(*args, **kwargs),
                with_shape=[batch_size])

    def action_embedding_function(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tf.Tensor:

        if self.action_discretizer:
            decoder = self.decode_action(
                latent_state=tf.cast(latent_state, dtype=tf.float32),
                latent_action=tf.cast(
                    tf.one_hot(
                        latent_action,
                        depth=self.number_of_discrete_actions),
                    dtype=tf.float32), )
            if self.deterministic_state_embedding:
                return decoder.mode()
            else:
                return decoder.sample()
        else:
            return latent_action

    @staticmethod
    @tf.function
    def norm(x: Float, axis: int = -1):
        """
        to replace tf.norm(x, order=2, axis) which has numerical instabilities (the derivative can yields NaN).
        """
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + epsilon)

    def cost(self, x: Float, y: Float, space: str):

        flatten = lambda _x: tf.concat(
            [tf.reshape(_component, [tf.shape(_component)[0], -1]) for _component in tf.nest.flatten(_x)],
            axis=-1)

        if space == 'state' and not self.reconstruct_state:
            return tf.zeros(tf.shape(flatten(x))[0])

        l1 = lambda _x, _y: tf.reduce_sum(tf.math.abs(_x - _y), axis=1)
        l2 = lambda _x, _y: self.norm(_x - _y)
        l22 = lambda _x, _y: tf.reduce_sum(tf.square(_x - _y), axis=1)
        cosine_similarity = lambda _x, _y: tf.reduce_sum(
            tf.math.l2_normalize(_x, axis=1) * tf.math.l2_normalize(_y, axis=1), axis=1)
        cosine_distance = lambda _x, _y: 1 - cosine_similarity(_x, _y)
        angular_distance = lambda _x, _y: tf.math.acos(cosine_similarity(_x, _y)) / np.pi
        binary_cross_entropy = lambda _x, _y: tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=_y,
                labels=_x, ),
            axis=-1)
        costs = {
            'l1': l1,
            'l2': l2,
            'l22': l22,
            'cosine_distance': cosine_distance,
            'angular_distance': angular_distance,
            'binary_cross_entropy': binary_cross_entropy,
        }
        try:
            len(self._cost_fn[space])
            return sum([costs[c](flatten(x_i), flatten(y_i)) * w for x_i, y_i, c, w in zip(
                x, y, self._cost_fn[space], self._cost_weights[space])])
        except:
            return costs[self._cost_fn[space]](x, y) * self._cost_weights[space]

    @tf.function
    def __call__(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_key: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            training: bool = False,
            optimization_direction: Optional['str'] = None,
            *args, **kwargs
    ):
        # handle state space with multiple components
        def flatten(_x):
            return tf.concat(
                [tf.reshape(_component, [tf.shape(_component)[0], -1]) for _component in tf.nest.flatten(_x)],
                axis=-1)

        batch_size = tf.shape(action)[0]
        # encoder sampling
        relaxed_latent_state = self.relaxed_state_encoding(
            state,
            label=label,
            temperature=self.state_encoder_temperature,
            training=training,
        ).sample()
        if self.straight_through:
            latent_state = relaxed_latent_state + tf.round(relaxed_latent_state) - \
                           tf.stop_gradient(relaxed_latent_state)
        else:
            latent_state = relaxed_latent_state

        relaxed_next_latent_state = self.relaxed_state_encoding(
            next_state,
            label=next_label,
            temperature=self.state_encoder_temperature,
            training=training,
        ).sample()
        if self.straight_through:
            next_latent_state = relaxed_next_latent_state + tf.round(relaxed_next_latent_state) - \
                                tf.stop_gradient(relaxed_next_latent_state)
        else:
            next_latent_state = relaxed_next_latent_state

        if self.policy_based_decoding:
            latent_action = self.relaxed_latent_policy(
                latent_state,
                temperature=self.latent_policy_temperature
            ).sample()
        else:
            relaxed_latent_action = self.relaxed_action_encoding(
                latent_state,
                action,
                temperature=self.action_encoder_temperature
            ).sample()  # note that latent_action = action when self.action_discretizer is False
        if self.straight_through:
            latent_action = relaxed_latent_action + tf.round(relaxed_latent_action) - \
                            tf.stop_gradient(relaxed_latent_action)
        else:
            latent_action = relaxed_latent_action

        (relaxed_stationary_latent_state,
         relaxed_stationary_latent_action,
         relaxed_next_stationary_latent_state) = tfd.JointDistributionSequential([
            self.relaxed_latent_steady_state_distribution(batch_size=batch_size),
            lambda _latent_state: self.relaxed_latent_policy(
                latent_state=_latent_state,
                temperature=self.latent_policy_temperature),
            lambda _latent_action, _latent_state: self.relaxed_latent_transition(
                _latent_state,
                _latent_action, ),
        ]).sample()
        if self.straight_through:
            next_stationary_latent_state = relaxed_next_stationary_latent_state + \
                                           tf.round(relaxed_next_stationary_latent_state) - \
                                           tf.stop_gradient(relaxed_next_stationary_latent_state)
        else:
            next_stationary_latent_state = relaxed_next_stationary_latent_state

        # next latent state from the latent transition function
        relaxed_next_transition_latent_state = self.relaxed_latent_transition(
            latent_state,
            latent_action,
        ).sample()
        if self.straight_through:
            next_transition_latent_state = relaxed_next_transition_latent_state + \
                                           tf.round(relaxed_next_transition_latent_state) - \
                                           tf.stop_gradient(relaxed_next_transition_latent_state)
        else:
            next_transition_latent_state = relaxed_next_transition_latent_state

        if optimization_direction is None or optimization_direction == 'min':
            # reconstruction loss
            # the reward as well as the state and action reconstruction functions are deterministic
            if not self.policy_based_decoding or self.enforce_upper_bound:
                _state, _action, _reward, _next_state = tfd.JointDistributionSequential([
                    self.decode_state(latent_state, training=training),
                    self.decode_action(
                        latent_state,
                        latent_action),
                    self.reward_distribution(
                        latent_state,
                        latent_action,
                        next_latent_state,
                        training=training),
                    self.decode_state(next_latent_state, training=training)
                ]).sample()
            else:
                mean_decoder_fn = tfd.JointDistributionSequential([
                    self.decode_state(latent_state, training=training),
                    self.action_generator(latent_state),
                    self.markov_chain_reward_distribution(latent_state, next_latent_state),
                    self.decode_state(next_latent_state, training=training)
                ]).mean
                _state, _action, _reward, _next_state = mean_decoder_fn()

            reconstruction_loss = (
                    self.cost(state, _state, 'state') +
                    self.cost(action, _action, 'action') +
                    self.cost(reward, _reward, 'reward') +
                    self.cost(next_state, _next_state, 'state')
            )
            if self.squared_wasserstein or self.policy_based_decoding:
                reconstruction_loss = tf.square(reconstruction_loss)

            if self.policy_based_decoding:
                # marginal variance of the reconstruction
                if self.enforce_upper_bound:
                    random_action, random_reward = _action, _reward
                    _, _action, _reward, _ = mean_decoder_fn()
                else:
                    random_action, random_reward = tfd.JointDistributionSequential([
                        self.decode_action(latent_state, latent_action),
                        self.reward_distribution(latent_state, latent_action, next_latent_state),
                    ]).sample()
                _flat_state = flatten(_state)
                _flat_next_state = flatten(_next_state)
                y = tf.concat([_flat_state, random_action, random_reward, _flat_next_state], axis=-1)
                mean = tf.concat([_flat_state, _action, _reward, _flat_next_state], axis=-1)
                marginal_variance = (self.norm(y - mean, axis=1) ** 2. +
                                     self.norm(mean - tf.reduce_mean(mean), axis=1) ** 2)

            else:
                random_action = _action
                random_reward = _reward
                marginal_variance = 0.
        else:
            reconstruction_loss = 0.
            marginal_variance = 0.

        # Wasserstein regularizers and Lipschitz constraint
        if self.policy_based_decoding:
            x = [relaxed_latent_state, relaxed_next_transition_latent_state]
            y = [relaxed_stationary_latent_state, relaxed_next_stationary_latent_state]
        else:
            x = [relaxed_latent_state, relaxed_latent_action, relaxed_next_transition_latent_state]
            y = [
                relaxed_stationary_latent_state, relaxed_stationary_latent_action, relaxed_next_stationary_latent_state
            ]
            if self.use_total_variation:
                for input_ in [x, y]:
                    for i in range(len(input_)):
                        input_[i] += tf.round(input_[i]) - tf.stop_gradient(input_[i])
        scale = .5 if self.use_total_variation else 1.
        f_x = tf.squeeze(self.steady_state_lipschitz_network(x))
        f_y = tf.squeeze(self.steady_state_lipschitz_network(y))
        steady_state_regularizer = scale * (f_x - f_y)
        if not self.use_total_variation and (optimization_direction is None or optimization_direction == "max"):
            steady_state_gradient_penalty = self.compute_gradient_penalty(
                x=tf.concat(x, axis=-1),
                y=tf.concat(y, axis=-1),
                lipschitz_function=lambda _x: self.steady_state_lipschitz_network(
                    [_x[:, :self.latent_state_size, ...]] +
                    (
                        [_x[:, self.latent_state_size: self.latent_state_size + self.number_of_discrete_actions, ...]]
                        if not self.policy_based_decoding else
                        []
                    ) +
                    [_x[:, -self.latent_state_size:, ...]]))
        elif self.use_total_variation and self._dTV_clip_method == 'penalty':
            f_x_penalty = tf.where(tf.abs(f_x) >= 1., tf.abs(f_x) - 1., tf.zeros_like(f_x))
            f_y_penalty = tf.where(tf.abs(f_y) >= 1., tf.abs(f_y) - 1., tf.zeros_like(f_y))
            steady_state_gradient_penalty = tf.square(f_x_penalty) + tf.square(f_y_penalty)
        else:
            steady_state_gradient_penalty = 0.

        if self.action_discretizer:
            x = [state, action, relaxed_latent_state, relaxed_latent_action, relaxed_next_latent_state]
            y = [state, action, relaxed_latent_state, relaxed_latent_action, relaxed_next_transition_latent_state]
        else:
            x = [state, action, relaxed_latent_state, relaxed_next_latent_state]
            y = [state, action, relaxed_latent_state, relaxed_next_transition_latent_state]
        if self.use_total_variation:
            for input_ in [x, y]:
                for i in range(2, len(input_)):
                    input_[i] += tf.round(input_[i]) - tf.stop_gradient(input_[i])
        f_x = tf.squeeze(self.transition_loss_lipschitz_network(x))
        f_y = tf.squeeze(self.transition_loss_lipschitz_network(y))
        transition_loss_regularizer = scale * (f_x - f_y)
        if not self.use_total_variation and (optimization_direction is None or optimization_direction == "max"):
            transition_loss_gradient_penalty = self.compute_gradient_penalty(
                x=next_latent_state,
                y=next_transition_latent_state,
                lipschitz_function=lambda _x: self.transition_loss_lipschitz_network(x[:-1] + [_x]))
        elif self.use_total_variation and self._dTV_clip_method == 'penalty':
            f_x_penalty = tf.where(tf.abs(f_x) >= 1., tf.abs(f_x) - 1., tf.zeros_like(f_x))
            f_y_penalty = tf.where(tf.abs(f_y) >= 1., tf.abs(f_y) - 1., tf.zeros_like(f_y))
            transition_loss_gradient_penalty = tf.square(f_x_penalty) + tf.square(f_y_penalty)
        else:
            transition_loss_gradient_penalty = 0.

        logits = self.state_encoder_network.get_logits(state, latent_state)
        if not self.include_state_encoder_entropy and not self.include_action_encoder_entropy:
            entropy_regularizer = 0.
        else:
            entropy_regularizer = self.entropy_regularizer(
                state=state,
                latent_state=latent_state,
                logits=logits,
                action=action if not self.policy_based_decoding else None,
                include_state_entropy=self.include_state_encoder_entropy,
                include_action_entropy=self.include_action_encoder_entropy,
                sample_probability=sample_probability, )

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(reconstruction_loss +
                                          marginal_variance)))

        # loss metrics
        if optimization_direction is None or optimization_direction == "min":
            self.loss_metrics['reconstruction_loss'](reconstruction_loss)
            if 'state_mse' in self.loss_metrics:
                fn = tf.nn.sigmoid if self._cost_fn['state'] == 'binary_cross_entropy' else tfb.Identity()
                self.loss_metrics['state_mse'](state, fn(_state))
                self.loss_metrics['state_mse'](next_state, fn(_next_state))
            elif self.reconstruct_state:
                names = [_input.name for _input in self.state_encoder_network.inputs]
                for x, y in [(state, _state), (next_state, _next_state)]:
                    for i in range(len(x)):
                        fn = tf.nn.sigmoid if self._cost_fn['state'][i] == 'binary_cross_entropy' else tfb.Identity()
                        self.loss_metrics[f'{names[i]}_mse'](x[i], fn(y[i]))
            self.loss_metrics['action_mse'](action, random_action)
            self.loss_metrics['reward_mse'](reward, random_reward)
            self.loss_metrics['marginal_state_encoder_entropy'](
                self.marginal_state_encoder_entropy(logits=logits, sample_probability=sample_probability))
            if self._state_encoder_type is not EncodingType.DETERMINISTIC:
                self.loss_metrics['state_encoder_entropy'](
                    tfd.Independent(
                        tfd.Bernoulli(logits=logits),
                        reinterpreted_batch_ndims=1
                    ).entropy())
            self.loss_metrics['latent_policy_entropy'](
                self.discrete_latent_policy(latent_state).entropy())
            if self._state_encoder_type is not EncodingType.DETERMINISTIC:
                self.loss_metrics['binary_encoding_log_probs'](
                    self.binary_encode_state(
                        state=state
                    ).log_prob(tf.round(latent_state)[..., self.atomic_prop_dims:]))
            if self.action_discretizer and not self.policy_based_decoding:
                self.loss_metrics['marginal_action_encoder_entropy'](
                    self.marginal_action_encoder_entropy(latent_state, action))
                self.loss_metrics['action_encoder_entropy'](
                    self.discrete_action_encoding(latent_state, action).entropy())
            elif self.policy_based_decoding:
                self.loss_metrics['marginal_variance'](marginal_variance)
            if self.include_state_encoder_entropy or self.include_action_encoder_entropy:
                self.loss_metrics['entropy_regularizer'](entropy_regularizer)
            self.loss_metrics['transition_loss'](transition_loss_regularizer)
            self.loss_metrics['steady_state_regularizer'](steady_state_regularizer)

        if optimization_direction is None or optimization_direction == "max":
            self.loss_metrics['gradient_penalty'](
                steady_state_gradient_penalty + transition_loss_gradient_penalty)

        # dynamic reward scaling
        self._dynamic_reward_scaling.assign(
            tf.math.minimum(
                self._dynamic_reward_scaling,
                tf.pow(2. * tf.reduce_max(tf.abs(reward)), -1.)))

        if debug:
            tf.print("latent_state", latent_state, summarize=-1)
            tf.print("next_latent_state", next_latent_state, summarize=-1)
            tf.print("next_stationary_latent_state", next_stationary_latent_state, summarize=-1)
            tf.print("next_transition_latent_state", next_transition_latent_state, summarize=-1)
            tf.print("latent_action", latent_action, summarize=-1)
            tf.print("loss", tf.stop_gradient(
                reconstruction_loss + marginal_variance +
                self.wasserstein_regularizer_scale_factor.stationary.scaling * steady_state_regularizer +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling * transition_loss_regularizer))

        return {
            'reconstruction_loss': reconstruction_loss + marginal_variance,
            'steady_state_regularizer': steady_state_regularizer,
            'steady_state_gradient_penalty': steady_state_gradient_penalty,
            'transition_loss_regularizer': transition_loss_regularizer,
            'transition_loss_gradient_penalty': transition_loss_gradient_penalty,
            'entropy_regularizer': entropy_regularizer,
        }

    def marginal_state_encoder_entropy(
            self,
            state: Optional[Float] = None,
            latent_state: Optional[Float] = None,
            logits: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
    ) -> Float:

        if logits is None:
            if state is None or latent_state is None:
                raise ValueError("A state and its encoding (i.e., as a latent state) "
                                 "should be provided when logits are not.")

            logits = self.state_encoder_network.get_logits(state, latent_state)

        if sample_probability is None:
            regularizer = tf.reduce_mean(
                - tf.sigmoid(logits) * tf.math.log(tf.reduce_mean(tf.sigmoid(logits), axis=0) + epsilon)
                - tf.sigmoid(-logits) * tf.math.log(tf.reduce_mean(tf.sigmoid(-logits), axis=0) + epsilon),
                axis=0)
        else:
            is_weights = (tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability) ** self.is_exponent
            regularizer = tf.reduce_mean(
                - tf.sigmoid(logits) * tf.math.log(
                    tf.reduce_mean(tf.expand_dims(is_weights, -1) * tf.sigmoid(logits), axis=0) + epsilon)
                - tf.sigmoid(-logits) * tf.math.log(
                    tf.reduce_mean(tf.expand_dims(is_weights, -1) * tf.sigmoid(-logits), axis=0) + epsilon),
                axis=0)
        return tf.reduce_sum(regularizer)

    def marginal_action_encoder_entropy(
            self,
            latent_state: Optional[Float] = None,
            action: Optional[Float] = None,
            logits: Optional[Float] = None,
    ) -> Float:
        if logits is None and (latent_state is None or action is None):
            raise ValueError("You should either provide the logits of the action distribution or a latent state"
                             " and an action to compute the marginal entropy")
        if logits is None:
            logits = self.discrete_action_encoding(latent_state, action).logits_parameter()
        batch_size = tf.cast(tf.shape(logits)[0], tf.float32)
        return -1. * tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.softmax(logits) * (
                        tf.reduce_logsumexp(
                            logits - tf.expand_dims(
                                tf.reduce_logsumexp(logits, axis=-1),
                                axis=-1),
                            axis=0) - tf.math.log(batch_size)),
                axis=-1),
            axis=0)

    @tf.function
    def entropy_regularizer(
            self,
            state: tf.Tensor,
            label: Optional[Float] = None,
            latent_state: Optional[Float] = None,
            logits: Optional[Float] = None,
            action: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            include_state_entropy: bool = True,
            include_action_entropy: bool = True,
            *args, **kwargs
    ) -> Float:
        if latent_state is None:
            if label is None:
                raise ValueError("either a latent state or a label should be provided")
            else:
                latent_state = self.relaxed_state_encoding(
                    state, label=label, temperature=self.state_encoder_temperature)

        regularizer = 0.

        if include_state_entropy:
            if logits is None:
                logits = self.state_encoder_network.get_logits(state, latent_state)
            regularizer += self.marginal_state_encoder_entropy(
                logits=logits,
                sample_probability=sample_probability)
            regularizer -= tfd.Independent(
                tfd.Bernoulli(logits=logits),
                reinterpreted_batch_ndims=1
            ).entropy()

        if include_action_entropy:
            if action is None or not self.action_discretizer:
                regularizer += self.action_entropy_regularizer_scaling * tf.reduce_mean(
                    self.discrete_latent_policy(latent_state).entropy(),
                    axis=0)
            else:
                logits = self.discrete_action_encoding(latent_state, action).logits_parameter()
                regularizer += self.action_entropy_regularizer_scaling * (
                        self.marginal_action_encoder_entropy(logits=logits) -
                        tf.reduce_mean(tfd.Categorical(logits=logits).entropy(), axis=0))
        return regularizer

    @tf.function
    def compute_gradient_penalty(
            self,
            x: Float,
            y: Float,
            lipschitz_function: Callable[[Float], Float],
            min_gradient_norm_epsilon: float = 1e-7,
    ):
        noise = tf.random.uniform(shape=(tf.shape(x)[0], 1), minval=0., maxval=1.)
        straight_lines = noise * x + (1. - noise) * y
        gradients = tf.gradients(lipschitz_function(straight_lines), straight_lines)[0]
        gradients_norm = self.norm(gradients, axis=1)
        gradients_norm = tf.where(gradients_norm < min_gradient_norm_epsilon, 1., gradients_norm)
        return tf.square(gradients_norm - 1.)

    def eval(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_probability: Optional[Float] = None,
            additional_transition_batch: Optional[Tuple[Float, ...]] = None,
            *args, **kwargs
    ):
        # handle state space with multiple components
        def flatten(_x):
            return tf.concat(
                [tf.reshape(_component, [tf.shape(_component)[0], -1]) for _component in tf.nest.flatten(_x)],
                axis=-1)

        flat_state = flatten(state)
        flat_next_state = flatten(next_state)

        batch_size = tf.shape(flat_state)[0]
        # sampling
        # encoders
        latent_state = self.binary_encode_state(state, label).sample()
        next_latent_state = self.binary_encode_state(next_state, next_label).sample()
        if self.policy_based_decoding:
            latent_action = tf.cast(self.discrete_latent_policy(latent_state).sample(), tf.float32)
        else:
            latent_action = tf.cast(self.discrete_action_encoding(latent_state, action).sample(), tf.float32)

        # latent steady-state distribution
        stationary_latent_state = self.discrete_latent_steady_state_distribution().sample(batch_size)
        stationary_latent_action = self.discrete_latent_policy(stationary_latent_state).sample()
        next_stationary_latent_state = self.discrete_latent_transition(
            latent_state=stationary_latent_state,
            latent_action=stationary_latent_action
        ).sample()
        next_stationary_latent_state = tf.cast(next_stationary_latent_state, tf.float32)

        # next latent state from the latent transition function
        next_transition_latent_state = self.discrete_latent_transition(
            latent_state,
            latent_action,
        ).sample()

        # reconstruction loss
        # the reward as well as the state and action reconstruction functions are deterministic
        _state, _action, _reward, _next_state = tfd.JointDistributionSequential([
            self.decode_state(latent_state),
            self.decode_action(
                latent_state,
                latent_action) if not self.policy_based_decoding else
            tfd.Deterministic(loc=self.action_generator(latent_state).mean()),
            self.reward_distribution(
                latent_state,
                latent_action,
                next_latent_state) if not self.policy_based_decoding else
            tfd.Deterministic(loc=self.markov_chain_reward_distribution(latent_state, next_latent_state).mean()),
            self.decode_state(next_latent_state)
        ]).sample()

        _flat_state = flatten(_state)
        _flat_next_state = flatten(_next_state)

        reconstruction_loss = (
                tf.norm(flat_state - _flat_state, ord=2, axis=1) +
                tf.norm(action - _action, ord=2, axis=1) +
                tf.norm(reward - _reward, ord=2, axis=1) +
                tf.norm(flat_next_state - _flat_next_state, ord=2, axis=1))
        if self.policy_based_decoding or self.squared_wasserstein:
            reconstruction_loss = tf.square(reconstruction_loss)

        # marginal variance of the reconstruction
        if self.policy_based_decoding:
            random_action, random_reward = tfd.JointDistributionSequential([
                self.decode_action(latent_state, latent_action),
                self.reward_distribution(latent_state, latent_action, next_latent_state),
            ]).sample()
            y = tf.concat([_flat_state, random_action, random_reward, _flat_next_state], axis=-1)
            mean = tf.concat([_flat_state, _action, _reward, _flat_next_state], axis=-1)
            marginal_variance = tf.reduce_sum((y - mean) ** 2. + (mean - tf.reduce_mean(mean)) ** 2., axis=-1)
        else:
            marginal_variance = 0.

        # Wasserstein regularizers and Lipschitz constraint
        if self.policy_based_decoding:
            x = [latent_state, next_transition_latent_state]
            y = [stationary_latent_state, next_stationary_latent_state]
        else:
            x = [latent_state, latent_action, next_transition_latent_state]
            y = [stationary_latent_state, stationary_latent_action, next_stationary_latent_state]
        steady_state_regularizer = tf.squeeze(
            self.steady_state_lipschitz_network(x) - self.steady_state_lipschitz_network(y))

        if self.action_discretizer:
            x = [state, action, latent_state, latent_action, next_latent_state]
            y = [state, action, latent_state, latent_action, next_transition_latent_state]
        else:
            x = [state, action, latent_state, next_latent_state]
            y = [state, action, latent_state, next_transition_latent_state]
        transition_loss_regularizer = tf.squeeze(
            self.transition_loss_lipschitz_network(x) - self.transition_loss_lipschitz_network(y))

        if debug:
            latent_policy = self.discrete_latent_policy(latent_state)
            tf.print("latent policy", latent_policy,
                     '\n latent policy: probs parameter', latent_policy.probs_parameter())
            tf.print("latent action ~ latent policy", latent_policy.sample())
            tf.print("latent_action hist:", tf.cast(tf.argmax(latent_action, axis=1), tf.int64))

        return {
            'reconstruction_loss': reconstruction_loss + marginal_variance,
            'wasserstein_regularizer':
                (self.wasserstein_regularizer_scale_factor.stationary.scaling * steady_state_regularizer +
                 self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling * transition_loss_regularizer),
            'latent_states': tf.concat([tf.cast(latent_state, tf.int64), tf.cast(next_latent_state, tf.int64)], axis=0),
            'latent_actions': (tf.cast(tf.argmax(latent_action, axis=1), tf.int64)
                               if self.action_discretizer else
                               tf.cast(tf.argmax(stationary_latent_action, axis=1), tf.int64))
        }

    @tf.function
    def compute_loss(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
            additional_transition_batch: Optional[Tuple[Float]] = None,
            training: bool = False,
            optimization_direction: Optional[str] = None,
            *args, **kwargs
    ):
        output = self(state, label, action, reward, next_state, next_label,
                      sample_key=sample_key,
                      sample_probability=sample_probability,
                      additional_transition_batch=additional_transition_batch,
                      training=training,
                      optimization_direction=optimization_direction)

        if debug:
            tf.print('call output', output, summarize=-1)

        # Importance sampling weights (is) for prioritized experience replay
        if sample_probability is not None:
            is_weights = (tf.stop_gradient(tf.reduce_min(sample_probability)) / sample_probability) ** self.is_exponent
        else:
            is_weights = 1.

        reconstruction_loss = output['reconstruction_loss']
        wasserstein_loss = (
                self.wasserstein_regularizer_scale_factor.stationary.scaling *
                output['steady_state_regularizer'] +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling *
                output['transition_loss_regularizer']
        )
        gradient_penalty = (
                self.wasserstein_regularizer_scale_factor.stationary.scaling *
                self.wasserstein_regularizer_scale_factor.stationary.gradient_penalty_multiplier *
                output['steady_state_gradient_penalty'] +
                self.wasserstein_regularizer_scale_factor.local_transition_loss.scaling *
                self.wasserstein_regularizer_scale_factor.local_transition_loss.gradient_penalty_multiplier *
                output['transition_loss_gradient_penalty']
        )

        if self.include_state_encoder_entropy:
            entropy_regularizer = self.entropy_regularizer_scale_factor * output['entropy_regularizer']
        elif self.include_action_encoder_entropy:
            entropy_regularizer = output['entropy_regularizer']
        else:
            entropy_regularizer = 0.

        loss = lambda minimize: tf.reduce_mean(
            (-1.) ** (1. - minimize) * is_weights * (
                    minimize * reconstruction_loss +
                    wasserstein_loss +
                    (minimize - 1.) * gradient_penalty -
                    minimize * entropy_regularizer
            )
        )

        if optimization_direction is not None:
            return {optimization_direction: loss({'min': 1., 'max': 0.}[optimization_direction])}
        else:
            return {'min': loss(1.), 'max': loss(0.)}

    @property
    def state_encoder_temperature(self):
        return self.encoder_temperature

    @property
    def reconstruct_state(self) -> bool:
        """
        Returns: whether the state reconstruction loss is computed or not.
        """
        return self._cost_weights['state'] != 0.

    @property
    def state_prior_temperature(self):
        return self.prior_temperature

    @property
    def action_encoder_temperature(self):
        return self._action_encoder_temperature

    @action_encoder_temperature.setter
    def action_encoder_temperature(self, value):
        self._action_encoder_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='action_encoder_temperature')

    @property
    def latent_policy_temperature(self):
        return self._latent_policy_temperature

    @latent_policy_temperature.setter
    def latent_policy_temperature(self, value):
        self._latent_policy_temperature = tf.Variable(
            value, dtype=tf.float32, trainable=False, name='latent_policy_temperature')

    @property
    def inference_variables(self):
        if self.action_discretizer and not self.policy_based_decoding:
            return self.state_encoder_network.trainable_variables + self.action_encoder_network.trainable_variables
        else:
            return self.state_encoder_network.trainable_variables

    @property
    def generator_variables(self):
        variables = self.latent_stationary_network.trainable_variables
        if self.action_discretizer:
            variables += self.action_reconstruction_network.trainable_variables
        for network in [
                           self.transition_network,
                           self.reward_network,
                       ] + (
                               [] if self.external_latent_policy is not None else
                               [self.latent_policy_network]
                       ):
            variables += network.trainable_variables
        if self.reconstruct_state:
            variables += self.reconstruction_network.trainable_variables
        return variables

    @property
    def wasserstein_variables(self):
        return (self.steady_state_lipschitz_network.trainable_variables +
                self.transition_loss_lipschitz_network.trainable_variables)

    def _compute_apply_gradients(
            self, state, label, action, reward, next_state, next_label,
            sample_key=None, sample_probability=None,
            step: Int = None,
            encoder_update: bool = True,
            decoder_update: bool = True,
            regularizer_update: bool = True,
            optimization_direction: Optional[str] = None,
            *args, **kwargs
    ):
        if step is None:
            step = self.n_critic

        encoder_variables = self.inference_variables if encoder_update else []
        decoder_variables = self.generator_variables if decoder_update else []
        autoencoder_variables = encoder_variables + decoder_variables
        wasserstein_regularizer_variables = self.wasserstein_variables if regularizer_update else []

        def numerical_error(x, list_of_tensors=False):
            detected = False
            if not list_of_tensors:
                x = [x]
            for value in x:
                if value is not None:
                    detected = detected or tf.reduce_any(tf.logical_or(
                        tf.math.is_nan(value),
                        tf.math.is_inf(value)))
            return detected

        with tf.GradientTape(persistent=True) as tape:
            loss = self.compute_loss(
                state, label, action, reward, next_state, next_label,
                sample_key=sample_key, sample_probability=sample_probability,
                training=True, optimization_direction=optimization_direction)
            if optimization_direction is None or optimization_direction == 'min':
                loss['min_encoder'] = loss['min']
        for opt_direction, variables in {
            'max': wasserstein_regularizer_variables,
            'min': autoencoder_variables if self._encoder_optimizer is None else decoder_variables,
            'min_encoder': None if self._encoder_optimizer is None else encoder_variables,
        }.items():
            if (
                    # optimization_direction is provided => only optimize for that direction
                    (optimization_direction is None or opt_direction == optimization_direction) and
                    # check whether the variables exist for that direction
                    variables is not None and
                    # check whether the loss has no numerical error
                    (not debug or not numerical_error(loss[opt_direction])) and
                    # opt_direction != max => only minimize the loss if step % n_critic = 0
                    (opt_direction == 'max' or (step % self.n_critic == 0 and opt_direction in ['min', 'min_encoder']))
            ):
                gradients = tape.gradient(loss[opt_direction], variables)
                optimizer = {
                    'max': self._maximizer,
                    'min': self._minimizer,
                    'min_encoder': self._encoder_optimizer,
                }[opt_direction]

                if self.clip_by_global_norm:
                    gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_by_global_norm)

                if not numerical_error(gradients, list_of_tensors=True):
                    if optimizer is not None:
                        optimizer.apply_gradients(zip(gradients, variables))
                else:
                    tf.print("[Warning] Numerical error detected in the gradients computation for variable")
                    for gradient, variable in zip(gradients, variables):
                        if numerical_error(gradient):
                            tf.print(variable.name, "values")
                            tf.print(variable, summarize=-1)
                            tf.print(variable.name, "gradients")
                            tf.print(gradient, summarize=-1)

                if 'gradients_' + opt_direction in self.loss_metrics.keys():
                    mean_abs_grads = tf.concat(
                        [tf.reshape(tf.abs(grad), [-1]) for grad in gradients],
                        axis=-1)
                    self.loss_metrics['gradients_' + opt_direction](mean_abs_grads)

                if debug_gradients:
                    for gradient, variable in zip(gradients, variables):
                        tf.print("Gradient for {} (direction={}):".format(variable.name, opt_direction),
                                 gradient)

        del tape

        if optimization_direction is not None:
            return {{
                        'min': 'loss_minimizer', 'max': 'loss_maximizer'
                    }[optimization_direction]: loss[optimization_direction]}

        return {'loss_minimizer': loss['min'], 'loss_maximizer': loss['max']}

    @tf.function
    def compute_apply_gradients(
            self,
            state: Float,
            label: Float,
            action: Float,
            reward: Float,
            next_state: Float,
            next_label: Float,
            sample_key: Optional[Float] = None,
            sample_probability: Optional[Float] = None,
            additional_transition_batch: Optional[Tuple[Float]] = None,
            step: Int = None,
            optimization_direction: Optional[str] = None
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            encoder_update=True, decoder_update=True, regularizer_update=True,
            sample_key=sample_key, sample_probability=sample_probability,
            additional_transition_batch=additional_transition_batch,
            step=step, optimization_direction=optimization_direction)

    @tf.function
    def inference_update(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            encoder_update=True, decoder_update=False, regularizer_update=True,
            sample_key=sample_key, sample_probability=sample_probability)

    @tf.function
    def generator_update(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
    ):
        return self._compute_apply_gradients(
            state, label, action, reward, next_state, next_label,
            encoder_update=False, decoder_update=True, regularizer_update=True,
            sample_key=sample_key, sample_probability=sample_probability)

    def mean_latent_bits_used(self, inputs, eps=1e-3, deterministic=True):
        state, label, action, reward, next_state, next_label = inputs[:6]
        latent_distribution = self.binary_encode_state(state, label)
        latent_state = latent_distribution.sample()
        if deterministic:
            mean = tf.reduce_mean(latent_distribution.mode(), axis=0)
        else:
            mean = tf.reduce_mean(latent_distribution.mean(), axis=0)
        check = lambda x: 1. if 1. - eps > x > eps else 0.
        mbu = {'mean_state_bits_used': tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()}
        if self.action_discretizer:
            mean = tf.reduce_mean(
                self.discrete_action_encoding(latent_state, action).probs_parameter()
                if not self.policy_based_decoding else
                self.discrete_latent_policy(latent_state).probs_parameter(),
                axis=0)
            check = lambda x: 1 if 1 - eps > x > eps else 0
            mean_bits_used = tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

            mbu.update({'mean_action_bits_used': mean_bits_used})
        return mbu

    def get_latent_policy(
            self,
            observation_dtype: tf.dtypes = tf.int32,
            action_dtype: tf.dtypes = tf.int32
    ) -> tf_policy.TFPolicy:
        if self.external_latent_policy:

            class LatentPolicy(tf_policy.TFPolicy):
                _wrapped_policy = self.external_latent_policy

                def _distribution(self, time_step, policy_state):
                    ps = self._wrapped_policy._distribution(time_step, policy_state)
                    return ps._replace(action=tfd.Categorical(logits=ps.action.logits_parameter(), dtype=action_dtype))

            return LatentPolicy(
                time_step_spec=self.external_latent_policy.time_step_spec,
                action_spec=self.external_latent_policy.action_spec)
        else:
            return super(WassersteinMarkovDecisionProcess, self).get_latent_policy(observation_dtype, action_dtype)

    def estimate_local_losses_from_samples(
            self,
            environment: tf_py_environment.TFPyEnvironment,
            steps: int,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            estimate_transition_function_from_samples: bool = False,
            assert_estimated_transition_function_distribution: bool = False,
            replay_buffer_max_frames: Optional[int] = int(1e5),
            reward_scaling: Optional[float] = 1.,
            estimate_value_difference: bool = True,
            latent_policy: Optional[tf_policy.TFPolicy] = None,
            memory_limit: int = int(4e9),
            *args, **kwargs
    ):
        if self.time_stacked_states:
            labeling_function = lambda x: labeling_function(x)[:, -1, ...]

        if latent_policy is None:
            latent_policy = self.get_latent_policy(action_dtype=tf.int64)

        return estimate_local_losses_from_samples(
            environment=environment,
            latent_policy=latent_policy,
            steps=steps,
            latent_state_size=self.latent_state_size,
            number_of_discrete_actions=self.number_of_discrete_actions,
            state_embedding_function=self.state_embedding_function,
            probabilistic_state_embedding=None if self.deterministic_state_embedding else self.binary_encode_state,
            action_embedding_function=self.action_embedding_function,
            latent_reward_function=lambda latent_state, latent_action, next_latent_state: (
                self.reward_distribution(
                    latent_state=tf.cast(latent_state, dtype=tf.float32),
                    latent_action=tf.cast(latent_action, dtype=tf.float32),
                    next_latent_state=tf.cast(next_latent_state, dtype=tf.float32),
                ).mode()),
            labeling_function=labeling_function,
            latent_transition_function=lambda _latent_state, _latent_action: self.discrete_latent_transition(
                tf.cast(_latent_state, tf.float32), tf.cast(_latent_action, tf.float32)),
            estimate_transition_function_from_samples=estimate_transition_function_from_samples,
            replay_buffer_max_frames=replay_buffer_max_frames,
            reward_scaling=reward_scaling,
            atomic_prop_dims=self.atomic_prop_dims,
            estimate_value_difference=estimate_value_difference,
            memory_limit=memory_limit, )

    def eval_and_save(
            self,
            eval_steps: int,
            global_step: tf.Variable,
            dataset: Optional = None,
            dataset_iterator: Optional = None,
            batch_size: Optional[int] = None,
            save_directory: Optional[str] = None,
            log_name: Optional[str] = None,
            train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
            eval_policy_driver: Optional[tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver] = None,
            local_losses_estimator: Optional[Callable] = None,
            *args, **kwargs
    ):

        if (dataset is None) == (dataset_iterator is None or batch_size is None):
            raise ValueError("Must either provide a dataset or a dataset iterator + batch size.")

        if dataset is not None:
            batch_size = eval_steps
            dataset_iterator = iter(dataset.batch(
                batch_size=batch_size,
                drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
            eval_steps = 1 if eval_steps > 0 else 0

        metrics = {
            'eval_loss': tf.metrics.Mean(),
            'eval_reconstruction_loss': tf.metrics.Mean(),
            'eval_wasserstein_regularizer': tf.metrics.Mean(),
        }

        data = {'states': None, 'actions': None}
        score = dict()
        local_losses_metrics = None

        if eval_steps > 0:
            eval_progressbar = Progbar(
                target=(eval_steps + 1) * batch_size, interval=0.1, stateful_metrics=['eval_ELBO'])

            tf.print("\nEvalutation over {} step(s)".format(eval_steps))

            for step in range(eval_steps):
                x = next(dataset_iterator)
                if self._sample_additional_transition:
                    x_prime = next(dataset_iterator)
                else:
                    x_prime = None

                if len(x) >= 8:
                    sample_probability = x[7]
                    # we consider is_exponent=1 for evaluation
                    is_weights = tf.reduce_min(sample_probability) / sample_probability
                else:
                    sample_probability = None
                    is_weights = 1.

                evaluation = self.eval(
                    *(x[:6]), sample_probability=sample_probability, additional_transition_batch=x_prime)
                for value in ('states', 'actions'):
                    latent = evaluation['latent_' + value]
                    data[value] = latent if data[value] is None else tf.concat([data[value], latent], axis=0)
                for value in ('loss', 'reconstruction_loss', 'wasserstein_regularizer'):
                    if value == 'loss':
                        metrics['eval_' + value](tf.reduce_mean(
                            is_weights * (evaluation['reconstruction_loss'] + evaluation['wasserstein_regularizer'])))
                    else:
                        metrics['eval_' + value](tf.reduce_mean(is_weights * evaluation[value]))
                eval_progressbar.add(batch_size, values=[('eval_loss', metrics['eval_loss'].result())])
            tf.print('\n')

        if eval_policy_driver is not None:
            score['eval_policy'] = self.eval_policy(
                eval_policy_driver=eval_policy_driver,
                train_summary_writer=train_summary_writer,
                global_step=global_step
            ).numpy()

        if local_losses_estimator is not None:
            local_losses_metrics = local_losses_estimator()

        if train_summary_writer is not None and eval_steps > 0:
            with train_summary_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(key, value.result(), step=global_step)
                for value in ('states', 'actions'):
                    if data[value] is not None:
                        if value == 'states':
                            data[value] = tf.reduce_sum(
                                data[value] * 2 ** tf.range(tf.cast(self.latent_state_size, dtype=tf.int64)),
                                axis=-1)
                        tf.summary.histogram('{}_frequency'.format(value[:-1]), data[value],
                                             step=global_step, buckets=32)
                if local_losses_metrics is not None:
                    tf.summary.scalar('local_reward_loss', local_losses_metrics.local_reward_loss, step=global_step)
                    tf.summary.scalar('local_transition_loss',
                                      local_losses_metrics.local_transition_loss, step=global_step)
                    if local_losses_metrics.local_transition_loss_transition_function_estimation is not None:
                        tf.summary.scalar('local_transition_loss_empirical_transition_function',
                                          local_losses_metrics.local_transition_loss_transition_function_estimation,
                                          step=global_step)
                    for key, value in local_losses_metrics.value_difference.items():
                        tf.summary.scalar(key, value, step=global_step)

        if local_losses_metrics is not None:
            tf.print('Local reward loss: {:.2f}'.format(local_losses_metrics.local_reward_loss))
            tf.print('Local transition loss: {:.2f}'.format(local_losses_metrics.local_transition_loss))
            tf.print('Local transition loss (empirical transition function): {:.2f}'
                     ''.format(local_losses_metrics.local_transition_loss_transition_function_estimation))
            score['local_reward_loss'] = local_losses_metrics.local_reward_loss.numpy()
            score['local_transition_loss'] = local_losses_metrics.local_transition_loss.numpy()
            if local_losses_metrics.local_transition_loss_transition_function_estimation is not None and \
                    local_losses_metrics.local_transition_loss_transition_function_estimation \
                    < local_losses_metrics.local_transition_loss:
                score['local_transition_loss'] = \
                    local_losses_metrics.local_transition_loss_transition_function_estimation.numpy()

            for key, value in local_losses_metrics.value_difference.items():
                tf.print(key, value)
            local_losses_metrics.print_time_metrics()

        if eval_steps > 0:
            print('eval loss: ', metrics['eval_loss'].result().numpy())

        if eval_policy_driver is not None:
            self.assign_score(
                score=score,
                checkpoint_model=save_directory is not None,
                save_directory=save_directory,
                model_name='model',
                training_step=global_step.numpy())

        gc.collect()

        return metrics['eval_loss'].result()

    def save(self, save_directory, model_name: str, infos: Optional[Dict] = None, *args, **kwargs):
        import os
        import json

        if infos is None:
            infos = dict()
        else:
            for key, value in infos.items():
                if type(value) in [float, np.float32, np.float64, np.float16]:
                    infos[key] = str(f'{value:.9g}')
                else:
                    infos[key] = str(value)

        save_path = os.path.join(save_directory, model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save model variables through checkpointing
        optimizer = self.detach_optimizer()
        priority_handler = self.priority_handler
        external_policy = self.external_latent_policy
        self.priority_handler = None
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint.save(os.path.join(save_path, 'ckpt'))
        self.attach_optimizer(optimizer)
        self.priority_handler = priority_handler
        self.external_latent_policy = external_policy

        # dump model infos
        with open(os.path.join(save_path, 'model_infos.json'), 'w') as file:
            json.dump({**self._params, **infos}, file)

        print('Model saved to:', save_path)

    def assign_score(
            self,
            score: Dict[str, float],
            checkpoint_model: bool,
            save_directory: str,
            model_name: str,
            training_step: int,
            save_best_only: bool = True,
    ) -> bool:
        self._score(score['eval_policy'])
        score['training_step'] = training_step
        self._last_score = score['eval_policy']
        print("assigning score:", score['eval_policy'])

        if checkpoint_model:
            print("save model...")
            import os
            if save_best_only and os.path.exists(os.path.join(save_directory, model_name, 'model_infos.json')):
                with open(os.path.join(save_directory, model_name, 'model_infos.json'), 'r') as f:
                    infos = json.load(f)
                eval_policy = float(infos.get('eval_policy', -1. * np.inf))
                local_transition_loss = float(infos.get('local_transition_loss', np.inf))
                local_reward_loss = float(infos.get('local_reward_loss', np.inf))
                print(
                    "current best model:, eval_policy={:.2f}, local_transitition_loss={:.2f}, local_reward_loss={:.2f}".format(
                        eval_policy, local_transition_loss, local_reward_loss))
                if score['eval_policy'] > eval_policy:
                    print(score['eval_policy'], "better")
                    self.save(save_directory, model_name, score)
                    return True
                elif np.abs(eval_policy - score['eval_policy']) < epsilon and (
                        'local_transition_loss' in score.keys() and 'local_reward_loss' in score.keys()):
                    if score['local_transition_loss'] < local_transition_loss:
                        print("local_transition_loss better:", score['local_transition_loss'])
                        self.save(save_directory, model_name, score)
                        return True
                    elif np.abs(score['local_transition_loss'] - local_transition_loss) < epsilon and (
                            score['local_reward_loss'] < local_reward_loss):
                        print("local_reward_loss better:", score['local_reward_loss'])
                        self.save(save_directory, model_name, score)
                        return True
            else:
                print("saving model")
                self.save(save_directory, model_name, score)
                return True

        return False


def load(model_path: str, **kwargs):
    with open(os.path.join(model_path, 'model_infos.json'), 'r') as f:
        infos = json.load(f)

    params = dict()
    for key, value in infos.items():
        try:
            params[key] = eval(value)
        except NameError:
            params[key] = value
        except SyntaxError:
            pass

    for kwarg in kwargs:
        params[kwarg] = kwargs[kwarg]

    model = WassersteinMarkovDecisionProcess(**params)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(os.path.join(model_path, 'ckpt-1'))

    return model
