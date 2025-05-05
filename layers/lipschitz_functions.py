from typing import Optional, Tuple, Union, Callable

import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp

tfb = tfp.bijectors

from util.nn import _get_elem


class SteadyStateLipschitzFunction(tfk.Model):

    def __init__(
            self,
            latent_state: tfk.Input,
            next_latent_state: tfk.Input,
            steady_state_lipschitz_network: tfk.Model,
            latent_action: Optional[tfkl.Input] = None,
            output_softclip: Optional[Callable[[tf.Tensor], tf.Tensor]] = tfb.Identity(),
    ):
        inputs = [latent_state] + ([latent_action] if latent_action is not None else []) + [next_latent_state]
        network_input = tfkl.Concatenate()(inputs)
        _steady_state_lipschitz_network = steady_state_lipschitz_network(network_input)
        _steady_state_lipschitz_network = tfkl.Dense(
            units=1,
            activation=output_softclip,
            name='steady_state_lipschitz_network_output'
        )(_steady_state_lipschitz_network)

        super(SteadyStateLipschitzFunction, self).__init__(
            inputs=inputs,
            outputs=_steady_state_lipschitz_network,
            name='steady_state_lipschitz_network')


class TransitionLossLipschitzFunction(tfk.Model):
    def __init__(
            self,
            state: Union[tfkl.Input, Tuple[tfkl.Input, ...]],
            action: tfkl.Input,
            latent_state: tfkl.Input,
            next_latent_state: tfkl.Input,
            transition_loss_lipschitz_network: tfk.Model,
            latent_action: Optional[tfkl.Input] = None,
            flatten_units: int = 32,
            pre_proc_net: Optional[Union[tfkl.Layer, tfk.Model, Callable]] = None,
            time_stacked_states: bool = False,
            time_stacked_lstm_units: int = 128,
            output_softclip: Optional[Callable[[tf.Tensor], tf.Tensor]] = tfb.Identity(),
    ):
        try:
            no_input_states = len(state)
            if no_input_states == 1:
                state = state[-1]
        except TypeError:
            no_input_states = 1

        if pre_proc_net is not None:
            pre_proc_net = tf.nest.flatten(pre_proc_net)
            assert len(pre_proc_net) == no_input_states or len(pre_proc_net) == 1, \
                "the number of pre-processing networks should be one or have the same size as the number of inputs"

        if no_input_states > 1:
            components = []
            nets = []
            for i, state_component in enumerate(state):
                net = _get_elem(pre_proc_net, i, default=None)
                if net is not None:
                    if time_stacked_states:
                        x = tfkl.TimeDistributed(_get_elem(pre_proc_net, i))(state_component)
                        x = tfkl.LSTM(units=time_stacked_lstm_units)(x)
                    else:
                        x = state_component
                    _state = net(x)
                    _state = tfkl.Concatenate()(tf.nest.flatten(_state))
                    _state = tfkl.Flatten()(_state)
                    _state = tfkl.Dense(units=flatten_units, activation='sigmoid')(_state)
                    components.append(_state)
            _state = tfkl.Concatenate()(components)
        else:
            _state = state

        inputs = [state, action, latent_state]

        if latent_action is not None:
            inputs.append(latent_action)
        inputs.append(next_latent_state)
        # combine multiple state-components into _state
        _transition_loss_lipschitz_network = tfkl.Concatenate()([_state] + inputs[1:])
        _transition_loss_lipschitz_network = transition_loss_lipschitz_network(_transition_loss_lipschitz_network)
        _transition_loss_lipschitz_network = tfkl.Dense(
            units=1,
            activation=output_softclip,
            name='transition_loss_lipschitz_network_output'
        )(_transition_loss_lipschitz_network)

        super(TransitionLossLipschitzFunction, self).__init__(
            inputs=inputs,
            outputs=_transition_loss_lipschitz_network,
            name='transition_loss_lipschitz_network')
        self.pre_proc_net = pre_proc_net
        self.transition_loss_lipschitz_network = transition_loss_lipschitz_network
