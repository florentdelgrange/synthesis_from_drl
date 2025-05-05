import enum
import time
from typing import Callable, Optional, Dict, Union

import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float, Int, Bool

from verification import binary_latent_space

prob_error = 1e-10
import sys


class Error(enum.Enum):
    ABSOLUTE = enum.auto()
    RELATIVE = enum.auto()


error = {
    'absolute': Error.ABSOLUTE,
    'relative': Error.RELATIVE,
}


def vi_tensor_size(
        latent_state_size: Int,
        num_actions: Int,
        episodic_return: bool = True,
) -> tf.int32:
    state_space_size = 2 ** latent_state_size
    return tf.float32.size * (
        state_space_size * latent_state_size + # state space
        num_actions ** 2 +  # action space
        (state_space_size if episodic_return else 0) +  # reset state test
        state_space_size * num_actions +  # policy probs
        2 * state_space_size ** 2 * num_actions +  # transitions and rewards
        state_space_size * num_actions +  # q-values
        3 * state_space_size  # values, next values, delta computation
    )

@tf.function
def value_iteration(
        latent_state_size: Int,
        num_actions: Int,
        transition_fn: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        reward_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        gamma: Float,
        policy: Optional[Callable[[tf.Tensor], tfd.OneHotCategorical]] = None,
        error_type: Union[str, Error] = Error.RELATIVE,
        epsilon: Float = 1e-6,
        is_reset_state_test_fn: Optional[Callable[[tf.Tensor], Bool]] = None,
        episodic_return: bool = True,
        debug: bool = False,
        v_init: Optional[Float] = None,
        transition_matrix: Optional[tf.Tensor] = None,
        reward_matrix: Optional[tf.Tensor] = None,
        policy_probs: Optional[tf.Tensor] = None,
        reward_n_dims = 1,
        return_q_values: bool = False,
        use_matrix_computation: bool = False,
) -> Dict[str, Float]:
    """
    Iteratively compute the value of (i.e., the expected return obtained from running an input policy from) each state up
    to a certain precision, depending on the error between two consecutive iterations.

    Args:
        latent_state_size: size of the state space in binary, i.e., number of bits used to represent the state space
        num_actions: size of the action space
        transition_fn: function mapping each state-action pair to a distribution over (binary encoded) states
        reward_fn: function mapping each transition to a tensor containing the reward
                   obtained by going through this transition.
        gamma: discount factor
        policy: function mapping each (binary encoded) state to a distribution over (one-hot) actions.
                If not provided, the action yielding the best values is chosen at each step.
        error_type: error type (absolute or relative)
        epsilon: error between two consecutive iterations
        is_reset_state_test_fn: function testing whether the input state is a reset (or null) state or not.
                                A reset state is, in this context, a state where the episode terminates.
                                If provided,
                                - Rewards obtained from transitions issued from states marked as 'True' are undiscounted
                                - Rewards obtained by transitioning from a state marked as 'True' are zero
        episodic_return: Whether to estimate the finite-horizon episodic return or infinite horizon return.
                         If True, is_reset_state_fn has to be provided. In that case, values obtained by transitioning
                         to a reset state will be ignored: this boils down to transitioning to an absorbing state with
                         zero reward.
        debug: whether to display iteration error and time metrics 
        v_init: (optional) initial values; if not provided, values are initialized with zeros
        transition_matrix: (optional) Transition probabilities in the form of a [S, A, S] tensor, where S is the number
                           of states and A is the number of actions. If provided along with reward_matrix, computations
                           are made directly via tensor operations (requires more memory, but yields faster operations).
        reward_matrix: (optional) Rewards in the form of a [S, A, S] tensor, where S is the number
                       of states and A is the number of actions. If provided along with transition_matrix, computations
                       are made directly via tensor operations (requires more memory, but yields faster operations).
    """
    if error_type not in [Error.RELATIVE, Error.ABSOLUTE]:
        error_type = error[error_type]
    num_states = 2 ** latent_state_size
    if v_init is None:
        if return_q_values:
            values = tf.zeros((num_states, num_actions), dtype=tf.float32)
        else:
            values = tf.zeros(num_states, dtype=tf.float32)

    else:
        values = v_init

    delta = tf.constant(float('inf'))
    state_space = binary_latent_space(latent_state_size, dtype=tf.float32)

    if is_reset_state_test_fn is None:
        not_reset_states = tf.ones(shape=tf.shape(state_space)[:1])
    else:
        not_reset_states = 1. - tf.cast(is_reset_state_test_fn(state_space), tf.float32)
    # take into account reward-shape
    if reward_n_dims > 1:
        not_reset_states = not_reset_states[..., None]
    if return_q_values:
        not_reset_states = not_reset_states[..., None]

    action_space = tf.one_hot(indices=tf.range(num_actions), depth=tf.cast(num_actions, tf.int32), dtype=tf.float32)

    assert (policy_probs is None and policy is None) or ((policy_probs is None) != (policy is None)), \
        "Either the policy or policy_probs should be provided."
    if policy is not None and policy_probs is None:
        policy_distribution = policy(state_space)
        if hasattr(policy_distribution, 'probs_parameter'):
            policy_probs = policy_distribution.probs_parameter()
        elif type(policy_distribution) is tfd.Deterministic:
            policy_probs = policy_distribution.mean()

    if policy_probs is not None and transition_matrix is not None and reward_matrix is not None:
        transition_matrix = tf.reduce_sum(transition_matrix * policy_probs[..., None], axis=1)
        if reward_n_dims == 1:
            reward_matrix = tf.reduce_sum(reward_matrix * policy_probs[..., None], axis=1)
        else:
            transition_matrix = transition_matrix[..., None]
            reward_matrix = tf.reduce_sum(reward_matrix * policy_probs[..., None, None], axis=1)
        is_markov_chain = True
    else:
        is_markov_chain = False

    if use_matrix_computation:
        reward_matrix = tf.reduce_sum(reward_matrix * transition_matrix, axis=-1)

    def _q_s(state: Float, values: tf.Tensor):
        _state = tf.reduce_sum(tf.cast(state, tf.int32) * 2 ** tf.range(tf.shape(state)[0]), axis=-1)
        return tf.transpose(
            tf.map_fn(
                fn=lambda action: tf.cond(
                    pred=policy_probs[_state, tf.cast(tf.argmax(action), dtype=tf.int32)] > prob_error,
                    true_fn=lambda: compute_next_q_value(
                        state=state,
                        action=action,
                        values=values,
                        transition_fn=transition_fn,
                        reward_fn=reward_fn,
                        gamma=gamma,
                        state_space=state_space,
                        is_reset_state_test_fn=is_reset_state_test_fn,
                        episodic_return=episodic_return),
                    false_fn=lambda: 0.),
                elems=action_space,
                fn_output_signature=tf.float32))

    @tf.function
    def _update_values(values: tf.Tensor, _):
        q_values = tf.map_fn(
            fn=lambda state: _q_s(state, values),
            elems=state_space,
            fn_output_signature=tf.float32)
        next_values = tf.map_fn(
            fn=lambda state: compute_next_value(
                state=state,
                q_values=q_values,
                policy_probs=policy_probs, ),
            elems=state_space)
        delta = _compute_error(values, next_values)
        return next_values, delta, q_values

    next_state_axis = 1 if is_markov_chain else 2
    action_axis = None if is_markov_chain else next_state_axis - 1

    def _q_values_to_values(q_values: tf.Tensor):
        if is_markov_chain:
            values = q_values
        elif policy_probs is None:
            values = tf.reduce_max(q_values, axis=action_axis)
        else:
            values = tf.reduce_sum(q_values * policy_probs, axis=action_axis)

        return values

    @tf.function
    def _update_values_matrices(values: tf.Tensor, _):
        if return_q_values:
            values = _q_values_to_values(values)

        if use_matrix_computation:
            if is_markov_chain:
                q_values = reward_matrix + gamma * tf.squeeze(
                    transition_matrix @ values[..., None])
            else:
                q_values = reward_matrix + gamma * tf.squeeze(
                    tf.transpose(
                        tf.transpose(transition_matrix, perm=[1, 0, 2])
                        @ tf.repeat(values[None], num_actions, axis=0)[..., None]))
        else:
            q_values = tf.reduce_sum(
                transition_matrix * (reward_matrix + gamma * values),
                axis=next_state_axis)

        if episodic_return and return_q_values:
            q_values *= not_reset_states

        next_values = _q_values_to_values(q_values)

        if episodic_return and not return_q_values:
            next_values *= not_reset_states

        delta = _compute_error(values, next_values)

        if return_q_values:
            return q_values, delta
        else:
            return next_values, delta

    @tf.function
    def _compute_error(values, next_values):
        if error_type is Error.ABSOLUTE:
            delta = tf.reduce_max(tf.abs(next_values - values))
        else:
            delta = tf.reduce_max(
                tf.abs(1. - tf.where(
                    condition=values == next_values,
                    x=tf.ones_like(values),
                    y=values / next_values)))
        if debug:
            progress = tf.clip_by_value(
                tf.math.log(delta) * 100. / tf.math.log(epsilon), 0., 100.)
            tf.print('\r', "VI progress:", progress, '% --', 'current error:', delta, output_stream=sys.stdout)
            sys.stdout.flush()
        return delta

    if transition_matrix is not None and reward_matrix is not None:
        update_values = _update_values_matrices
    else:
        update_values = _update_values

    values, _ = tf.while_loop(
        cond=lambda _, _delta: tf.greater_equal(_delta, epsilon),
        body=update_values,
        loop_vars=[values, delta], )

    return values


def compute_next_value(
        state: tf.Tensor,
        q_values: tf.Tensor,
        policy_probs: Optional[tf.Tensor] = None,
        policy: Optional[Callable[[Int], tfd.OneHotCategorical]] = None,
) -> Float:
    """

    Args:
        state: unbatched (single) binary state; expected shape: [S]
               where S is the number of bits used to represent each individual state
        q_values: tensor containing the Q-values of the current step; expected shape: [2**S, A]
               where 2**S is the size of the state space, and A is the size of the action space
        policy_probs: tensor containing the probability of each individual action returned by the policy;
                      expected shape: [2**S, A].
                      If not provided, then the policy function is used directly to compute those probabilities.
        policy: function mapping each (binary encoded) state to a distribution over (one-hot) actions.
                If not provided, then the values of the best action is chosen.
    Returns: the next value of the input state (shape=()).
    """
    _state = tf.reduce_sum(tf.cast(state, tf.int32) * 2 ** tf.range(tf.shape(state)[0]), axis=-1)
    v = q_values[_state, ...]
    if policy_probs is not None:
        return tf.reduce_sum(policy_probs[_state, ...] * v, axis=0)
    elif policy is not None:
        return tf.reduce_sum(
            # the latent policy is assumed batched
            policy(tf.expand_dims(state, 0)).probs_parameter()[0, ...] * v,
            axis=0)
    else:
        return tf.math.maximum(v, axis=0)


def compute_next_q_value(
        state: Int,
        action: Int,
        values: tf.Tensor,
        transition_fn: Callable[[Int, Int], tfd.Distribution],
        reward_fn: Callable[[Int, Int, Int], tf.Tensor],
        gamma: Float,
        state_space: Optional[tf.Tensor] = None,
        is_reset_state_test_fn: Optional[Callable[[tf.Tensor], Bool]] = None,
        episodic_return: bool = True,
) -> Float:
    """
    Compute the next-step Q-value of the input state-action pair.

    Args:
        state: unbatched binary state; expected shape: [S]
               where S is the number of bits used to represent each individual state
        action: unbatched one-hot encoded action; expected shape: [A]
                where A is the size of the action space
        values: tensor containing the value of each state; expected shape: [2**S]
        transition_fn: function mapping each state-action pair to a distribution over (binary encoded) states
        reward_fn: function mapping each transition to a tensor containing the reward
                   obtained by going through this transition.
        gamma: discount factor
        state_space: full binary-encoded state space
        is_reset_state_test_fn: function testing whether the input state is a reset (or null) state or not.
                                If provided,
                                - Rewards obtained from transitions issued from states marked as 'True' are undiscounted
                                - Rewards obtained by transitioning from a state marked as 'True' are null
        episodic_return: Whether to estimate the finite-horizon episodic return or infinite horizon return.
                         If True, is_reset_state_fn has to be provided. In that case, values obtained by transitioning
                         to a reset state will be ignored.

    Returns: the Q-value of the input state-action pair
    """
    if (is_reset_state_test_fn is not None and
            episodic_return and
            is_reset_state_test_fn(state)):
        return 0.
    else:
        num_states = 2 ** tf.shape(state)[0]
        tile = lambda t: tf.tile(
            tf.expand_dims(t, 0),
            [num_states, 1])
        if state_space is None:
            next_states = binary_latent_space(num_states, dtype=tf.float32)
        else:
            next_states = state_space
        tiled_state = tile(state)
        tiled_action = tile(action)

        reward = tf.squeeze(reward_fn(tiled_state, tiled_action, next_states))

        return tf.reduce_sum(
            transition_fn(
                tiled_state, tiled_action
            ).prob(next_states, full_latent_state_space=True) *
            (reward + gamma * values), )
