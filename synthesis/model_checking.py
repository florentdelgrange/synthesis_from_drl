import collections
import os.path
import time
from typing import Optional, Callable, Dict, Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.environments import PyEnvironment, suite_gym, tf_py_environment, parallel_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import TFPolicy
from tf_agents.typing.types import Float, Bool
from tf_agents.trajectories.policy_step import PolicyStep

import verification
import wasserstein_mdp
from reinforcement_learning import labeling_functions
from reinforcement_learning.environments import EnvironmentLoader
from reinforcement_learning.environments.two_level_env import Directions
from reinforcement_learning.environments.perturbed_env import PerturbedEnvironment
from util.io.dataset_generator import ergodic_batched_labeling_function, is_reset_state
from verification import binary_latent_space
from verification.value_iteration import value_iteration

tfd = tfp.distributions


def retrieve_epsilon_greedy_distribution(
        wae_mdp: wasserstein_mdp.WassersteinMarkovDecisionProcess,
        policy: TFPolicy,
        epsilon: float = .05,
        return_distribution: bool = True,
) -> Union[tfd.Distribution, TFPolicy]:
    """
    The policy uses the tf_agent's epsilon greedy implementation, which is nasty;
    the latter records the epsilon greedy distribution as an action instead of a distribution.
    Consequently, the optimal action obtained by doing policy.action() is gathered with
    probability 1 - epsilon, and the policy.distribution() is merely a deterministic
    distribution which yields policy.action(). However, the policy reports the log probability
    with which the action has been sampled, via policy.action().info.
    To gather the optimal actions, we repeatedly sample actions until all the actions are
    drawn with log probability zero (i.e., probability one).

    """
    latent_size = policy.time_step_spec.observation.shape[0]
    state_space = binary_latent_space(latent_size, dtype=tf.float32)
    step = policy.action(
        ts.transition(observation=state_space, reward=tf.zeros((2 ** wae_mdp.latent_state_size,)))
    )
    actions = step.action
    if hasattr(step.info, 'log_probability'):
        log_probs = step.info.log_probability

        while len(tf.where(log_probs != 0.)) > 0:
            step = policy.action(
                ts.transition(observation=state_space, reward=tf.zeros((2 ** wae_mdp.latent_state_size,)))
            )
            actions = tf.where(
                log_probs == 0.,
                actions,
                step.action
            )
            log_probs = tf.where(
                log_probs == 0.,
                log_probs,
                step.info.log_probability
            )

        non_optimal_action_prob = epsilon / (policy.action_spec.maximum + 1)
        probs = (
            # location
                tf.one_hot(actions, depth=policy.action_spec.maximum + 1)
                # scale
                * (1. - epsilon - non_optimal_action_prob)
                # shift
                + non_optimal_action_prob
        )
    else:
        probs = tf.one_hot(actions, depth=policy.action_spec.maximum + 1)

    if return_distribution:
        return tfd.Categorical(probs=probs)
    else:
        class _Policy(TFPolicy):
            def __init__(self, probs: Float):
                super(_Policy, self).__init__(
                    time_step_spec=policy.time_step_spec,
                    action_spec=policy.action_spec,
                )
                self._probs = probs

            def _distribution(self, time_step, policy_state):
                latent_state = tf.cast(time_step.observation, dtype=tf.float32)
                #  idx = tf.where(tf.reduce_all(latent_state[:, None, ...] == state_space, axis=-1))[..., 1:]
                idx = tf.reduce_sum(
                    2. ** tf.range(latent_size, dtype=tf.float32) * latent_state,
                    axis=-1
                )[..., None]
                _probs = tf.gather_nd(self._probs, tf.cast(idx, dtype=tf.int64))
                return PolicyStep(
                    tfd.Categorical(
                        probs=_probs,
                        dtype=tf.int64),
                    (), ())

            def _variables(self):
                return self._probs

        return _Policy(probs=probs)


@tf.function
def get_p_init(
        wae_mdp: wasserstein_mdp.WassersteinMarkovDecisionProcess,
        py_env: PyEnvironment,
        policy: tfd.Distribution,
        environment_name: Optional[str] = None,
):
    if not environment_name:
        environment_name = py_env.gym.unwrapped.spec.id
    latent_state_space = binary_latent_space(wae_mdp.latent_state_size)
    is_reset_state_test_fn = lambda latent_state: is_reset_state(latent_state, wae_mdp.atomic_prop_dims)

    if len(tf.nest.flatten(py_env.observation_spec())) == 1:
        original_reset_state = np.zeros(
            shape=py_env.observation_spec().shape,
            dtype=py_env.observation_spec().dtype)[None]
    elif isinstance(py_env.observation_spec(), collections.OrderedDict):
        original_reset_state = collections.OrderedDict(
            [(key, tf.zeros(shape=value.shape, dtype=tf.float32)[None])
             for key, value in py_env.observation_spec().items()])

    reset_state = wae_mdp.state_embedding_function(
        original_reset_state,
        ergodic_batched_labeling_function(
            labeling_functions[environment_name]
        )(original_reset_state))
    reset_state = tf.cast(reset_state, tf.float32)
    reset_state = tf.tile(reset_state, [tf.shape(latent_state_space)[0], 1])

    latent_action_space = tf.one_hot(
        indices=tf.range(wae_mdp.number_of_discrete_actions),
        depth=tf.cast(wae_mdp.number_of_discrete_actions, tf.int32),
        dtype=tf.float32)

    return tf.reduce_sum(
        tf.transpose(
            policy.probs_parameter()
        ) * tf.map_fn(
            fn=lambda latent_action: wae_mdp.discrete_latent_transition(
                reset_state,
                tf.tile(tf.expand_dims(latent_action, 0), [tf.shape(latent_state_space)[0], 1]),
            ).prob(
                tf.cast(latent_state_space, tf.float32)),
            elems=latent_action_space),
        axis=0) * (1. - tf.cast(is_reset_state_test_fn(latent_state_space), tf.float32))


def C_until_T_values(
        C_fn: Callable[[Float], Bool],
        T_fn: Callable[[Float], Bool],
        transition_matrix: Float,
        latent_state_size: int,
        A: int,
        latent_policy: tfd.Distribution,
        gamma: Float = 0.99,
        transition_to_T_reward: Optional[Float] = None,
        use_matrix_computation: bool = True,
        error_type: str = 'absolute',
        epsilon: float = 1e-6,
) -> Float:
    S = tf.pow(2, latent_state_size)
    state_space = binary_latent_space(latent_state_size, dtype=tf.float32)

    # make absorbing ¬C and T
    absorbing_states = lambda latent_state: tf.math.logical_or(
        tf.math.logical_not(C_fn(latent_state)),
        T_fn(latent_state))

    # reward of 1 when transitioning to T;
    # set the reward to the input values if provided
    reward_objective = tf.ones(
        shape=(S, A, S),
    ) * tf.cast(T_fn(state_space), tf.float32)
    if transition_to_T_reward is not None:
        reward_objective *= transition_to_T_reward

    policy_probs = latent_policy.probs_parameter()

    values = value_iteration(
        latent_state_size=latent_state_size,
        num_actions=A,
        transition_fn=transition_matrix,
        reward_fn=reward_objective,
        gamma=gamma,
        policy_probs=policy_probs,
        epsilon=epsilon,
        v_init=tf.zeros(S, dtype=tf.float32),
        episodic_return=True,
        is_reset_state_test_fn=absorbing_states,
        error_type=error_type,
        transition_matrix=transition_matrix,
        reward_matrix=reward_objective,
        use_matrix_computation=use_matrix_computation)

    # set the values of the target states to either one or the input values if provided
    if transition_to_T_reward is None:
        values = values + tf.cast(T_fn(state_space), tf.float32)
    else:
        values = values + (tf.cast(T_fn(state_space), tf.float32) * transition_to_T_reward)

    return values


def get_transition_matrix(
        wae_mdp: wasserstein_mdp.WassersteinMarkovDecisionProcess,
        sparse=False,
        batch_mode=True,
):
    _latent_transition_fn = lambda latent_state, latent_action: \
        wae_mdp.discrete_latent_transition(
            tf.cast(latent_state, tf.float32),
            tf.cast(latent_action, tf.float32))

    start = time.time()

    #  write the transition function to tensors,
    #  to formally check the values in an efficient way
    latent_transition_fn = verification.model.TransitionFunctionCopy(
        num_states=tf.cast(tf.pow(2, wae_mdp.latent_state_size), dtype=tf.int32),
        num_actions=wae_mdp.number_of_discrete_actions,
        transition_function=_latent_transition_fn,
        epsilon=1e-6,
        sparse=sparse,
        batch_mode=batch_mode)

    end = time.time() - start

    print("Time to generate the transition matrix: {:.2g} sec".format(end))

    return latent_transition_fn.to_dense()


def compute_latent_mdp_values(
        latent_models: Dict[Directions, Dict[str, Union[wasserstein_mdp.WassersteinMarkovDecisionProcess, TFPolicy]]],
        discount: float = 0.99,
        batch_mode: Optional[bool] = None,
) -> Tuple[Dict[Directions, Dict[str, Float]], Dict[str, float]]:
    values = dict()
    time_metrics = dict()

    for direction in [_dir for _dir in Directions if _dir != Directions.NOOP]:

        print(f"Latent MDP for {str(direction)}")

        wae_mdp = latent_models[direction]['model']
        policy = latent_models[direction]['policy']
        policy_distribution = retrieve_epsilon_greedy_distribution(wae_mdp, policy)

        available_devices = [device.device_type for device in tf.config.experimental.list_physical_devices()]
        if 'GPU' in available_devices and wae_mdp.latent_state_size < 13:
            device = 'GPU'
        else:
            device = 'CPU'
        print(">>", f"latent state size: {wae_mdp.latent_state_size:d}"
                    f" ({2 ** wae_mdp.latent_state_size:d} states)")
        print(">>", f"computing values on {device}")

        if batch_mode is None:
            _batch_mode = device == 'GPU'
        else:
            _batch_mode = batch_mode

        with tf.device(f'/{device}:0'):
            time_metrics[f'model generation ({str(Directions(direction))})'] = time.time()

            transition_matrix = get_transition_matrix(
                wae_mdp,
                sparse=False,
                batch_mode=_batch_mode)
            time_metrics[f'model generation ({str(Directions(direction))})'] = \
                time.time() - time_metrics[f'model generation ({str(Directions(direction))})']

            time_metrics[f'verifying latent model ({str(Directions(direction))})'] = time.time()
            start = time.time()

            # safe and not reset;
            # latent_state[..., 1] embeds the bad states
            C = lambda latent_state: tf.math.logical_and(
                tf.math.logical_not(tf.cast(latent_state[..., 1], tf.bool)),
                tf.math.logical_not(
                    is_reset_state(latent_state, wae_mdp.atomic_prop_dims)))
            # succeeds
            # latent_state[..., 0] embeds the target states
            T = lambda latent_state: tf.cast(latent_state[..., 0], tf.bool)

            safe_until_goal = C_until_T_values(
                # be safe during the episode
                C_fn=C,
                # ... until the goal position is reached
                T_fn=T,
                transition_matrix=transition_matrix,
                latent_state_size=wae_mdp.latent_state_size,
                A=wae_mdp.number_of_discrete_actions,
                latent_policy=policy_distribution,
                gamma=discount,
                epsilon=1e-6,
                error_type='absolute')

            end = time.time() - start

            print('>>', 'Time to compute values for specification'
                        '" [¬fail and ¬reset] U goal": {:.3g} sec'.format(end))

            start = time.time()

            # not reset and not target state
            C = lambda latent_state: tf.math.logical_and(
                tf.math.logical_not(is_reset_state(latent_state, wae_mdp.atomic_prop_dims)),
                tf.math.logical_not(tf.cast(latent_state[..., 0], tf.bool)))
            # failure
            T = lambda latent_state: tf.cast(latent_state[..., 1], tf.bool)

            reach_unsafe_pos = C_until_T_values(
                C_fn=C,
                # reach a bad state during an episode
                T_fn=T,
                transition_matrix=transition_matrix,
                latent_state_size=wae_mdp.latent_state_size,
                A=wae_mdp.number_of_discrete_actions,
                latent_policy=policy_distribution,
                gamma=discount,
                epsilon=1e-6,
                error_type='absolute')

            end = time.time() - start
            time_metrics[f'verifying latent model ({str(Directions(direction))})'] = \
                time.time() - time_metrics[f'verifying latent model ({str(Directions(direction))})']

            print('>>', 'Time to compute values for specification'
                        '" [¬reset and ¬target] U fail": {:.3g} sec'.format(end))

            del transition_matrix

        values[direction] = {
            'succeed': safe_until_goal,
            'fail': reach_unsafe_pos,
        }

    return values, time_metrics


def save_values(
        values: Dict[Directions, Dict[str, Float]],
        discount: float = 0.99,
        path: str = 'latent_values',
):
    for direction in values.keys():
        dataset_ = tf.data.Dataset.from_tensor_slices(values[direction])

        # Define the path where you want to save the dataset
        save_path = os.path.join(path, f'discount={discount:.2g}', f'values_{direction}')

        # Save the dataset
        tf.data.experimental.save(dataset_, save_path)


def load_values(
        discount: float = 0.99,
        path: str = 'latent_values',
) -> Dict[Directions, Dict[str, Float]]:
    values = dict()
    for direction in [_dir for _dir in Directions if _dir != Directions.NOOP]:
        # Define the path where you want to save the dataset
        load_path = os.path.join(path, f'discount={discount:.2g}', f'values_{direction}')
        # Load dataset
        dataset_ = tf.data.experimental.load(load_path)
        _dataset = {}

        for data in dataset_:
            for key, value in data.items():
                if key not in _dataset:
                    _dataset[key] = []
                _dataset[key].append(value)

        for key, value in _dataset.items():
            _dataset[key] = tf.stack(value)
        values[direction] = _dataset

    return values


def get_pac_bounds(
        latent_models: Dict[Directions, Dict[str, Union[wasserstein_mdp.WassersteinMarkovDecisionProcess, TFPolicy]]],
        environment_prefix_name: str,
        epsilon: float = 0.05,
        delta: float = 0.01,
        reward_scaling: bool = False,
        num_parallel_environments: int = 8,
        seed: int = 42,
        average_episode_length: int = 100,
        *args, **kwargs
) -> Tuple[Dict[str, Float], Dict[str, Float]]:
    pac_metrics = dict()
    time_metrics = dict()
    steps = int(np.ceil(-np.log(delta / 4) / (2 * epsilon**2)))
    env_loader = EnvironmentLoader(suite_gym, seed=seed)
    rb_max_frames = average_episode_length * steps // num_parallel_environments

    for direction in [_dir for _dir in Directions if _dir != Directions.NOOP]:
        env_name = f'{environment_prefix_name}{direction.to_cardinal().capitalize()}-v0'
        print(">>", f"environment: {env_name}")

        env_wrappers = [lambda env: PerturbedEnvironment(
            env,
            perturbation=.01,
            recursive_perturbation=True)]
        # load the environment
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_loader.load(env_name, env_wrappers)] * num_parallel_environments))
        metrics = latent_models[direction]['model'].estimate_local_losses_from_samples(
            environment=tf_env,
            steps=steps,
            labeling_function=labeling_functions[env_name],
            latent_policy=latent_models[direction]['policy'],
            reward_scaling=latent_models[direction]['model']._dynamic_reward_scaling if reward_scaling else 1.,
            estimate_value_difference=False,
            replay_buffer_max_frames=rb_max_frames
        )
        tf_env.close()

        pac_metrics[f'reward loss ({str(Directions(direction))})'] = metrics.local_reward_loss.numpy()
        pac_metrics[f'transition loss ({str(Directions(direction))})'] = metrics.local_transition_loss.numpy()
        time_metrics = {
            **time_metrics,
            **{f'{key} ({str(Directions(direction))})': value
               for key, value in metrics.time_metrics.items()
               if key != 'start'}}

    return pac_metrics, time_metrics
