from typing import Optional, Text, Callable, cast, Tuple

import tensorflow as tf
from tf_agents.agents.categorical_dqn.categorical_dqn_agent import project_distribution
from tf_agents.policies import categorical_q_policy, boltzmann_policy, epsilon_greedy_policy, greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.agents import data_converter, tf_agent

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import utils as network_utils
from tf_agents.trajectories import time_step as ts, trajectory
from tf_agents.networks import network
from tf_agents.typing import types
from tf_agents.typing.types import Float
from tf_agents.utils import common, eager_utils, nest_utils, value_ops

from layers.encoders import TFAgentEncodingNetworkWrapper
from util.io.dataset_generator import map_rl_trajectory_to_vae_input
from wasserstein_mdp import WassersteinMarkovDecisionProcess


class WaeDqnAgent(dqn_agent.DqnAgent):

    def __init__(
            self,
            time_step_spec: ts.TimeStep,
            latent_time_step_spec: ts.TimeStep,
            action_spec: types.NestedTensorSpec,
            label_spec: tf.TensorSpec,
            q_network: network.Network,
            optimizer: types.Optimizer,
            wae_mdp: WassersteinMarkovDecisionProcess,
            labeling_fn: Callable[[Float], Float],
            observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
            epsilon_greedy: Optional[types.FloatOrReturningFloat] = 0.1,
            n_step_update: int = 1,
            encoder_optimizer: Optional[types.Optimizer] = None,
            boltzmann_temperature: Optional[types.FloatOrReturningFloat] = None,
            emit_log_probability: bool = False,
            target_q_network: Optional[network.Network] = None,
            target_update_tau: types.Float = 1.0,
            target_update_period: int = 1,
            td_errors_loss_fn: Optional[types.LossFn] = None,
            gamma: types.Float = 1.0,
            reward_scale_factor: types.Float = 1.0,
            categorical: bool = False,
            gradient_clipping: Optional[types.Float] = None,
            debug_summaries: bool = False,
            summarize_grads_and_vars: bool = False,
            train_step_counter: Optional[tf.Variable] = None,
            name: Optional[Text] = None,
            min_q_value: types.Float = -10.0,
            max_q_value: types.Float = 10.0,
            concatenate_wae_loss: bool = False,
            concatenate_alpha: float = 0.01,
            straight_through: bool = False,
    ):
        self.concatenate_wae_loss = concatenate_wae_loss
        self.concatenate_alpha = concatenate_alpha
        if categorical:
            def check_atoms(net, label):
                try:
                    num_atoms = net.num_atoms
                except AttributeError:
                    raise TypeError('Expected {} to have property `num_atoms`, but it '
                                    'doesn\'t. (Note: you likely want to use a '
                                    'CategoricalQNetwork.) Network is: {}'.format(
                        label, net))
                return num_atoms

            self._num_atoms = check_atoms(
                q_network, 'categorical_q_network')

            if target_q_network is not None:
                target_num_atoms = check_atoms(
                    target_q_network, 'target_categorical_q_network')
                if self._num_atoms != target_num_atoms:
                    raise ValueError(
                        'categorical_q_network and target_categorical_q_network have '
                        'different numbers of atoms: {} vs. {}'.format(
                            self._num_atoms, target_num_atoms))

            self._min_q_value = min_q_value
            self._max_q_value = max_q_value
            min_q_value = tf.convert_to_tensor(min_q_value, dtype_hint=tf.float32)
            max_q_value = tf.convert_to_tensor(max_q_value, dtype_hint=tf.float32)
            self._support = tf.linspace(min_q_value, max_q_value, self._num_atoms)

        self._categorical = categorical
        super().__init__(latent_time_step_spec, action_spec, q_network, optimizer,
                         observation_and_action_constraint_splitter,
                         epsilon_greedy, n_step_update, boltzmann_temperature, emit_log_probability, target_q_network,
                         target_update_tau, target_update_period, td_errors_loss_fn, gamma, reward_scale_factor,
                         gradient_clipping, debug_summaries, summarize_grads_and_vars, train_step_counter, name)

        self._fix_time_step_spec(time_step_spec, action_spec, q_network, gamma, n_step_update)
        self._state_embedding = TFAgentEncodingNetworkWrapper(
            label_spec,
            wae_mdp.state_encoder_temperature,
            state_encoder_network=wae_mdp.state_encoder_network,
            name='CriticStateEmbedding')
        self._state_embedding.create_variables(tf.nest.flatten(time_step_spec.observation) + [label_spec])
        self._target_state_embedding = common.maybe_copy_target_network_with_checks(
            self._state_embedding, None, input_spec=tf.nest.flatten(time_step_spec.observation) + [label_spec],
            name='TargetStateEmbedding')
        self._labeling_fn = labeling_fn
        self._wae_mdp = wae_mdp

        encoder_optimizer = None if self.concatenate_wae_loss else encoder_optimizer
        self._encoder_optimizer = encoder_optimizer
        self.straight_through = straight_through

    def _setup_policy(self, time_step_spec, action_spec,
                      boltzmann_temperature, emit_log_probability):
        if self._categorical:
            policy = categorical_q_policy.CategoricalQPolicy(
                time_step_spec,
                action_spec,
                self._q_network,
                self._min_q_value,
                self._max_q_value,
                observation_and_action_constraint_splitter=(
                    self._observation_and_action_constraint_splitter))

            if boltzmann_temperature is not None:
                collect_policy = boltzmann_policy.BoltzmannPolicy(
                    policy, temperature=boltzmann_temperature)
            else:
                collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                    policy, epsilon=self._epsilon_greedy)
            policy = greedy_policy.GreedyPolicy(policy)

            target_policy = categorical_q_policy.CategoricalQPolicy(
                time_step_spec,
                action_spec,
                self._target_q_network,
                self._min_q_value,
                self._max_q_value,
                observation_and_action_constraint_splitter=(
                    self._observation_and_action_constraint_splitter))
            self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)

            return policy, collect_policy

        else:
            return super()._setup_policy(time_step_spec, action_spec, boltzmann_temperature, emit_log_probability)

    def _check_network_output(self, net, label):
        if self._categorical:
            network_utils.check_single_floating_network_output(
                net.create_variables(),
                expected_output_shape=(self._num_actions, self._num_atoms),
                label=label)
        else:
            super()._check_network_output(net, label)

    def _fix_time_step_spec(self, time_step_spec, action_spec, q_network, gamma, n_step_update):
        # fix diverse time_step_spec issues occurring when repeatedly encoding input observations
        if not isinstance(time_step_spec, ts.TimeStep):
            raise TypeError(
                "The `time_step_spec` must be an instance of `TimeStep`, but is `{}`.".format(type(time_step_spec)))
        time_step_spec = tensor_spec.from_spec(time_step_spec)
        self._time_step_spec = time_step_spec

        # Data context for data collected directly from the collect policy.
        self._collect_data_context = data_converter.DataContext(
            time_step_spec=self._time_step_spec,
            action_spec=self._action_spec,
            info_spec=self._collect_policy.info_spec)
        self._data_context = data_converter.DataContext(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=self._collect_policy.info_spec)
        if q_network.state_spec:
            # AsNStepTransition does not support emitting [B, T, ...] tensors,
            # which we need for DQN-RNN.
            self._as_transition = data_converter.AsTransition(
                self.data_context, squeeze_time_dim=False)
        else:
            # This reduces the n-step return and removes the extra time dimension,
            # allowing the rest of the computations to be independent of the
            # n-step parameter.
            self._as_transition = data_converter.AsNStepTransition(
                self.data_context, gamma=gamma, n=n_step_update)

    def _initialize(self):
        super(WaeDqnAgent, self)._initialize()
        common.soft_variables_update(
            self._state_embedding.variables, self._target_state_embedding.variables, tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        with tf.name_scope('update_targets'):
            def update():
                return common.soft_variables_update(
                    self._q_network.variables + self._state_embedding.variables,
                    self._target_q_network.variables + self._target_state_embedding.variables,
                    tau,
                    tau_non_trainable=1.0)

            return common.Periodically(update, period, 'periodic_update_targets')

    def _train(self, experience, weights):
        # copy pasta from DQN TFAgent
        with tf.GradientTape(persistent=self._encoder_optimizer is not None) as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True)
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        # changes from here
        variables_to_train = self._q_network.trainable_weights
        encoder_variables_to_train = self._state_embedding.trainable_weights
        non_trainable_weights = self._q_network.non_trainable_weights
        encoder_non_trainable_weights = self._state_embedding.non_trainable_weights

        if self.concatenate_wae_loss:
            variables_to_train += self._wae_mdp.inference_variables
            variables_to_train += self._wae_mdp.generator_variables

        if self._encoder_optimizer is None:
            variables_to_train += encoder_variables_to_train
            non_trainable_weights += encoder_non_trainable_weights

        assert list(variables_to_train), "No variables in the agent's q_network."
        for optimizer, _variables_to_train, _non_trainable_weights in [
            (self._optimizer, variables_to_train, non_trainable_weights),
            (self._encoder_optimizer, encoder_variables_to_train, encoder_non_trainable_weights)
        ]:
            if optimizer is not None:
                grads = tape.gradient(loss_info.loss, _variables_to_train)
                grads_and_vars = list(zip(grads, _variables_to_train))
                if self._gradient_clipping is not None:
                    grads_and_vars = eager_utils.clip_gradient_norms(
                        grads_and_vars, self._gradient_clipping)

                if self._summarize_grads_and_vars:
                    grads_and_vars_with_non_trainable = (
                            grads_and_vars + [(None, v) for v in _non_trainable_weights])
                    eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
                                                        self.train_step_counter)
                    eager_utils.add_gradients_summaries(grads_and_vars,
                                                        self.train_step_counter)
                optimizer.apply_gradients(grads_and_vars)

        self.train_step_counter.assign_add(1)

        self._update_target()

        return loss_info

    def _compute_q_values(self, time_steps, actions, training=False):
        # copy pasta from TFAgent DQN
        network_observation = time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        # changes here
        embedded_observation, _ = self._state_embedding(
            tuple(tf.cast(_obs, tf.float32) for _obs in tf.nest.flatten(network_observation)) +
            (self._labeling_fn(network_observation), ),
            training=training)
        if self.straight_through:
            embedded_observation = tf.round(embedded_observation) + embedded_observation - \
                                   tf.stop_gradient(embedded_observation)

        q_values, _ = self._q_network(embedded_observation,
                                      step_type=time_steps.step_type,
                                      training=training)
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        action_spec = cast(tensor_spec.BoundedTensorSpec, self._action_spec)
        multi_dim_actions = action_spec.shape.rank > 0
        return common.index_with_actions(
            q_values,
            tf.cast(actions, dtype=tf.int32),
            multi_dim_actions=multi_dim_actions)

    def _compute_next_q_values(self, next_time_steps, info):
        # copy pasta from TFAgent DQN
        del info
        network_observation = next_time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        # changes here
        embedded_observation, _ = self._target_state_embedding(
            tuple(tf.cast(_obs, tf.float32) for _obs in tf.nest.flatten(network_observation)) +
            (self._labeling_fn(network_observation),),
            training=False)
        if self.straight_through:
            embedded_observation = tf.round(embedded_observation) + embedded_observation - \
                                      tf.stop_gradient(embedded_observation)

        next_target_q_values, _ = self._target_q_network(
            embedded_observation, step_type=next_time_steps.step_type)
        batch_size = (
                next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
        dummy_state = self._policy.get_initial_state(batch_size)
        # Find the greedy actions using our greedy policy. This ensures that action
        # constraints are respected and helps centralize the greedy logic.
        best_next_actions = self._policy.action(
            next_time_steps._replace(observation=embedded_observation),
            dummy_state
        ).action

        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.rank > 0
        return common.index_with_actions(
            next_target_q_values,
            best_next_actions,
            multi_dim_actions=multi_dim_actions)

    def _loss(self,
              experience,
              td_errors_loss_fn=None,
              gamma=1.0,
              reward_scale_factor=1.0,
              weights=None,
              training=False):
        reward_scale_factor = tf.cast(reward_scale_factor, tf.float32)
        if self._categorical:
            # copy-pasted from tf_agents.agents.categorical_dqn.categorical_dqn_agent
            squeeze_time_dim = not self._q_network.state_spec
            if self._n_step_update == 1:
                time_steps, policy_steps, next_time_steps = (
                    trajectory.experience_to_transitions(experience, squeeze_time_dim))
                actions = policy_steps.action
            else:
                # To compute n-step returns, we need the first time steps, the first
                # actions, and the last time steps. Therefore, we extract the first and
                # last transitions from our Trajectory.
                first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
                last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
                time_steps, policy_steps, _ = (
                    trajectory.experience_to_transitions(
                        first_two_steps, squeeze_time_dim))
                actions = policy_steps.action
                _, _, next_time_steps = (
                    trajectory.experience_to_transitions(
                        last_two_steps, squeeze_time_dim))

            with tf.name_scope('critic_loss'):
                nest_utils.assert_same_structure(actions, self.action_spec)
                nest_utils.assert_same_structure(time_steps, self.time_step_spec)
                nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

                rank = nest_utils.get_outer_rank(time_steps.observation,
                                                 self._time_step_spec.observation)

                # If inputs have a time dimension and the q_network is stateful,
                # combine the batch and time dimension.
                batch_squash = (None
                                if rank <= 1 or self._q_network.state_spec in ((), None)
                                else network_utils.BatchSquash(rank))

                network_observation = time_steps.observation

                if self._observation_and_action_constraint_splitter is not None:
                    network_observation, _ = (
                        self._observation_and_action_constraint_splitter(
                            network_observation))

                # changes here
                embedded_observation, _ = self._state_embedding(
                    tuple(tf.cast(_obs, tf.float32) for _obs in tf.nest.flatten(network_observation)) +
                    (self._labeling_fn(network_observation),),
                    training=training)
                if self.straight_through:
                    embedded_observation = tf.round(embedded_observation) + embedded_observation - \
                                             tf.stop_gradient(embedded_observation)

                # q_logits contains the Q-value logits for all actions.
                q_logits, _ = self._q_network(embedded_observation,
                                              step_type=time_steps.step_type,
                                              training=training)

                if batch_squash is not None:
                    # Squash outer dimensions to a single dimensions for facilitation
                    # computing the loss the following. Required for supporting temporal
                    # inputs, for example.
                    q_logits = batch_squash.flatten(q_logits)
                    actions = batch_squash.flatten(actions)
                    next_time_steps = tf.nest.map_structure(batch_squash.flatten,
                                                            next_time_steps)

                next_q_distribution = self._next_q_distribution(next_time_steps)

                if actions.shape.rank > 1:
                    actions = tf.squeeze(actions, list(range(1, actions.shape.rank)))

                # Project the sample Bellman update \hat{T}Z_{\theta} onto the original
                # support of Z_{\theta} (see Figure 1 in paper).
                batch_size = q_logits.shape[0] or tf.shape(q_logits)[0]
                tiled_support = tf.tile(self._support, [batch_size])
                tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

                if self._n_step_update == 1:
                    discount = next_time_steps.discount
                    if discount.shape.rank == 1:
                        # We expect discount to have a shape of [batch_size], while
                        # tiled_support will have a shape of [batch_size, num_atoms]. To
                        # multiply these, we add a second dimension of 1 to the discount.
                        discount = tf.expand_dims(discount, -1)
                    next_value_term = tf.multiply(discount,
                                                  tiled_support,
                                                  name='next_value_term')

                    reward = next_time_steps.reward
                    if reward.shape.rank == 1:
                        # See the explanation above.
                        reward = tf.expand_dims(reward, -1)
                    reward_term = tf.multiply(reward_scale_factor,
                                              reward,
                                              name='reward_term')

                    target_support = tf.add(reward_term, gamma * next_value_term,
                                            name='target_support')
                else:
                    # When computing discounted return, we need to throw out the last time
                    # index of both reward and discount, which are filled with dummy values
                    # to match the dimensions of the observation.
                    rewards = reward_scale_factor * experience.reward[:, :-1]
                    discounts = gamma * experience.discount[:, :-1]

                    # TODO(b/134618876): Properly handle Trajectories that include episode
                    # boundaries with nonzero discount.

                    discounted_returns = value_ops.discounted_return(
                        rewards=rewards,
                        discounts=discounts,
                        final_value=tf.zeros([batch_size], dtype=discounts.dtype),
                        time_major=False,
                        provide_all_returns=False)

                    # Convert discounted_returns from [batch_size] to [batch_size, 1]
                    discounted_returns = tf.expand_dims(discounted_returns, -1)

                    final_value_discount = tf.reduce_prod(discounts, axis=1)
                    final_value_discount = tf.expand_dims(final_value_discount, -1)

                    # Save the values of discounted_returns and final_value_discount in
                    # order to check them in unit tests.
                    self._discounted_returns = discounted_returns
                    self._final_value_discount = final_value_discount

                    target_support = tf.add(discounted_returns,
                                            final_value_discount * tiled_support,
                                            name='target_support')

                target_distribution = tf.stop_gradient(project_distribution(
                    target_support, next_q_distribution, self._support))

                # Obtain the current Q-value logits for the selected actions.
                indices = tf.range(batch_size)
                indices = tf.cast(indices, actions.dtype)
                reshaped_actions = tf.stack([indices, actions], axis=-1)
                chosen_action_logits = tf.gather_nd(q_logits, reshaped_actions)

                # Compute the cross-entropy loss between the logits. If inputs have
                # a time dimension, compute the sum over the time dimension before
                # computing the mean over the batch dimension.
                if batch_squash is not None:
                    target_distribution = batch_squash.unflatten(target_distribution)
                    chosen_action_logits = batch_squash.unflatten(chosen_action_logits)
                    critic_loss = tf.reduce_sum(
                        tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                            labels=target_distribution,
                            logits=chosen_action_logits),
                        axis=1)
                else:
                    critic_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                        labels=target_distribution,
                        logits=chosen_action_logits)

                agg_loss = common.aggregate_losses(
                    per_example_loss=critic_loss,
                    regularization_loss=self._q_network.losses)
                total_loss = agg_loss.total_loss

                dict_losses = {'critic_loss': agg_loss.weighted,
                               'reg_loss': agg_loss.regularization,
                               'total_loss': total_loss}

                common.summarize_scalar_dict(dict_losses,
                                             step=self.train_step_counter,
                                             name_scope='Losses/')

                if self._debug_summaries:
                    distribution_errors = target_distribution - chosen_action_logits
                    with tf.name_scope('distribution_errors'):
                        common.generate_tensor_summaries(
                            'distribution_errors', distribution_errors,
                            step=self.train_step_counter)
                        tf.compat.v2.summary.scalar(
                            'mean', tf.reduce_mean(distribution_errors),
                            step=self.train_step_counter)
                        tf.compat.v2.summary.scalar(
                            'mean_abs', tf.reduce_mean(tf.abs(distribution_errors)),
                            step=self.train_step_counter)
                        tf.compat.v2.summary.scalar(
                            'max', tf.reduce_max(distribution_errors),
                            step=self.train_step_counter)
                        tf.compat.v2.summary.scalar(
                            'min', tf.reduce_min(distribution_errors),
                            step=self.train_step_counter)
                    with tf.name_scope('target_distribution'):
                        common.generate_tensor_summaries(
                            'target_distribution', target_distribution,
                            step=self.train_step_counter)

                # TODO(b/127318640): Give appropriate values for td_loss and td_error for
                # prioritized replay.
                loss = tf_agent.LossInfo(total_loss, dqn_agent.DqnLossInfo(td_loss=(),
                                                                           td_error=()))

        else:
            loss = super()._loss(experience, td_errors_loss_fn, gamma, reward_scale_factor, weights, training)

        if self.concatenate_wae_loss and self.concatenate_alpha > 0.:
            
            time_steps, _, next_time_steps = self._as_transition(experience)

            observations = tuple(tf.cast(_obs, tf.float32) for _obs in tf.nest.flatten(experience.observation))
            state = tuple(_obs[:, 0, ...] for _obs in observations)
            next_state = tuple(_obs[:, 1, ...] for _obs in observations)
            action = experience.action[:, 0, ...]
            action = tf.one_hot(
                indices=action,
                depth=tf.cast(self._wae_mdp.number_of_discrete_actions, dtype=tf.int32),
                dtype=tf.float32)
            label = tf.cast(self._labeling_fn(time_steps.observation), tf.float32)
            next_label = tf.cast(self._labeling_fn(next_time_steps.observation), tf.float32)
            reward = experience.reward[:, 0, None]
            if len(state) == 1:
                state = state[0]
                next_state = next_state[0]

            wae_loss = self._wae_mdp.compute_loss(
                state=state,
                label=label,
                action=action,
                reward=reward,
                next_state=next_state,
                next_label=next_label,
                training=training,
                optimization_direction='min',
            )['min']

            loss = tf_agent.LossInfo(
                (1 - self.concatenate_alpha) * loss.loss + self.concatenate_alpha * wae_loss,
                loss.extra
            )

        return loss

    def _next_q_distribution(self, next_time_steps):
        """Compute the q distribution of the next state for TD error computation.

        Args:
          next_time_steps: A batch of next timesteps

        Returns:
          A [batch_size, num_atoms] tensor representing the Q-distribution for the
          next state.
        """
        # copy-pasted from tf_agents.agents.categorical_dqn.categorical_projection_network
        network_observation = next_time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        # changes here
        embedded_observation, _ = self._target_state_embedding(
            tuple(tf.cast(_obs, tf.float32) for _obs in tf.nest.flatten(network_observation)) +
            (self._labeling_fn(network_observation),),
            training=False)
        if self.straight_through:
            embedded_observation = tf.round(embedded_observation) + embedded_observation - \
                                   tf.stop_gradient(embedded_observation)

        next_target_logits, _ = self._target_q_network(
            embedded_observation,
            step_type=next_time_steps.step_type,
            training=False)
        batch_size = next_target_logits.shape[0] or tf.shape(next_target_logits)[0]
        next_target_probabilities = tf.nn.softmax(next_target_logits)
        next_target_q_values = tf.reduce_sum(
            self._support * next_target_probabilities, axis=-1)
        dummy_state = self._policy.get_initial_state(batch_size)
        # Find the greedy actions using our target greedy policy. This ensures that
        # action constraints are respected and helps centralize the greedy logic.
        # double dqn
        best_next_actions = self._policy.action(
            next_time_steps._replace(observation=embedded_observation),
            dummy_state
        ).action
        next_qt_argmax = tf.cast(best_next_actions, tf.int32)[:, None]
        batch_indices = tf.range(
            tf.cast(tf.shape(next_target_q_values)[0], tf.int32))[:, None]
        next_qt_argmax = tf.concat([batch_indices, next_qt_argmax], axis=-1)
        return tf.gather_nd(next_target_probabilities, next_qt_argmax)
