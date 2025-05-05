import enum
from enum import IntEnum
from typing import Tuple, Callable, Any, Optional, Union

import tensorflow as tf
import tf_agents.replay_buffers.tf_uniform_replay_buffer
import tf_agents.trajectories.time_step as ts
from numpy.typing import ArrayLike
from tensorflow.keras import layers as tfkl

import numpy as np
from tf_agents.environments import PyEnvironmentBaseWrapper
from tf_agents.replay_buffers import table, tf_uniform_replay_buffer
from tf_agents.trajectories import Trajectory
from tf_agents.typing.types import PyEnv, Float
from tf_agents.utils import common

from reinforcement_learning.environments.goal_augmented import GoalAugmentedEnvironment


# ==================================
#  Goal-conditioned utils
# ==================================

class ConditionalLabelingCombiner(tfkl.Layer):
    def __init__(self, conditional_labeling: Tuple[int, int], conditional_units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.dense_0 = tfkl.Dense(
            units=conditional_units,
            activation=tf.nn.sigmoid,
            name='dense_combiner_latent_state')
        self.dense_1 = tfkl.Dense(
            units=conditional_units,
            activation=tf.nn.sigmoid,
            name='dense_combiner_one_hot_label')
        self._conditional_labeling = conditional_labeling

    def build(self, input_shape):
        i, j = self._conditional_labeling
        self.dense_0.build(input_shape=input_shape)
        self.dense_1.build(input_shape=(2 ** (j - i),))

    def call(self, inputs, *args, **kwargs):
        i, j = self._conditional_labeling
        n = 2 ** (j - i)
        one_hot_labeling = tf.stop_gradient(
            tf.one_hot(
                indices=tf.reduce_sum(
                    2 ** tf.range(j - i) * tf.cast(inputs[..., i:j], dtype=tf.int32),
                    axis=-1),
                depth=n,
                dtype=tf.float32))
        return self.dense_0(inputs) * self.dense_1(one_hot_labeling)


class SamplingStrategy(IntEnum):
    FINAL = 0
    FUTURE = 1


class HindsightExperienceReplayBuffer(tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer):

    def __init__(
            self,
            data_spec,
            batch_size,
            goal_augmented_env: Union[GoalAugmentedEnvironment, PyEnv],
            sampling_strategy: Optional[SamplingStrategy] = SamplingStrategy.FINAL,
            n_future_samples: int = 4,
            gamma: float = 1.,
            time_step_cost: float = 1.,
            state_penalty_multiplier: float = 1.,
            goal_reward_multiplier: float = 0.,
            reward_horizon: int = 100,
            max_length=1000,
            scope='TFUniformReplayBuffer',
            device='cpu:*',
            table_fn=table.Table,
            dataset_drop_remainder=False,
            dataset_window_shift=None,
            stateful_dataset=False
    ):
        assert batch_size == 1, "Batch size > 1 is currently not supported."
        super().__init__(data_spec, batch_size, max_length, scope, device, table_fn,
                         dataset_drop_remainder, dataset_window_shift, stateful_dataset)
        self._env = goal_augmented_env
        self.sampling_strategy = sampling_strategy
        self._current_episode = []
        self.n_future_samples = n_future_samples
        self._gamma = gamma
        self._time_step_cost = -1. * np.abs(time_step_cost)
        assert self._time_step_cost < 0., "The non-final time step cost should be non-nul"
        self._state_penalty_multiplier = state_penalty_multiplier
        self._goal_reward_multiplier = goal_reward_multiplier
        self._reward_horizon = reward_horizon
        self._her_ratio = 1 - (1.0 / (n_future_samples + 1))

        with tf.device(self._device), tf.compat.v1.variable_scope(self._scope):
            self._episode = common.create_variable('episode', 0)
            self._episode_cs = tf.CriticalSection(name='episode')
            self._last_episode = common.create_variable('last_episode', 0)
            self._episode_cs = tf.CriticalSection(name='last_episode')
            self._episode_table = table_fn(self._id_spec, self._capacity_value)
            self._episode_end_indices_table = table_fn(self._id_spec, self._capacity_value)


    def reward_fn(self, observation: tf.Tensor, goal: tf.Tensor) -> Float:
        is_achieved = tf.cast(
            self._env.is_achieved(observation, goal, batched=True, lib=tf),
            tf.float32)
        is_unsafe = tf.cast(self._env.is_unsafe(observation, lib=tf), tf.float32)
        ts_cost = (1. - is_achieved) * (1. - is_unsafe) * self._time_step_cost
        unsafe_cost = is_unsafe * self._state_penalty_multiplier * np.sum(
            self._time_step_cost *
            np.array([self._gamma ** i for i in range(self._reward_horizon)]))
        goal_reward = is_achieved * (1. - is_unsafe) * self._goal_reward_multiplier * np.sum(
            -1. * self._time_step_cost *
            np.array([self._gamma ** i for i in range(self._reward_horizon)]))

        return ts_cost + unsafe_cost + goal_reward

    def _write_episode_row_and_increment_episode(self, id_, episode, write_episode_op):
        write_episode_row = self._get_rows_for_id(episode)
        write_episode_index_op = self._episode_end_indices_table.write(write_episode_row, id_)
        self._increment_episode()

        return tf.group(write_episode_op, write_episode_index_op)

    def _add_batch(self, items: Trajectory):
        super(HindsightExperienceReplayBuffer, self)._add_batch(items)
        id_ = self._get_last_id()
        episode = self._get_current_episode()
        write_row = self._get_rows_for_id(id_)
        write_episode_op = self._episode_table.write(write_row, episode)

        return tf.cond(
            pred=tf.reduce_any(items.is_boundary()[0]),
            true_fn=lambda: self._write_episode_row_and_increment_episode(id_, episode, write_episode_op),
            false_fn=lambda: tf.group(write_episode_op))

    def _sample(self, num_steps, sample_batch_size, ids, batch_offsets, sample_goals: bool = False):
        step_range = tf.range(num_steps, dtype=tf.int64)
        if sample_batch_size:
            step_range = tf.reshape(step_range, [1, num_steps])
            step_range = tf.tile(step_range, [sample_batch_size, 1])
            ids = tf.tile(tf.expand_dims(ids, -1), [1, num_steps])
            batch_offsets = batch_offsets[:, None]
        else:
            step_range = tf.reshape(step_range, [num_steps])

        rows_to_get = tf.math.mod(
            step_range + ids, self._max_length
        ) + batch_offsets
        data = self._data_table.read(rows_to_get)
        data_ids = self._id_table.read(rows_to_get)

        if sample_goals:
            # get the row of the first state of the transition
            episode = self._episode_table.read(rows_to_get[..., 0])
            # get the index of the final state of this episode
            episode_end_indices = self._episode_end_indices_table.read(episode)
            if self.sampling_strategy is SamplingStrategy.FINAL:
                goal_indices = episode_end_indices
            elif self.sampling_strategy is SamplingStrategy.FUTURE:
                goal_indices = tf.cast(
                    tf.random.uniform(shape=(sample_batch_size, ) if sample_batch_size else (), dtype=tf.float64)
                    * tf.cast(episode_end_indices + 1 - ids[..., 0], tf.float64)
                    + tf.cast(ids[..., 0], tf.float64),
                    tf.int64)
            else:
                raise NotImplementedError

            future_rows_to_get = self._get_rows_for_id(goal_indices)
            future_data = self._data_table.read(future_rows_to_get)
            future_goals = self._env.state_to_goal(future_data.observation, batched=True, lib=tf)
            stacked_future_goals = tf.stack([future_goals, future_goals], axis=1)

            data = self._replace_trajectory_goal_and_reward(traj=data, goal=stacked_future_goals)

        else:
            goals = data.observation['goal']
            data = self._replace_trajectory_goal_and_reward(
                traj=data, goal=goals, batched=sample_batch_size is not None)

        return data, data_ids

    def _replace_trajectory_goal_and_reward(self, traj: Trajectory, goal, batched=True):
        if batched:
            rewards = tf.stack(
                [self.reward_fn(traj.observation, goal)[:, 1, ...],
                 traj.reward[:, 1, ...]],
                axis=1)
        else:
            rewards = tf.stack(
                [self.reward_fn(traj.observation, goal)[1, ...],
                 traj.reward[1, ...]],
                axis=0)
        return traj.replace(
            observation={
                key: value if key != 'goal' else goal
                for key, value in traj.observation.items()
            },
            reward=rewards)

    def _get_next(
            self,
            sample_batch_size=None,
            num_steps=None,
            time_stacked=True
    ):
        # index of the last terminated episode
        with tf.device(self._device), tf.name_scope(self._scope):
            with tf.name_scope('get_next'):
                end_id = self._episode_end_indices_table.read(self._get_current_episode() - 1)
                min_val, max_val = tf_agents.replay_buffers.tf_uniform_replay_buffer._valid_range_ids(
                    end_id, self._max_length, num_steps)
                rows_shape = () if sample_batch_size is None else (sample_batch_size,)
                assert_nonempty = tf.compat.v1.assert_greater(
                    max_val,
                    min_val,
                    message='HindsightExperienceReplayBuffer is empty. Make sure to add items '
                            'before sampling the buffer.')
                with tf.control_dependencies([assert_nonempty]):
                    num_ids = max_val - min_val
                    probability = tf.cond(
                        pred=tf.equal(num_ids, 0),
                        true_fn=lambda: 0.,
                        false_fn=lambda: 1. / tf.cast(num_ids * self._batch_size,  # pylint: disable=g-long-lambda
                                                      tf.float32))
                    ids = tf.random.uniform(
                        rows_shape, minval=min_val, maxval=max_val, dtype=tf.int64)

                    if sample_batch_size is None:
                        her_batch_size = 1
                        her_ids = ids
                        replay_ids = ids
                    else:
                        her_batch_size = int(sample_batch_size * self._her_ratio)
                        her_ids = ids[:her_batch_size]
                        replay_ids = ids[her_batch_size:]

                # Move each id sample to a random batch.
                batch_offsets = tf.random.uniform(
                    rows_shape, minval=0, maxval=self._batch_size, dtype=tf.int64)
                batch_offsets *= self._max_length

                if num_steps is None:
                    rows_to_get = tf.math.mod(ids, self._max_length) + batch_offsets
                    data = self._data_table.read(rows_to_get)
                    data_ids = self._id_table.read(rows_to_get)
                else:
                    if time_stacked:
                        if sample_batch_size is None:
                            hindsight = tf.random.uniform(shape=()) <= self._her_ratio
                            data, data_ids = self._sample(
                                num_steps=num_steps,
                                sample_batch_size=1,
                                ids=ids[None],
                                batch_offsets=batch_offsets[None],
                                sample_goals=hindsight)
                            data = squeeze_trajectory(data, axis=0)
                            data_ids = squeeze_trajectory(data_ids, axis=0)
                        else:
                            replay_data, replay_data_ids = self._sample(
                                num_steps=num_steps,
                                sample_batch_size=sample_batch_size - her_batch_size,
                                ids=replay_ids,
                                batch_offsets=batch_offsets[her_batch_size:])
                            her_data, her_data_ids = self._sample(
                                num_steps=num_steps,
                                sample_batch_size=her_batch_size,
                                ids=her_ids,
                                batch_offsets=batch_offsets[:her_batch_size],
                                sample_goals=True)
                            data = concatenate_trajectory(replay_data, her_data)
                            data_ids = tf.concat([replay_data_ids, her_data_ids], axis=0)
                    else:
                        data = []
                        data_ids = []
                        for step in range(num_steps):
                            steps_to_get = tf.math.mod(ids + step,
                                                       self._max_length) + batch_offsets
                            items = self._data_table.read(steps_to_get)
                            data.append(items)
                            data_ids.append(self._id_table.read(steps_to_get))
                        data = tuple(data)
                        data_ids = tuple(data_ids)
                probabilities = tf.fill(rows_shape, probability)

                buffer_info = tf_uniform_replay_buffer.BufferInfo(ids=data_ids, probabilities=probabilities)
        return data, buffer_info

    def _get_current_episode(self):
        def last_episode():
            return self._episode.value()

        return self._episode_cs.execute(last_episode)

    def _increment_episode(self, increment=1):
        """Increments the last_id in a thread safe manner.

        Args:
          increment: amount to increment last_id by.
        Returns:
          An op that increments the last_id.
        """

        def _assign_add():
            return self._episode.assign_add(increment).value()

        return self._episode_cs.execute(_assign_add)


class HindsightExperienceReplay:
    """
    Implementation of Hindsight experience replay for GoalAugmentedEnvironments.
    """

    def __init__(
            self,
            replay_buffer_add_batch_fn: Callable[[ts.TimeStep], Any],
            goal_augmented_env: Union[GoalAugmentedEnvironment, PyEnv],
            batch_size: int = 1,
            sampling_strategy: Optional[SamplingStrategy] = SamplingStrategy.FINAL,
            n_future_samples: int = 4,
            gamma: float = 1.,
            time_step_cost: float = 1.,
            state_penalty_multiplier: float = 1.,
            goal_reward_multiplier: float = 0.,
            reward_horizon: int = 100,
    ):
        self._add_to_replay_buffer = replay_buffer_add_batch_fn
        self._env = goal_augmented_env
        self.sampling_strategy = sampling_strategy
        self._current_episode = []
        self._batch_size = batch_size
        assert self._batch_size == 1, "Batch size > 1 is currently not supported."
        self.n_future_samples = n_future_samples
        self._gamma = gamma
        self._time_step_cost = -1. * np.abs(time_step_cost)
        assert self._time_step_cost < 0., "The non-final time step cost should be non-nul"
        self._state_penalty_multiplier = state_penalty_multiplier
        self._goal_reward_multiplier = goal_reward_multiplier
        self._reward_horizon = reward_horizon

    def __call__(self, trajectory: Trajectory, *args, **kwargs):
        self._current_episode.append(trajectory)
        if trajectory.is_boundary()[0]:
            self.process_episode()

    def reward_fn(self, traj: Trajectory, goal: Union[ArrayLike, tf.Tensor]) -> Float:
        is_achieved = tf.cast(
            self._env.is_achieved(traj.observation, goal, batched=True, lib=tf),
            tf.float32)
        is_unsafe = tf.cast(self._env.is_unsafe(traj.observation, lib=tf), tf.float32)
        ts_cost = (1. - is_achieved) * (1. - is_unsafe) * self._time_step_cost
        unsafe_cost = is_unsafe * self._state_penalty_multiplier * np.sum(
            self._time_step_cost *
            np.array([self._gamma ** i for i in range(self._reward_horizon)]))
        goal_reward = is_achieved * (1. - is_unsafe) * self._goal_reward_multiplier * np.sum(
            -1. * self._time_step_cost *
            np.array([self._gamma ** i for i in range(self._reward_horizon)]))

        return ts_cost + unsafe_cost + goal_reward

    def add_to_rb(self, traj: Trajectory, sampled_goal: tf.Tensor, next_traj: Optional[Trajectory] = None):
        self._add_to_replay_buffer(
            traj.replace(
                observation={
                    key: value if key != 'goal' else sampled_goal
                    for key, value in traj.observation.items()
                },
                reward=(self.reward_fn(next_traj, sampled_goal)
                        if next_traj is not None else
                        tf.zeros_like(traj.reward))))

    def process_episode(self):
        episode = self._current_episode
        self._current_episode = []
        episode_goal = episode[-1].observation['goal']

        for i, traj in enumerate(episode):
            if i < len(episode) - 1:
                sparse_rew_traj = traj.replace(reward=self.reward_fn(episode[i + 1], episode_goal))
            else:
                sparse_rew_traj = traj.replace(reward=tf.zeros_like(traj.reward))
            self._add_to_replay_buffer(sparse_rew_traj)

        safe_indices = [i for i in range(len(episode)) if not self._env.is_unsafe(episode[i].observation, lib=tf)]
        if self.sampling_strategy is SamplingStrategy.FINAL:
            sampled_goal = self._env.state_to_goal(episode[safe_indices[-1]].observation)
            for i, traj in enumerate(episode):
                self.add_to_rb(
                    traj=traj,
                    next_traj=episode[i + 1] if i < len(episode) - 1 else None,
                    sampled_goal=sampled_goal)
        elif self.sampling_strategy is SamplingStrategy.FUTURE:
            for _ in range(self.n_future_samples):
                for i, traj in enumerate(episode):
                    future_indices = [index for index in safe_indices if index > i]
                    if future_indices:
                        j = future_indices[tf.random.uniform(shape=(), maxval=len(future_indices), dtype=tf.int64)]
                    else:
                        j = safe_indices[-1]
                    future_goal = self._env.state_to_goal(episode[j].observation)
                    self.add_to_rb(
                        traj=traj,
                        next_traj=episode[i + 1] if i < len(episode) - 1 else None,
                        sampled_goal=future_goal)

def concatenate_trajectory(trajectory1: Trajectory, trajectory2: Trajectory):
    flat_trajectory1 = tf.nest.flatten(trajectory1)
    flat_trajectory2 = tf.nest.flatten(trajectory2)
    concatenated_flat_trajectory = [tf.concat([x, y], axis=0) for x, y in zip(flat_trajectory1, flat_trajectory2)]
    return tf.nest.pack_sequence_as(structure=trajectory1, flat_sequence=concatenated_flat_trajectory)

def squeeze_trajectory(trajectory: Trajectory, axis=0):
    flat_trajectory = tf.nest.flatten(trajectory)
    squeezed_flat_trajectory = [tf.squeeze(x, axis=0) for x in flat_trajectory]
    return tf.nest.pack_sequence_as(structure=trajectory, flat_sequence=squeezed_flat_trajectory)