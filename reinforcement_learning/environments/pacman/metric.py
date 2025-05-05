import tensorflow as tf
from tf_agents.metrics import tf_metric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.trajectories import Trajectory
from tf_agents.utils import common


class LabelMetric(tf_metric.TFStepMetric):

    def __init__(self,
                 name='LabelMetric',
                 prefix='Metrics',
                 dtype=tf.float32,
                 batch_size=1,
                 buffer_size=50):
        super(LabelMetric, self).__init__(name=name, prefix=prefix)
        self._buffer = [TFDeque(buffer_size, dtype), TFDeque(buffer_size, dtype)]
        self._dtype = dtype
        self._goal_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='GoalAccumulator')
        self._game_over_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='GameOverAccumulator')

    @common.function(autograph=True)
    def call(self, trajectory: Trajectory):
        # Zero out batch indices where a new episode is starting.
        self._goal_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._goal_accumulator),
                     self._goal_accumulator))
        self._game_over_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._game_over_accumulator),
                     self._game_over_accumulator))

        # Update accumulator with received label. We are summing over all
        # non-batch dimensions in case the reward is a vector.
        found = True
        if 'label' not in trajectory.observation and \
                'state' in trajectory.observation and 'label' in trajectory.observation['state']:
            label = trajectory.observation['state']['label']
        elif 'label' in trajectory.observation:
            label = trajectory.observation['label']
        else:
            found = False
        if found:
            self._goal_accumulator.assign_add(
                tf.reduce_sum(
                    tf.cast(label[..., 0, None], dtype=self._dtype),
                    axis=range(1, len(label.shape))))
            self._game_over_accumulator.assign_add(
                tf.reduce_sum(
                    tf.cast(label[..., 1, None], dtype=self._dtype),
                    axis=range(1, len(label.shape))))

            # Add final returns to buffer.
            last_episode_indices = tf.squeeze(tf.where(trajectory.is_boundary()), axis=-1)
            for indx in last_episode_indices:
                self._buffer[0].add(self._goal_accumulator[indx])
                self._buffer[1].add(self._game_over_accumulator[indx])

            return trajectory

    def result(self):
        return {'goal': self._buffer[0].mean(),
                'unsafe': self._buffer[1].mean()}

    @common.function
    def reset(self):
        for buf in self._buffer:
            buf.clear()
        self._goal_accumulator.assign(tf.zeros_like(self._goal_accumulator))
        self._game_over_accumulator.assign(tf.zeros_like(self._game_over_accumulator))
