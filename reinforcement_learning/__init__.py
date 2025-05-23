from typing import Dict, Callable

import gym.envs
import tensorflow as tf
import math

from tf_agents.typing.types import Bool


@tf.function
def lunar_lander_labels(s):
    """
    from LunarLander heuristic
    """
    labels = []
    angle_targ = s[..., 0] * 0.5 + s[..., 2] * 1.0  # angle should point towards center
    labels.append(tf.logical_or(angle_targ > 0.4,  # more than 0.4 radians (22 degrees) is bad
                                angle_targ < -0.4))
    angle_targ = tf.map_fn(lambda x: tf.cond(x > 0.4, lambda: 0.4, lambda: x), angle_targ)
    angle_targ = tf.map_fn(lambda x: tf.cond(x < -.4, lambda: -.4, lambda: x), angle_targ)
    hover_targ = 0.55 * tf.abs(s[..., 0])  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[..., 4]) * 0.5 - (s[..., 5]) * 1.0
    hover_todo = (hover_targ - s[..., 1]) * 0.5 - (s[..., 3]) * 0.5

    # legs contact
    labels.append(tf.logical_or(tf.cast(s[..., 6], dtype=tf.bool), tf.cast(s[..., 7], dtype=tf.bool)))
    legs_contact = (s[..., 6] + s[..., 7]) / tf.maximum(s[..., 6] + s[..., 7], 1.)
    angle_todo = angle_todo * (1. - legs_contact)
    # override to reduce fall speed, that's all we need after contact
    hover_todo = hover_todo * (1. - legs_contact) - .5 * s[..., 3] * legs_contact

    labels.append(tf.logical_and(hover_todo > tf.abs(angle_todo), hover_todo > 0.05))
    labels.append(angle_todo < -0.05)
    labels.append(angle_todo > 0.05)
    labels.append(tf.logical_and(s[..., 2] == 0.,  # horizontal speed is 0
                                 s[..., 3] == 0.))  # vertical speed is 0

    return tf.stack(labels, axis=-1)


labeling_functions = {
    'Humanoid-v3': lambda observation: tf.math.logical_and(1. < observation[..., 0], observation[..., 0] < 2.),
    'HumanoidBulletEnv-v0':
        lambda observation: tf.stack([
            # falling down -- observation[0] is the head position, 0.8 is the initial position
            observation[..., 0] + 0.8 <= 0.78,
            # has stuck joints
            # tf.expand_dims(tf.math.count_nonzero(tf.abs(observation[..., 8: 42][0::2]) > 0.99) > 0, axis=-1),
            # feet contact
            tf.cast(observation[..., -2], tf.bool), tf.cast(observation[..., -1], tf.bool),
            # move forward
            observation[..., 3] > 0.,
        ], axis=-1),
    'BipedalWalker-v2':
        lambda observation: tf.stack([
            tf.math.abs(observation[..., 0]) > math.pi / 3.,  # hull angle too high/low (unsafe flag)
            tf.cast(observation[..., 8], tf.bool),  # contact of the left leg with the ground
            tf.cast(observation[..., 13], tf.bool)  # contact of the right leg with the ground
        ], axis=-1),
    'Pendulum-v0':  # safe labels
        lambda observation: tf.stack([
            # cos(θ) >= cos(π / 3 rad) = cos(2 π - π / 3 rad) = cos(60°) = cos(-60°)
            observation[..., 0] >= tf.math.cos(math.pi / 3),
            # push direction
            observation[..., 2] >= 0,
            # cos(θ) >= 0, i.e., the pendulum is at the top of the screen
            observation[..., 0] >= 0.,
            # sin(θ) >= 0, i.e., the pendulum is on the left side of the screen
            observation[..., 1] >= 0.,
        ], axis=-1),
    'CartPole-v0':  # safe labels
        lambda observation: tf.stack([
            tf.abs(observation[..., 0]) < 1.5,  # cart position is less than 1.5
            tf.abs(observation[..., 2]) < 0.15,  # pole angle is inferior to 9 degrees
        ], axis=-1),
    'LunarLander-v2': lunar_lander_labels,
    'MountainCar-v0': lambda observation: tf.stack([
        observation[..., 0] >= 0.5,  # has reached the goal
        observation[..., 0] >= -.5,  # right-hand side -- positive slope
        observation[..., 1] >= 0.,  # is going forward
    ], axis=-1),
    'Acrobot-v1': lambda observation: tf.stack([
        (-1. * observation[..., 0] - observation[..., 2] * observation[..., 0] +
         observation[..., 3] * observation[..., 1] > 1.),  # objective
        observation[..., 0] >= 0.,  # cos of the first pendulum angle
        observation[..., 1] >= 0.,  # sin of the first pendulum angle
        observation[..., 2] >= 0.,  # cos of the second pendulum angle
        observation[..., 3] >= 0.,  # cos of the first pendulum angle
        observation[..., 4] >= 0.,  # angular velocity of the first pendulum
        observation[..., 5] >= 0.  # angular velocity of the second pendulum
    ], axis=-1),
    'Hopper-v3': lambda observation: tf.stack([
        # An element of `observation[1:] is contained in the closed interval specified by the environment
        # argument `healthy_state_range`
        tf.reduce_all(tf.logical_and(-100. < observation[..., 1:], observation[..., 1:] < 100.), axis=-1),
        # The height of the hopper is contained in the closed interval specified by the argument
        # `healthy_z_range` (usually meaning that it has fallen)
        observation[..., 0] > 0.7,
        # The angle is contained in the closed interval specified by the argument `healthy_angle_range`
        tf.logical_and(-0.2 < observation[..., 1], observation[..., 1] < 0.2)
    ], axis=-1),
    'Walker2d-v3': lambda observation: tf.stack([
        # The height of the walker is in the closed interval specified by `healthy_z_range`
        tf.logical_and(0.8 < observation[..., 0], observation[..., 0] < 2.),
        # The absolute value of the angle is in the closed interval specified by `healthy_angle_range`
        tf.abs(observation[..., 1]) < 1.,
    ], axis=-1),
    'HalfCheetah-v3': lambda observation: tf.stack([
        # angles
        observation[..., 1] < 0.,
        observation[..., 2] < 0.,
        observation[..., 3] < 0.,
    ], axis=-1),
    'Ant-v3': lambda observation: tf.stack([
        # Any of the state space values is no longer finite
        tf.reduce_all(tf.math.is_finite(observation), axis=-1),
        # The z-coordinate of the torso is in the closed interval given by `healthy_z_range`
        # (defaults to [0.2, 1.0])
        tf.logical_and(0.2 < observation[..., 0], observation[..., 0] < 1.)
    ], axis=-1),
    "Swimmer-v3": lambda observation: tf.stack([
        # angles
        observation[..., 0] < 0.,
        observation[..., 1] < 0.,
        observation[..., 2] < 0.
    ], axis=-1),
    "finger": lambda observation: tf.stack([
        # touch top
        observation[..., 4] > 0.,
        # touch bottom
        observation[..., 5] > 0.,
        # target velocity (spin) -- should always stay at this velocity once reached
        observation[..., -1] <= -15.,
    ], axis=-1),
    'cheetah': lambda observation: tf.stack([
        # low speed -- should always stay above this threshold once overtaken
        observation[..., 8] > 3.,
        # high speed -- remaining above this speed is great
        observation[..., 8] > 7.,
        # very high speed -- target speed
        observation[..., 8] > 10.,
    ], axis=-1),
    'walker': lambda observation: tf.stack([
        # stand up -- should eventually remain above the threshold
        observation[..., 0] >= 1.2,
        # walk speed threshold
        observation[..., 16] >= 1.,
    ], axis=-1),
    'reacher': lambda observation: tf.stack([
        # distance to target easy
        tf.norm(observation[..., 2:4], axis=-1) <= 0.05,
        # distance to target hard
        tf.norm(observation[..., 2:4], axis=-1) <= 0.015,
    ], axis=-1),
    'GoalOrientedPacmanRoom-v0': lambda observation: tf.nest.flatten(observation)[1],
    'PacmanGrid-v0': lambda observation: tf.nest.flatten(observation)[0],
    # 'SingleCleaner-v0': lambda observation: tf.zeros(shape=(tf.shape(tf.nest.flatten(observation)[0])[0], 0), dtype=tf.bool),
    'DoomHighLevel-v0': lambda observation: tf.nest.flatten(observation)[1],
}


reward_scaling = {
    'Pendulum-v0': 1. / (2 * (math.pi ** 2 + 0.1 * 8 ** 2 + 0.001 * 2 ** 2)),
    'CartPole-v0': 1. / 2,
    # 'LunarLander-v2': 1. / 400,
    'MountainCar-v0': 1. / 2,
    # 'Acrobot-v1': 1. / 2,
    # 'BipedalWalker-v2': 1. / 200.,
}  # scale the rewards in [-1./2, 1./2]

for d in [labeling_functions, reward_scaling]:
    for key, value in {
        'LunarLanderContinuous-v2': 'LunarLander-v2',
        'LunarLanderNoRewardShaping-v2': 'LunarLander-v2',
        'LunarLanderRandomInit-v2': 'LunarLander-v2',
        'LunarLanderContinuousRandomInit-v2': 'LunarLander-v2',
        'LunarLanderContinuousRandomInitNoRewardShaping-v2': 'LunarLander-v2',
        'LunarLanderRewardShapingAugmented-v2': 'LunarLander-v2',
        'LunarLanderRandomInitRewardShapingAugmented-v2': 'LunarLander-v2',
        'LunarLanderRandomInitNoRewardShaping-v2': 'LunarLander-v2',
        'LunarLanderContinuousRewardShapingAugmented-v2': 'LunarLander-v2',
        'LunarLanderContinuousRandomInitRewardShapingAugmented-v2': 'LunarLander-v2',
        'MountainCarContinuous-v0': 'MountainCar-v0',
        'PendulumRandomInit-v0': 'Pendulum-v0',
        'AcrobotRandomInit-v1': 'Acrobot-v1',
        'Pendulum-v1': 'Pendulum-v0',
        'PendulumRandomInit-v1': 'Pendulum-v1',
        'BipedalWalker-v3': 'BipedalWalker-v2',
        'GoalOrientedPacmanRoomSimple-v0': 'GoalOrientedPacmanRoom-v0',
        'PacmanRoom-v0': 'GoalOrientedPacmanRoom-v0',
        'PacmanGridEast-v0': 'PacmanGrid-v0',
        'PacmanGridWest-v0': 'PacmanGrid-v0',
        'PacmanGridNorth-v0': 'PacmanGrid-v0',
        'PacmanGridSouth-v0': 'PacmanGrid-v0',
        'PacmanGridEasy-v0': 'PacmanGrid-v0',
        'PacmanGridEasyEast-v0': 'PacmanGrid-v0',
        'PacmanGridEasyWest-v0': 'PacmanGrid-v0',
        'PacmanGridEasyNorth-v0': 'PacmanGrid-v0',
        'PacmanGridEasySouth-v0': 'PacmanGrid-v0',
        'PacmanGridHighLevel-v0': 'PacmanGrid-v0',
        'DoomRoom-v0': 'DoomHighLevel-v0',
        'DoomMediumHighLevel-v0': 'DoomHighLevel-v0',
        'DoomHardHighLevel-v0': 'DoomHighLevel-v0',
        'DoomRoomNorth-v0': 'DoomHighLevel-v0',
        'DoomRoomEast-v0': 'DoomHighLevel-v0',
        'DoomRoomWest-v0': 'DoomHighLevel-v0',
        'DoomRoomSouth-v0': 'DoomHighLevel-v0',
        'DoomNorth-v0': 'DoomHighLevel-v0',
        'DoomEast-v0': 'DoomHighLevel-v0',
        'DoomWest-v0': 'DoomHighLevel-v0',
        'DoomSouth-v0': 'DoomHighLevel-v0',
        'DoomMediumNorth-v0': 'DoomHighLevel-v0',
        'DoomMediumEast-v0': 'DoomHighLevel-v0',
        'DoomMediumWest-v0': 'DoomHighLevel-v0',
        'DoomMediumSouth-v0': 'DoomHighLevel-v0',
        'DoomHardNorth-v0': 'DoomHighLevel-v0',
        'DoomHardEast-v0': 'DoomHighLevel-v0',
        'DoomHardWest-v0': 'DoomHighLevel-v0',
        'DoomHardSouth-v0': 'DoomHighLevel-v0',
    }.items():
        if value in d.keys():
            d[key] = d[value]
