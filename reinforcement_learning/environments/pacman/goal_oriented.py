from collections import OrderedDict
from typing import Collection, Tuple, Optional, Union, Dict, List

import gym
import numpy as np
# from gym.core import ActType, ObsType
from dm_env import StepType
from gym.vector.utils import spaces
from numpy.typing import ArrayLike
import tensorflow as tf

import os
import sys

from tf_agents.typing.types import Float

from reinforcement_learning.goal_conditioned import HindsightExperienceReplay, SamplingStrategy

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../../..')

from reinforcement_learning.environments.pacman import PacmanEnv
from reinforcement_learning.environments.pacman.pacman_env import DIM, MapItems, MapIndices
from reinforcement_learning.environments.goal_augmented import GoalAugmentedEnvironment


class GoalOrientedPacmanEnv(GoalAugmentedEnvironment):
    """
    Goal oriented environment where the observation space is augmented with
    the Map space and the Goal space.
    Furthermore, the observation is split in two: Pacman observation and Ghosts observation.
    Therefore, an observation is a dict where the keys have the form:
        { pacman_observation, ghost_observation, map, goal positions, [optional: label] }.
    If the flag `include_label` is set, then the label of the current observation is added to the
    observation space.
    Drawing the initial observation by calling the reset operator of the environment exactly corresponds to
        1) sampling a map from the Map space,
        2) sampling a set of goal positions from the Goal space (depending on the current map), and
        3) sampling an initial state from the resulting environment.
    Note that sampling is done uniformly in the map and goal space.


    """

    @property
    def current_goal(self):
        return self._current_goal

    def state_to_goal(
            self,
            state: Union[np.typing.ArrayLike, tf.Tensor],
            batched: bool = False,
            lib=np
    ) -> Union[np.typing.ArrayLike, tf.Tensor]:
        if self._split_adversary_observations:
            x = lib.where(state['pacman_observation'][..., 0])
        else:
            x = lib.where(state['observation'][..., 0])
        if lib == np:
            x = lib.transpose(x).astype(np.float32)
        elif lib == tf:
            x = tf.cast(x, dtype=tf.float32)
        else:
            raise NotImplementedError
        if batched:
            return x[..., 1:]
        else:
            return x

    def is_safe(self, state: np.typing.ArrayLike, batched: bool = False, lib=np) -> Union[np.typing.ArrayLike, Float]:
        if lib == np:
            all = np.all
        elif lib == tf:
            all = tf.reduce_all
        else:
            raise NotImplementedError
        if self._split_adversary_observations:
            return all(lib.logical_not(
                lib.logical_and(
                    state['pacman_observation'][..., 0],  # first position is pacman
                    state['ghosts_observation'][..., 0])),  # first position is the ghost
                axis=(1, 2) if batched else None)
        else:
            return all(lib.logical_not(
                lib.logical_and(
                    state['observation'][..., 0],  # 0 is pacman
                    state['observation'][..., 1])),  # 1 is the ghost
                axis=(1, 2) if batched else None)

    def is_achieved(self, state: np.typing.ArrayLike, goal: np.typing.ArrayLike, batched: bool = False, lib=np):
        if lib == np:
            goal = goal.astype(np.float32)
        elif lib == tf:
            goal = tf.cast(goal, tf.float32)
        else:
            raise NotImplementedError
        return lib.linalg.norm(self.state_to_goal(state, batched=batched, lib=lib) - goal, ord=1, axis=-1) < 1e-5

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            maps: List[str],
            include_label: bool = True,
            split_adversary_observations: bool = False,
            *args, **kwargs
    ):
        self._include_label = include_label
        self._split_adversary_observations = split_adversary_observations
        self._maps = maps
        sample = self._sample()
        self._wrapped_env = PacmanEnv(
            map_path=sample['map'],
            *args, **kwargs)
        self.reset()

        if split_adversary_observations:
            obs_spaces = {
                # Full observation excluding ghosts (i.e., map, items, and pacman)
                'pacman_observation': spaces.Box(
                    low=0,
                    high=DIM * DIM,
                    shape=(3,),
                    dtype=np.int32),
                # Full observation excluding PACMAN
                'ghosts_observation': spaces.Box(
                    low=0,
                    high=DIM * DIM,
                    shape=(DIM, DIM, 3,),
                    dtype=np.int32),
            }
        else:
            obs_spaces = {
                # Full observation
                'observation': spaces.Box(
                    low=0,
                    high=DIM * DIM,
                    shape=(DIM, DIM, 4,),
                    dtype=np.int32),
            }
        # goal position
        obs_spaces['goal'] = spaces.Box(
            low=0,
            high=DIM,
            shape=(2,),
            dtype=np.int32)

        if self._include_label:
            label_size = len(self._wrapped_env.labeling_fn(obs=self._wrapped_env.reset(map_file_name=sample['map'])))
            obs_spaces = {
                **obs_spaces,
                **{'label': spaces.Box(
                    low=0,
                    high=1,
                    shape=(label_size,), )}
            }

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = self._wrapped_env.action_space

    def labeling_fn(self, full_observation: ArrayLike):
        return self._wrapped_env.labeling_fn(full_observation)

    def _sample(self):
        return {
            'map': np.random.choice(self._maps),
        }

    def _get_observation(self, full_observation: ArrayLike):
        if self._split_adversary_observations:
            observation = OrderedDict({
                'pacman_observation': np.concatenate([
                    full_observation[..., MapIndices[item]][..., None]
                    for item in [MapItems.PACMAN, MapItems.WALL, MapItems.DOOR]  # , MapItems.BULLET, MapItems.CHERRY]
                ], axis=-1),
                'ghosts_observation': np.concatenate([
                    full_observation[..., MapIndices[item]][..., None]
                    for item in [MapItems.GHOST, MapItems.WALL, MapItems.DOOR]  # , MapItems.BULLET, MapItems.CHERRY]
                ], axis=-1),
            })
        else:
            observation = {
                'observation': np.concatenate([
                    full_observation[..., MapIndices[item]][..., None]
                    for item in [MapItems.PACMAN, MapItems.GHOST, MapItems.WALL, MapItems.DOOR]
                    # , MapItems.BULLET, MapItems.CHERRY]
                ], axis=-1)}

        observation['goal'] = self._current_goal

        if self._include_label:
            observation['label'] = self.labeling_fn(full_observation)

        return observation

    def step(self, action):
        obs, rew, done, info = self._wrapped_env.step(action)
        return self._get_observation(obs), rew, done, info

    def reset(
            self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None
    ):
        sample = self._sample()
        obs = self._wrapped_env.reset(map_file_name=sample['map'])
        self._current_map = sample['map']
        self._current_goal = self._wrapped_env.goal_positions[np.random.choice(len(self._wrapped_env.goal_positions))]
        self._wrapped_env.goal_positions = self._current_goal[None]
        return self._get_observation(obs)

    def render(self, mode="human", close=False):
        return self._wrapped_env.render(mode=mode, close=close)


if __name__ == '__main__':
    import reinforcement_learning.environments

    from tf_agents.environments import suite_gym, tf_py_environment, GoalReplayEnvWrapper
    from tf_agents.policies import random_tf_policy
    from tf_agents.drivers import dynamic_episode_driver
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import trajectory
    from reinforcement_learning.environments.perturbed_env import PerturbedEnvironment

    env_kwargs = {
        'split_adversary_observations': False
    }
    with suite_gym.load(
            environment_name="GoalOrientedPacmanRoom-v0",
            gym_kwargs=env_kwargs,
    ) as py_env:
        """
        done=False
        py_env.reset()
        py_env.render(mode='human')
        while not done:
            action = input()
            action = {
                'z': 2,
                's': 0,
                'q': 3,
                'd': 1,
                'x': 4,
            }[action[0]]
            step = py_env.step(action)
            done = step.step_type == StepType.LAST
            py_env.render(mode='human')
            print("label:", step.observation['label'], "position", py_env.state_to_goal(step.observation))
            # print("state", step.observation)
        # py_env.render(mode='human')
        print("Done!")
        """


        #"""

        # py_env = PerturbedEnvironment(env=py_env, perturbation=1. / 3)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        tf_policy = random_tf_policy.RandomTFPolicy(
            action_spec=tf_env.action_spec(),
            time_step_spec=tf_env.time_step_spec())

        rb = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=trajectory.from_transition(
                time_step=tf_env.time_step_spec(),
                action_step=tf_policy.policy_step_spec,
                next_time_step=tf_env.time_step_spec()),
            batch_size=tf_env.batch_size,
            max_length=1000)

        her = HindsightExperienceReplay(
            replay_buffer_add_batch_fn=rb.add_batch,
            goal_augmented_env=py_env,
            sampling_strategy=SamplingStrategy.FUTURE)

        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            env=tf_env,
            policy=tf_policy,
            observers=[
                # lambda _: py_env.render(mode='human'),
                # lambda traj: tf.print("observation", traj.observation),
                her,
            ],
            num_episodes=10,
        ).run()
        #"""
        pass
