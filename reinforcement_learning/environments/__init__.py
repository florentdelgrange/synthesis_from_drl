import os
import sys
from typing import Optional, List

import tensorflow as tf
#from tf_agents.typing.types import Sequence, PyEnvWrapper
from gym.envs.registration import register
from tf_agents.environments.wrappers import HistoryWrapper
from tf_agents.environments import FlattenObservationsWrapper

from reinforcement_learning.environments.two_level_env import Directions

register(
    id='LunarLanderNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuousNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderRandomInitNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRandomInitNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuousRandomInitNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousRandomInitNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderRandomInit-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRandomInit',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderContinuousRandomInit-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousRandomInit',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderContinuousRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderContinuousRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderRandomInitRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRandomInitRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderContinuousRandomInitRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:'
                'LunarLanderContinuousRandomInitRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='PendulumRandomInit-v1',
    entry_point='reinforcement_learning.environments.pendulum:PendulumRandomInit',
    max_episode_steps=150,
)

register(
    id='AcrobotRandomInit-v1',
    entry_point='reinforcement_learning.environments.acrobot:AcrobotEnvRandomInit',
    reward_threshold=-100.0,
    max_episode_steps=500,
)

register(
    id='PacmanRoom-v0',
    entry_point='reinforcement_learning.environments.pacman:PacmanEnv',
    max_episode_steps=200,
    kwargs={'gamma': .99}
)

for eval in ['', 'Eval']:
    register(
        id=f'PacmanGrid{eval}-v0',
        entry_point=f'reinforcement_learning.environments.pacman:PacmanEnv{eval}',
        max_episode_steps=200,
        kwargs={
            'grid_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', f'grid_easy{eval}.txt'),
            'adversarial_ghosts_entropy': 3.25e-2,
            'include_pacman_position': True,
            'p_cherry': 1./100,
        }
    )
    if eval:
        register(
            id=f'PacmanGridHighLevel-v0',
            entry_point=f'reinforcement_learning.environments.pacman:PacmanEnv{eval}',
            kwargs={
                'grid_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', f'grid_easy{eval}.txt'),
                'adversarial_ghosts_entropy': 3.25e-2,
                'include_pacman_position': True,
                'p_cherry': 1. / 100,
            }
        )
        register(
            id=f'PacmanGrid{eval}EasyHL-v0',
            entry_point=f'reinforcement_learning.environments.pacman:HighLevelEnvironment',
            max_episode_steps=400,
            kwargs={
                'grid_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', f'grid_easy{eval}.txt'),
                'adversarial_ghosts_entropy': 3.25e-2,
                'include_pacman_position': True,
                'p_cherry': 1. / 100,
            }
        )

register(
    id='PacmanGridNorth-v0',
    entry_point='reinforcement_learning.environments.pacman:PacmanEnvironment',
    max_episode_steps=200,
    kwargs={
        'grid_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', 'grid_easy.txt'),
        'adversarial_ghosts_entropy': 3.25e-2,
        'include_pacman_position': True,
        'direction': 3,
        'p_cherry': 1/100
    }
)
register(
    id='PacmanGridSouth-v0',
    entry_point='reinforcement_learning.environments.pacman:PacmanEnvironment',
    max_episode_steps=200,
    kwargs={
        'grid_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', 'grid_easy.txt'),
        'adversarial_ghosts_entropy': 3.25e-2,
        'include_pacman_position': True,
        'direction': 1,
        'p_cherry': 1/100
    }
)
register(
    id='PacmanGridEast-v0',
    entry_point='reinforcement_learning.environments.pacman:PacmanEnvironment',
    max_episode_steps=200,
    kwargs={
        'grid_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', 'grid_easy.txt'),
        'adversarial_ghosts_entropy': 3.25e-2,
        'include_pacman_position': True,
        'direction': 0,
        'p_cherry': 1/100
    }
)
register(
    id='PacmanGridWest-v0',
    entry_point='reinforcement_learning.environments.pacman:PacmanEnvironment',
    max_episode_steps=200,
    kwargs={
        'grid_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', 'grid_easy.txt'),
        'adversarial_ghosts_entropy': 3.25e-2,
        'include_pacman_position': True,
        'direction': 2,
        'p_cherry': 1/100
    }
)
register(
    id='GoalOrientedPacmanRoom-v0',
    entry_point='reinforcement_learning.environments.pacman.goal_oriented:GoalOrientedPacmanEnv',
    max_episode_steps=200,
    kwargs={
        'maps': [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pacman', 'map.txt')],
        'gamma': .99}
)

for difficulty in ['', 'medium', 'hard']:
    register(
        id=f'Doom{difficulty.capitalize()}HighLevel-v0',
        entry_point='reinforcement_learning.environments.doom.doom_maze:DoomMaze',
        max_episode_steps=20000,
        kwargs={
            'mode': 'eval',
            'render_mode': 'rgb_array',
            'difficulty': difficulty if difficulty else 'normal'
        }
    )

    for direction in [d for d in Directions if d != Directions.NOOP]:
        register(
            id=f'Doom{difficulty.capitalize()}Room{direction.to_cardinal().capitalize()}-v0',
            entry_point='reinforcement_learning.environments.doom.doom_maze:DoomMaze',
            max_episode_steps=1000,
            kwargs={
                'mode': 'room',
                'render_mode': 'rgb_array',
                'direction': direction,
                'difficulty': difficulty if difficulty else 'normal'
            }
        )
        register(
            id=f'Doom{difficulty.capitalize()}{direction.to_cardinal().capitalize()}-v0',
            entry_point='reinforcement_learning.environments.doom.doom_maze:DoomMaze',
            max_episode_steps=1000,
            kwargs={
                'mode': 'room',
                'render_mode': 'rgb_array',
                'direction': direction,
                'difficulty': difficulty if difficulty else 'normal'
            }
        )
register(
    id='DoomRoom-v0',
    entry_point='reinforcement_learning.environments.doom.doom_maze:DoomMaze',
    max_episode_steps=1000,
    kwargs={
        'mode': 'room',
        'render_mode': 'rgb_array',
    }
)

try:
    import jax
    import jumanji

    for env_name in jumanji.registered_environments():
        register(
            id=env_name,
            entry_point=f'reinforcement_learning.environments.jumanji:JumanjiEnv',
            kwargs={
                'env_name': env_name
            }
        )
    register(
        id='SingleCleaner-v0',
        entry_point=f'reinforcement_learning.environments.jumanji:SingleCleaner',
    )
    register(
        id='SingleCleaner-v1',
        entry_point=f'reinforcement_learning.environments.jumanji:SingleCleanerV2',
    )
except ImportError as ie:
    pass

class EnvironmentLoader:
    def __init__(
            self,
            environment_suite,
            seed=None,
            time_stacked_states=1,
            env_args: Optional[List[str]] = None,
            flatten: bool = False,
    ):
        self.n = 0
        self.environment_suite = environment_suite
        self.seed = seed
        self.time_stacked_states = time_stacked_states
        self.env_args = env_args if env_args is not None else []
        self._flatten = flatten

    def load(self, env_name: str, env_wrappers=(), **env_kwargs):
        if self.time_stacked_states > 1:
            env_wrappers = list(env_wrappers) + \
                           [lambda env: HistoryWrapper(env=env, history_length=self.time_stacked_states)]
        environment = self.environment_suite.load(
            *([env_name] + self.env_args), env_wrappers=env_wrappers, **env_kwargs)

        if self._flatten and len(tf.nest.flatten(environment.observation_spec())) > 1:
            del environment
            return self.load(env_name, env_wrappers=[FlattenObservationsWrapper] + list(env_wrappers))

        if self.seed is not None:
            try:
                environment.seed(self.seed + self.n)
                self.n += 1
            except (NotImplementedError, AttributeError):
                print("Environment {} has no seed support.".format(env_name))
        return environment
