import gym
from gym import spaces
import numpy as np
import jumanji.wrappers
from jumanji.environments.routing.cleaner.generator import RandomGenerator

from functools import cached_property
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.environments.routing.cleaner.constants import DIRTY, WALL
from jumanji.environments.routing.cleaner.env import Cleaner as CleanerOriginal
from jumanji.environments.routing.cleaner.generator import Generator
from jumanji.environments.routing.cleaner.types import Observation, State
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer

MOVES = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])  # Up, right, down, left, noop

class JumanjiEnv(gym.Wrapper):

    def __init__(self, env_name: str, env_kwargs=None):
        if env_kwargs is None:
            env_kwargs = dict()
        if env_name == 'SingleCleaner-v1':
            jumanji_env = Cleaner(**env_kwargs)
        else:
            jumanji_env = jumanji.make(env_name, **env_kwargs)
        env = jumanji.wrappers.JumanjiToGymWrapper(jumanji_env)
        self._spec = None
        super().__init__(env)
        observation_space = dict()
        for key, value in env.observation_space.items():
            observation_space[key] = spaces.Box(
                low=value.low,
                high=value.high,
                shape=value.shape,
                dtype=value.dtype if value.dtype != bool else np.uint8,
            )
        self.observation_space = spaces.Dict(observation_space)
        
    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        self._spec = value

    @property
    def unwrapped(self):
        return self


class SingleCleaner(JumanjiEnv):
    
    def __init__(self):
        super().__init__(
            env_name='Cleaner-v0',
            env_kwargs={'generator': RandomGenerator(
                num_rows=10, num_cols=10, num_agents=1
            )}
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'action_mask': spaces.Box(
                low=0, high=1, shape=(4, ), dtype=np.uint8),
            'agents_locations': spaces.Box(
                low=0, high=10, shape=(2, ), dtype=np.int32),
            'grid': spaces.Box(
                low=0, high=1, shape=(10, 10, 3), dtype=np.int32),
            'step_count': self.observation_space['step_count']
        })

    def process_observation(self, observation):
        observation['action_mask'] = np.squeeze(observation['action_mask'])
        observation['agents_locations'] = np.squeeze(observation['agents_locations'])
        grid = observation['grid']
        dirty_channel = np.where(grid == 0, 1, 0)
        clean_channel = np.where(grid == 1, 1, 0)
        wall_channel = np.where(grid == 2, 1, 0)
        observation['grid'] = np.stack([dirty_channel, clean_channel, wall_channel], axis=-1)
        return observation

    
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        return self.process_observation(observation)


    def step(self, action):
        action = np.array(action)[..., None]
        observation, reward, done, info = super().step(action)
        observation = self.process_observation(observation)
        return observation, reward, done, info


class Cleaner(CleanerOriginal):
    def __init__(
        self,
        generator: Optional[Generator] = None,
        time_limit: Optional[int] = None,
        penalty_per_timestep: float = 0.5,
        viewer: Optional[Viewer[State]] = None,
        allow_noop: bool = True,
        end_if_illegal: bool = False,
    ) -> None:
        self.allow_noop = allow_noop
        self.end_if_illegal = end_if_illegal
        super().__init__(generator, time_limit, penalty_per_timestep, viewer)

    @cached_property
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec.

        Returns:
            action_spec: a MultiDiscreteArray spec.
        """
        num_actions = 5 if self.allow_noop else 4
        return specs.MultiDiscreteArray(
            num_values=jnp.full(self.num_agents, num_actions, jnp.int32),
            dtype=jnp.int32,
            name="action_spec",
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the `Cleaner` environment.

        Returns:
            Spec for the `Observation`, consisting of the fields:
                - grid: BoundedArray (int8) of shape (num_rows, num_cols). Values
                    are between 0 and 2 (inclusive).
                - agent_locations_spec: BoundedArray (int32) of shape (num_agents, 2).
                    Maximum value for the first column is num_rows, and maximum value
                    for the second is num_cols.
                - action_mask: BoundedArray (bool) of shape (num_agent, 5).
                - step_count: BoundedArray (int32) of shape ().
        """
        num_actions = 5 if self.allow_noop else 4
        grid = specs.BoundedArray(self.grid_shape, jnp.int8, 0, 2, "grid")
        agents_locations = specs.BoundedArray((self.num_agents, 2), jnp.int32, [0, 0], self.grid_shape, "agents_locations")
        action_mask = specs.BoundedArray((self.num_agents, num_actions), bool, False, True, "action_mask")
        step_count = specs.BoundedArray((), jnp.int32, 0, self.time_limit, "step_count")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            agents_locations=agents_locations,
            action_mask=action_mask,
            step_count=step_count,
        )

    def _update_agents_locations(
        self,
        prev_locations: chex.Array,
        action: chex.Array,
        action_is_valid: chex.Array,
    ) -> chex.Array:
        """Update the agents locations according to the action taken.

        Args:
            prev_locations: array containing the x and y coordinates of every agent.
            action: array containing the action to take.
            action_is_valid: boolean array specifying, for each agent, which action
                is legal.

        Returns:
            agents_locations: array containing the x and y coordinates of every agent.
        """
        moves = jnp.where(action_is_valid[:, None], MOVES[action], 0)
        return prev_locations + moves

    def _compute_action_mask(self, grid: chex.Array, agents_locations: chex.Array) -> chex.Array:
        def is_move_valid(agent_location: chex.Array, move: chex.Array) -> chex.Array:
            y, x = agent_location + move
            return (x >= 0) & (x < self.num_rows) & (y >= 0) & (y < self.num_cols) & (grid[y, x] != WALL)

        # vmap over the moves and agents
        action_mask = jax.vmap(jax.vmap(is_move_valid, in_axes=(None, 0)), in_axes=(0, None))(
            agents_locations, MOVES if self.allow_noop else MOVES[:-1]
        )

        return action_mask

    def _should_terminate(self, state: State, valid_actions: chex.Array) -> chex.Array:
        should_terminate = ~(state.grid == DIRTY).any() | (state.step_count >= self.time_limit)
        if self.end_if_illegal:
            should_terminate |= ~valid_actions.any()
        return should_terminate

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Reset the environment to its initial state.

        All the tiles except upper left are dirty, and the agents start in the upper left
        corner of the grid.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: `State` object corresponding to the new state of the environment after a reset.
            timestep: `TimeStep` object corresponding to the first timestep returned by the
                environment after a reset.
        """
        # Agents start in upper left corner

        state = self.generator(key)
        agents_locations = state.agents_locations

        # Create the action mask and update the state
        state.action_mask = self._compute_action_mask(state.grid, agents_locations)

        observation = self._observation_from_state(state)

        extras = self._compute_extras(state)
        timestep = restart(observation, extras)

        return state, timestep

    def _clean_tiles_containing_agents(self, grid: chex.Array, agents_locations: chex.Array) -> chex.Array:
        """Clean all tiles containing an agent."""
        # Create a mask of the same shape as grid, initialized to False
        mask = jnp.zeros_like(grid, dtype=bool)

        # Set the locations of agents in the mask to True
        mask = mask.at[agents_locations[:, 0], agents_locations[:, 1]].set(True)

        # Use jnp.where to set the value CLEAN at the locations of agents, leaving other values unchanged
        new_grid = jnp.where(mask, 1, grid)

        return new_grid

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of the environment's dynamics.

        If an action is invalid, the corresponding agent does not move and
        the episode terminates.

        Args:
            state: current environment state.
            action: Jax array of shape (num_agents,). Each agent moves one step in
                the specified direction (0: up, 1: right, 2: down, 3: left).

        Returns:
            state: `State` object corresponding to the next state of the environment.
            timestep: `TimeStep` object corresponding to the timestep returned by the environment.
        """
        is_action_valid = self._is_action_valid(action, state.action_mask)

        agents_locations = self._update_agents_locations(state.agents_locations, action, is_action_valid)

        grid = self._clean_tiles_containing_agents(state.grid, agents_locations)

        prev_state = state

        state = State(
            agents_locations=agents_locations,
            grid=grid,
            action_mask=self._compute_action_mask(grid, agents_locations),
            step_count=state.step_count + 1,
            key=state.key,
        )

        if self.end_if_illegal:
            invalid = ~is_action_valid.any()
            reward = self._compute_reward(prev_state, state) * ~invalid - 50.0 * invalid
        else:
            reward = self._compute_reward(prev_state, state)

        observation = self._observation_from_state(state)

        done = self._should_terminate(state, is_action_valid)

        extras = self._compute_extras(state)
        # Return either a MID or a LAST timestep depending on done.
        timestep = jax.lax.cond(
            done,
            lambda reward, observation, extras: termination(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            lambda reward, observation, extras: transition(
                reward=reward,
                observation=observation,
                extras=extras,
            ),
            reward,
            observation,
            extras,
        )

        return state, timestep


class SingleCleanerV2(JumanjiEnv):

    def __init__(self):
        super().__init__(
            env_name='SingleCleaner-v1',
            env_kwargs={'generator': RandomGenerator(
                num_rows=10, num_cols=10, num_agents=1
            )}
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=1, shape=(10, 10, 3), dtype=np.int32),
            'agents_locations': spaces.Box(
                low=0, high=10, shape=(2, ), dtype=np.int32),
        })

    def process_observation(self, observation):
        agent_pos = observation['agents_locations'].squeeze()
        grid = observation['grid']
        dirty_channel = np.where(grid == 0, 1, 0)
        clean_channel = np.where(grid == 1, 1, 0)
        wall_channel = np.where(grid == 2, 1, 0)
        processed_grid = np.stack([dirty_channel, clean_channel, wall_channel], axis=-1)
        return {'grid': processed_grid, 'agents_locations': agent_pos}

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        return self.process_observation(observation)

    def step(self, action):
        action = np.array(action)[..., None]
        observation, reward, done, info = super().step(action)
        observation = self.process_observation(observation)
        return observation, reward, done, info
