from typing import Optional, Tuple, Dict, Collection, List

import gym
import gymnasium
import numpy as np
from strenum import StrEnum
from gymnasium.envs.registration import register
import os
import sys
import cv2
import math

from numpy.typing import ArrayLike

from reinforcement_learning.environments.doom.corridor_policy import prescribe_action

directory_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(directory_path, '..', '..', '..')
sys.path.append(path)

from reinforcement_learning.environments.two_level_env import TwoLevelEnv, Directions

register(
    id="Doom-v0",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": os.path.join(directory_path, "env.cfg"), "frame_skip": 4}
)
register(
    id="DoomMedium-v0",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": os.path.join(directory_path, "env_3.cfg"), "frame_skip": 4}
)
register(
    id="DoomHard-v0",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": os.path.join(directory_path, "env_4.cfg"), "frame_skip": 4}
)
register(
    id="DoomRandomInit-v0",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": os.path.join(directory_path, "env_2.cfg"), "frame_skip": 4}
)


class EnvMode(StrEnum):
    ROOM = 'room'
    EVAL = 'eval'


def l1_dist_to_area(x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> float:
    if x > x_max:
        new_x = x_max
    elif x < x_min:
        new_x = x_min
    else:
        new_x = x
    if y > y_max:
        new_y = y_max
    elif y < y_min:
        new_y = y_min
    else:
        new_y = y
    return np.abs(new_x - x) + np.abs(new_y - y)


class DoomMaze(TwoLevelEnv, gym.Env):

    def __init__(
            self,
            mode: EnvMode = 'room',
            potential_discount: float = .99,
            normalize_potential: bool = True,
            render_mode: str = 'rgb_array',
            direction: Optional[Directions] = None,
            difficulty: str = 'normal',
            **kwargs
    ):
        if EnvMode(mode) not in EnvMode:
            raise ValueError(f"Invalid mode: {mode}")

        self._mode = mode
        mode = 'RandomInit' if mode == 'room' else ''
        if mode == '':
            difficulty = {
                'normal': '',
                'medium': 'Medium',
                'hard': 'Hard',
            }[difficulty]
        else:
            difficulty = ''

        self._gymnasium_env = gymnasium.make(f'Doom{difficulty}{mode}-v0', render_mode=render_mode)
        observation_space = dict()
        for key, value in self._gymnasium_env.observation_space.items():
            if key == 'gamevariables':
                observation_space['game_variables'] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
            elif key in ['screen', 'depth', 'labels']:
                observation_space[key] = gym.spaces.Box(
                    low=np.mean(value.low),
                    high=np.mean(value.high),
                    shape=(45, 60, value.shape[-1]),
                    dtype=value.dtype
                )
            else:
                observation_space[key] = gym.spaces.Box(
                    low=np.mean(value.low),
                    high=np.mean(value.high),
                    shape=value.shape,
                    dtype=value.dtype
                )
        observation_space['label'] = gym.spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = gym.spaces.Discrete(self._gymnasium_env.action_space.n)
        self._boundaries = [
            # shape: ((x_min, x_max), (y_min, y_max))
            ((-320, 544), (-480, 192)),  # Room 0
            ((672, 1216), (-480, 192)),  # Room 1
            ((1344, 1888), (-1312, -160)),  # Room 2
            ((-320, 1216), (-704, -576)),  # Room 3
            ((416, 1440), (-1888, -1440)),  # Room 4
            ((320, 1248), (-1312, -992)),  # Room 5
            ((-320, 192), (-1344, -800)),  # Room 6
            ((-768, -448), (-992, -769)),  # Room 7
        ]
        # positions of the areas (boundaries) containing the low-level goals, per room and per direction
        self._low_level_goals: List[Dict[Directions, Tuple[Tuple[int, int], ...]]] = [
            # Room 0
            {
                Directions.DOWN: ((64, 192), (-480, -416)),
                Directions.RIGHT: ((480, 544), (-224, -96)),
            },
            # Room 1
            {
                Directions.LEFT: ((672, 736), (-224, -96)),
                Directions.DOWN: ((896, 1024), (-480, -416)),
                Directions.RIGHT: ((1152, 1216), (-128, 0)),
            },
            # Room 2
            {
                Directions.UP: ((1408, 1536), (-224, -160)),
                Directions.LEFT: ((1344, 1408), (-1312, -1216)),
                Directions.DOWN: ((1600, 1728), (-1312, -1248)),
            },
            # Room 3
            {
                Directions.UP: ((32, 160), (-640, -576)),
                Directions.DOWN: ((672, 800), (-704, -640)),
            },
            # Room 4
            {
                Directions.UP: ((672, 800), (-1504, -1440)),
                Directions.RIGHT: ((1376, 1440), (-1568, -1440)),
            },
            # Room 5
            {
                Directions.UP: ((672, 800), (-1056, -992)),
                Directions.DOWN: ((768, 896), (-1312, -1248)),
                Directions.LEFT: ((320, 384), (-1184, -1056)),
                Directions.RIGHT: ((1184, 1248), (-1312, -1184)),
            },
            # Room 6
            {
                Directions.LEFT: ((-320, -256), (-928, -800)),
                Directions.RIGHT: ((128, 192), (-1152, -1024)),
            },
            # Room 7
            {
                Directions.RIGHT: ((-512, -448), (-928, -800)),
            },
        ]

        def _calculate_middle_points(rooms):
            results = []
            for _, room in enumerate(rooms):
                room_results = {}
                for direction_, ((x_min, x_max), (y_min, y_max)) in room.items():
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    room_results[direction_] = (x_center, y_center)
                results.append(room_results)
            return results

        self._low_level_goals_middle_points = _calculate_middle_points(self._low_level_goals)
        self._high_level_objective_boundaries = self._boundaries[-1]
        self.game_variables = None
        self._current_room = -1
        self._prev_room = -1
        self._has_been_hard_reset = False
        self._low_level_objective_boundaries = None
        self.discount = potential_discount
        self.normalize_potential = normalize_potential
        self._unsafe_state = False
        self.input_direction = Directions.NOOP
        self._target_direction = Directions.NOOP
        assert direction != Directions.NOOP, "The direction should be set to a valid value."
        self._force_direction = direction
        self._step_backward = 0
        self._just_won = False

    @property
    def current_room(self) -> int:
        return self._current_room

    @current_room.setter
    def current_room(self, value):
        self._prev_room = self._current_room
        self._current_room = value

    @property
    def screen_resolution(self) -> Tuple[int, int]:
        return self._gymnasium_env.env.env.game.get_screen_height(), \
            self._gymnasium_env.env.env.game.get_screen_width()

    def process_game_variables(self, game_variables: np.ndarray) -> Dict[str, np.float32]:

        game_variables = {
            'selected_weapon_ammo': game_variables[..., 0],
            'velocity_x': game_variables[..., 1],
            'velocity_y': game_variables[..., 2],
            'health': game_variables[..., 3],
            'armor': game_variables[..., 4],
            'dead': game_variables[..., 5],
            'position_x': game_variables[..., 6],
            'position_y': game_variables[..., 7],
            'angle': game_variables[..., 8],
            'killcount': game_variables[..., 9],
            'itemcount': game_variables[..., 10],
            'damage_taken': game_variables[..., 11],
            'damagecount': game_variables[..., 12],
            'attack_ready': game_variables[..., 13],
        }
        if self.is_initialized():
            game_variables['current_room'] = self.current_room
            game_variables['target_direction'] = self._target_direction
            game_variables['potential'] = self.potential_fn(
                game_variables['position_x'], game_variables['position_y'], normalized=self.normalize_potential)

        return game_variables

    @property
    def rooms(self):
        return list(range(len(self._low_level_goals)))

    def _get_state(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        new_obs = dict()
        for key, value in obs.items():
            if key == 'gamevariables':
                game_variables = self.process_game_variables(value)
                new_obs['game_variables'] = np.stack([
                    game_variables['velocity_x'],
                    game_variables['velocity_y'],
                    game_variables['angle'],
                    game_variables['selected_weapon_ammo'],
                    game_variables['health'],
                    game_variables['armor'],
                    game_variables['attack_ready'],
                ])
            elif key in ['screen', 'depth', 'labels']:
                new_obs[key] = cv2.resize(value, (60, 45), interpolation=cv2.INTER_LINEAR)
            else:
                new_obs[key] = value
            new_obs['label'] = self.labeling_fn()

        self._prev_state = new_obs

        return new_obs

    def has_won(
            self, ignore_low_level_objectives: bool = False, game_variables: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Return True iff the agent reached either the high-level objective or the low-level objective.
        Args:
            ignore_low_level_objectives: if set, the method return True iff the high-level objective is reached.
            game_variables: the game variables to consider.
                            If None, the game variables gathered during the last step are used.
        """
        if game_variables is None:
            game_variables = self.game_variables
        if self._low_level_objective_boundaries is None or not game_variables:
            return False
        x, y = game_variables['position_x'], game_variables['position_y']
        won = False
        boundary_set = [self._high_level_objective_boundaries]
        if not ignore_low_level_objectives:
            boundary_set = [self._low_level_objective_boundaries] + boundary_set
        for boundaries in boundary_set:
            x_min, x_max = boundaries[0]
            y_min, y_max = boundaries[1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                won = True
                break
        return won

    def is_initialized(self) -> bool:
        return self.game_variables is not None and self.current_room > -1 and \
            self._low_level_objective_boundaries is not None

    def reset(self, **kwargs):
        if (
                self.is_initialized() and
                not self.should_hard_reset() and
                self._mode == 'eval' and
                self.has_won() and not self.has_won(ignore_low_level_objectives=True)
        ):
            # if the low-level objective is achieved but not the high-level one,
            # and the mode is in evaluation, then go to the next room
            next_room = self.get_next_room(self.current_room, self.target_direction)
            self.input_direction = self.get_entrance_direction(self.current_room, next_room, self.target_direction)
            self.current_room = next_room
            self._just_won = True
            self.between_two_rooms = True
            state = self._prev_state
        else:
            state = self.hard_reset()
        if self._mode == 'room' or self._has_been_hard_reset:
            if self._force_direction is not None:
                self._target_direction = self._force_direction
            else:
                self._target_direction = np.random.choice(self.get_available_directions(self.current_room))
            self.set_low_level_objective(self._target_direction)

        return state

    def potential_fn(self, x, y, normalized: bool = False):
        if self.current_room == 7:
            # if the agent is located in the room containing the high-level goal,
            # just focus on reaching it
            x_min, x_max = self._high_level_objective_boundaries[0]
            y_min, y_max = self._high_level_objective_boundaries[1]
        else:
            x_min, x_max = self._low_level_objective_boundaries[0]
            y_min, y_max = self._low_level_objective_boundaries[1]
        dist = l1_dist_to_area(x, y, x_min, x_max, y_min, y_max)

        if normalized:
            x_min, x_max = self._boundaries[self.current_room][0]
            y_min, y_max = self._boundaries[self.current_room][1]
            return 1 - dist / (x_max + y_max - x_min - y_min)
        else:
            return -1. * dist

    def check_boundaries(self):
        x, y = self.game_variables['position_x'], self.game_variables['position_y']
        x_min, x_max = self._boundaries[self.current_room][0]
        y_min, y_max = self._boundaries[self.current_room][1]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        return False

    def step(self, action):
        assert self._low_level_objective_boundaries is not None, "The low-level objective should be set"
        assert self.is_initialized(), "The environment should be initialized before taking a step."
        if self.has_been_hard_reset:
            self._has_been_hard_reset = False

        if self._just_won and not self.check_boundaries() and not (
                self._prev_room == -1 or self._prev_room == self.current_room
        ):
            pending_action = action
            action = prescribe_action(
                x=self.game_variables['position_x'],
                y=self.game_variables['position_y'],
                angle=self.game_variables['angle'],
                delta=15,
                out_room=self._prev_room,
                in_room=self.current_room,
                low_level_goals=self._low_level_goals_middle_points)
            in_corridor = True
        else:
            pending_action = None
            in_corridor = False
            self.between_two_rooms = False
            self._just_won = False

        prev_game_variables = self.game_variables

        _obs, _, done, _, info = self._gymnasium_env.step(action)

        next_game_variables = self.process_game_variables(_obs['gamevariables'])
        goal_reached = not in_corridor and self.has_won(game_variables=next_game_variables)
        done = done or goal_reached or next_game_variables['health'] <= 0.
        health_delta = next_game_variables['health'] - prev_game_variables['health']
        self._unsafe_state = health_delta < 0 or next_game_variables['dead'].astype(bool)

        # reward computation
        reward = health_delta
        reward -= next_game_variables['dead'] * 100.
        reward += .1 * max(0., next_game_variables['armor'] - prev_game_variables['armor'])
        reward += 10. * (next_game_variables['killcount'] - prev_game_variables['killcount'])
        reward += 5. * (next_game_variables['itemcount'] - prev_game_variables['itemcount'])
        reward += prev_game_variables['damage_taken'] - next_game_variables['damage_taken']
        used_ammo = max(0., prev_game_variables['selected_weapon_ammo'] - next_game_variables['selected_weapon_ammo'])
        if used_ammo > 0.:
            reward -= 1.
        if goal_reached:
            reward += 100.
        # reward shaping
        potential = self.potential_fn(
            prev_game_variables['position_x'],
            prev_game_variables['position_y'],
            normalized=self.normalize_potential)
        reward_shaping = self.discount * self.potential_fn(
            next_game_variables['position_x'],
            next_game_variables['position_y'],
            normalized=self.normalize_potential
        ) - potential
        if self.normalize_potential:
            reward_shaping *= 100
        reward += reward_shaping
        # update game variables
        self.game_variables = next_game_variables
        state = self._get_state(_obs)

        if in_corridor:
            assert pending_action is not None
            # state, _, _, info = self.step(pending_action)
            reward = 0
        elif not done:
            # if the agent is out of the boundaries, it is pushed back inside
            x, y = self.game_variables['position_x'], self.game_variables['position_y']
            x_min, x_max = self._boundaries[self.current_room][0]
            y_min, y_max = self._boundaries[self.current_room][1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                # boundaries checked
                self._step_backward = 0
            elif self._step_backward < 35:
                # OOB, step backward
                self._step_backward += 1
                angle = self.game_variables['angle']
                dx_forward = math.cos(math.radians(angle))
                dy_forward = math.sin(math.radians(angle))
                dx_backward = -dx_forward
                dy_backward = -dy_forward

                def _proximity_to_zone(px, py):
                    dx = max(x_min - px, 0, px - x_max)
                    dy = max(y_min - py, 0, py - y_max)
                    return math.sqrt(dx ** 2 + dy ** 2)

                forward_proximity = _proximity_to_zone(x + dx_forward, y + dy_forward)
                backward_proximity = _proximity_to_zone(x + dx_backward, y + dy_backward)

                if forward_proximity < backward_proximity:
                    go_back = 4
                else:
                    go_back = 5
                state, _reward, _done, info = self.step(go_back)
                reward += _reward
                done = done or _done
            else:
                # stuck
                print(f"[WARNING] player stuck in room {self.current_room}"
                      f" at position ({x}, {y}) and angle {self.game_variables['angle']}")
                if self._mode != 'eval':
                    done = True
                self._step_backward = 0

        info = {
            **info,
            **self.game_variables,
            'just_won': self._just_won,
            'in_corridor': in_corridor,
            'target_location': self.target_location,
            'goal_reached': self.has_won(
                ignore_low_level_objectives=self._mode == 'eval',
                game_variables=self.game_variables),
        }
        return state, reward, done, info

    def _update_room(self) -> None:
        assert self.game_variables is not None, "The game variables should be initialized before updating the room."
        for room, boundaries in enumerate(self._boundaries):
            x_min, x_max = boundaries[0]
            y_min, y_max = boundaries[1]
            x, y = self.game_variables['position_x'], self.game_variables['position_y']
            if x_min <= x <= x_max and y_min <= y <= y_max:
                self.current_room = room
                break

    def get_available_directions(self, room_number: int) -> Collection[Directions]:
        return list(self._low_level_goals[room_number].keys())

    @property
    def available_directions(self) -> Collection[Directions]:
        if self.current_room == -1:
            self._update_room()
        return self.get_available_directions(self.current_room)

    @property
    def initial_directions(self) -> Dict[int, Collection[Directions]]:
        return {0: [Directions.LEFT]}

    def set_low_level_objective(self, direction: Directions) -> None:
        assert direction in self.available_directions, \
            f"Invalid direction ({str(direction)}) for room {self.current_room + 1:d}"
        self._low_level_objective_boundaries = self._low_level_goals[self.current_room][direction]
        self._target_direction = direction

    @property
    def target_direction(self) -> Directions:
        return self._target_direction

    @property
    def target_location(self):
        return self._low_level_objective_boundaries

    def hard_reset(self, **kwargs):
        self._has_been_hard_reset = True
        reset_flag = False
        while not reset_flag:
            obs, _ = self._gymnasium_env.reset()
            self.game_variables = self.process_game_variables(obs['gamevariables'])
            self._update_room()
            if self._mode == 'eval' and self.current_room == 0:
                self.input_direction = Directions.LEFT
            else:
                self.input_direction = Directions.NOOP
            if self._force_direction is None or \
                    self._force_direction in self.get_available_directions(self.current_room):
                reset_flag = True

        return self._get_state(obs)

    def should_hard_reset(self) -> bool:
        return self.game_variables['dead'].astype(bool) or self.has_won(ignore_low_level_objectives=True) \
            or self.game_variables['health'] <= 0.

    @property
    def has_been_hard_reset(self) -> bool:
        return self._has_been_hard_reset

    def get_goal_direction(self, room_number: int) -> Optional[Directions]:
        assert room_number != -1, "The room number should be properly initialized before getting the goal direction."
        if room_number == 6:
            return Directions.LEFT
        if room_number == 7:
            return Directions.RIGHT
        else:
            return None

    def get_next_room(self, room_number: int, direction: Directions) -> int:
        assert room_number != -1, "The room number should be properly initialized before getting the next room."
        available_directions = self.get_available_directions(room_number)
        if direction not in available_directions:
            # if this is not available to leave in the input direction, return the current room
            return room_number
        return [
            # Room 0
            {
                Directions.DOWN: 3,
                Directions.RIGHT: 1,
            },
            # Room 1
            {
                Directions.LEFT: 0,
                Directions.DOWN: 3,
                Directions.RIGHT: 2,
            },
            # Room 2
            {
                Directions.UP: 1,
                Directions.LEFT: 5,
                Directions.DOWN: 4,
            },
            # Room 3
            {
                Directions.UP: 0,
                Directions.DOWN: 5,
            },
            # Room 4
            {
                Directions.UP: 5,
                Directions.RIGHT: 2,
            },
            # Room 5
            {
                Directions.UP: 3,
                Directions.DOWN: 4,
                Directions.LEFT: 6,
                Directions.RIGHT: 2,
            },
            # Room 6
            {
                Directions.LEFT: 7,
                Directions.RIGHT: 5,
            },
            # Room 7
            {
                Directions.RIGHT: 6,
            },
        ][room_number][direction]

    def close(self):
        self._gymnasium_env.close()
        super().close()

    def render(self, *args, **kwargs):
        return self._gymnasium_env.render()

    def labeling_fn(self) -> ArrayLike:
        return np.array([self.has_won(), self._unsafe_state], dtype=np.float32)

    def get_entrance_direction(self, room_0, room_1, direction: Directions) -> Directions:
        if room_0 == 2 and room_1 == 4:
            return Directions.RIGHT
        if room_0 == 4 and room_1 == 2:
            return Directions.DOWN
        if room_0 == 2 and room_1 == 1:
            return Directions.RIGHT
        if room_0 == 1 and room_1 == 2:
            return Directions.UP
        else:
            return super().get_entrance_direction(room_0, room_1, direction)

    @property
    def n_rooms(self) -> int:
        """
        Get the number of rooms in the environment.
        """
        return len(self.rooms)
