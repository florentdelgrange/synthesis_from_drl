import enum
import os
import sys
import time

import numpy as np
import tensorflow as tf
from typing import Optional, Collection

from PIL import Image, ImageDraw, ImageFont
from gym.vector.utils import spaces
from numpy.typing import ArrayLike
from tf_agents.drivers.py_driver import PyDriver
from tf_agents.policies.random_py_policy import RandomPyPolicy

from reinforcement_learning.environments.two_level_env import Directions, TwoLevelEnv

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../../..')

from reinforcement_learning.environments.goal_augmented import GoalAugmentedEnvironment

# Settings
FPS = 21
DEBUG = False
# size of the window around PACMAN
DIM = 32


# Map items
class MapItems(enum.Enum):
    EMPTY = ' '
    WALL = '#'
    PACMAN = 'P'
    # ghost patrolling the map
    GHOST_PATROL = 3
    # ghost chasing pacman
    GHOST_ADV = 4
    # ghost acting randomly
    GHOST_RANDOM = 5
    GHOST = 6
    DOOR = 'X'
    BULLET = 7
    # the cherry is a power up which allows to eat the ghosts for a limited amount of time
    CHERRY = 8
    # initial positions represent the positions where PACMAN pops
    # (chosen randomly) during the initialization of the map
    INIT = 'I'
    # Goal: position that Pacman needs to reach
    GOAL = 'G'

    @staticmethod
    def indices():
        return {e: i for i, e in enumerate(MapItems)}

    def __str__(self, dir: Directions = Directions.NOOP):
        if self is MapItems.PACMAN:
            if dir is Directions.LEFT:
                return u'\u15e4'
            else:
                return u'\u15e7'
        elif self in [MapItems.GHOST, MapItems.GHOST_ADV, MapItems.GHOST_PATROL, MapItems.GHOST_RANDOM]:
            return u'\u15e3'
        elif self is MapItems.WALL:
            if dir in [Directions.UP, Directions.DOWN]:
                return u'\u2551'
            elif dir is not Directions.NOOP:
                return u'═'
            else:
                return '='
        else:
            return {
                MapItems.INIT: '*',
                MapItems.EMPTY: u' ',
                MapItems.CHERRY: u"\U0001F352",
                MapItems.BULLET: u'\u25e6',
                MapItems.DOOR: u'\u2592',
                MapItems.GOAL: u'\u26CB',
            }[self]


# map items used in the Pacman observation
MapIndices = {e: i for i, e in enumerate([
    MapItems.GHOST,
    MapItems.WALL,
    MapItems.CHERRY,
    # MapItems.GOAL,
])}


class Agent:
    def __init__(self, x, y, grid):
        self.x = x
        self.y = y
        self.grid = grid
        self.type: Optional[MapItems] = None

    def get_new_position(self, act: Directions):
        x = self.x
        y = self.y
        return act.next_position(x, y)

    def update_position(self, act: Directions):
        x, y = self.get_new_position(act)
        self.x = x
        self.y = y
        return self.x, self.y

    def get_allowed_actions(self):
        acts = []
        for dir in Directions:
            next_x, next_y = dir.next_position(self.x, self.y)
            if (
                    0 <= next_x < len(self.grid) and
                    0 <= next_y < len(self.grid[0]) and
                    self.grid[next_x][next_y] != MapItems.WALL
            ):
                acts.append(dir)
        return acts

    def get_random_allowed_action(self):
        return Directions(np.random.choice(self.get_allowed_actions()))

    def __str__(self):
        return str(self.type)


class Pacman(Agent):
    def __init__(self, x, y, grid):
        Agent.__init__(self, x, y, grid)
        self.type = MapItems.PACMAN
        self._str = str(self.type)

    def update_position(self, act: Directions):
        x, y = super().update_position(act)
        if act in [Directions.LEFT, Directions.RIGHT]:
            self._str = self.type.__str__(act)
        return x, y

    def get_allowed_actions(self):
        acts = super(Pacman, self).get_allowed_actions()
        acts.remove(Directions.NOOP)
        return acts

    def __str__(self):
        return self._str


class Ghost(Agent):
    def update_position(self, act: Directions):
        old_x, old_y = self.x, self.y
        new_x, new_y = super(Ghost, self).update_position(act)
        if (old_x, old_y) != (new_x, new_y) and self.grid[old_x, old_y] == self.type:
            self.grid[old_x, old_y] = MapItems.EMPTY
        if self.grid[new_x, new_y] == MapItems.EMPTY:
            self.grid[new_x, new_y] = self.type


class PatrolGhost(Ghost):
    def __init__(self, x, y, grid):
        """
        The ghost patrols the grid in a circular fashion.
        """
        Agent.__init__(self, x, y, grid)
        self.type = MapItems.GHOST_PATROL
        self.direction = Directions.RIGHT
        self.prev_direction = Directions.UP

    def get_act(self, pacman):
        if self.prev_direction in self.get_allowed_actions():
            self.direction = self.prev_direction
        if self.direction in self.get_allowed_actions():
            return self.direction
        else:
            # NOOP is always part of the set of allowed actions, so we check if this set contains more directions.
            if len(self.get_allowed_actions()) > 1:
                self.prev_direction = self.direction
                self.direction = self.direction.next_clockwise_direction()
            else:
                # if this is not the case, the ghost is stuck and it does nothing
                self.direction = Directions.NOOP

        return self.get_act(pacman)


class RandomGhost(Ghost):
    def __init__(self, x: int, y: int, grid):
        super(RandomGhost, self).__init__(x, y, grid)
        self.type = MapItems.GHOST_RANDOM

    def get_act(self, _):
        return self.get_random_allowed_action()


class AdversarialGhost(Ghost):
    def __init__(self, x: int, y: int, grid, entropy_temperature: float = 1.5e-2):
        """
        The agent moves in the direction which allows to increase its L1 distance to Pacman
        with probability exp(-dist_to_pacman * entropy_temperature)
        Otherwise, it moves randomly.
        """
        Agent.__init__(self, x, y, grid)
        self.entropy_temperature = np.abs(entropy_temperature)
        self.type = MapItems.GHOST_ADV

    def get_act(self, pacman):
        l1_dist_to_pacman = lambda x, y: l1_dist([x, y], [pacman.x, pacman.y])

        if np.random.binomial(n=1, p=np.exp(-l1_dist_to_pacman(self.x, self.y) * self.entropy_temperature)):
            action_values = {
                action: l1_dist_to_pacman(*action.next_position(self.x, self.y))
                for action in self.get_allowed_actions()
            }
            return min(action_values, key=lambda k: action_values[k])
        else:
            return self.get_random_allowed_action()


ghosts = {
    MapItems.GHOST_ADV: AdversarialGhost,
    MapItems.GHOST_PATROL: PatrolGhost,
    MapItems.GHOST_RANDOM: RandomGhost
}


def l1_dist(x: ArrayLike, y: ArrayLike):
    """
    Compute the L1 (Manhattan) distance between vectors x and y (on the last dimension)
    """
    x = np.array(x)
    y = np.array(y)
    return np.min(np.linalg.norm(x - y, ord=1, axis=-1))


class GridLoader:

    def __init__(self, file, walls: Collection[MapItems] = (MapItems.WALL, MapItems.DOOR, MapItems.GOAL)):
        self.walls = walls
        grid = [list(map(self.get_item, line.strip())) for line in file]
        self.grid = np.array(grid, dtype=object)
        self.rooms = self.find_rooms()

    @staticmethod
    def get_item(symbol: str):
        try:
            symbol_ = int(symbol)
        except ValueError:
            symbol_ = symbol
        return next(filter(lambda item: item.value == symbol_, MapItems))

    def find_rooms(self):
        rooms = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # upper left corner detection
                if (
                        self.grid[i, j] in self.walls and
                        i + 1 < self.grid.shape[0] and self.grid[i + 1, j] in self.walls and
                        j + 1 < self.grid.shape[1] and self.grid[i, j + 1] in self.walls
                ):
                    # start to follow the walls to eventually find a room in the grid
                    visited = np.full_like(self.grid, fill_value=False, dtype=bool)
                    found_borders, borders = self.explore(i, j, visited, initial_position=(i, j))
                    if found_borders:
                        rooms.append(borders)
        return rooms

    def explore(self, i, j, visited, direction=Directions.RIGHT, initial_position=None):
        if initial_position == (i, j) and direction is Directions.UP:
            return True, [(i, j), (i, j)]
        elif visited[i][j]:
            return False, [(i, j), (i, j)]

        visited[i, j] = True
        found_borders = False
        borders = [(i, j), (i, j)]

        next_direction = direction.next_clockwise_direction()
        next_i, next_j = next_direction.next_position(i, j)

        # continue to follow the wall according to the next position
        if (
                direction is not direction.UP and initial_position != (i, j) and
                next_i < self.grid.shape[0] and next_j < self.grid.shape[1] and
                self.grid[next_i, next_j] in self.walls
        ):
            found_borders, borders = self.explore(next_i, next_j, visited, next_direction, initial_position)

        # if no room has been found by following the wall in the next direction,
        # then continue the exploration by following the current direction
        next_i, next_j = direction.next_position(i, j)
        if (
                not found_borders and
                next_i < self.grid.shape[0] and next_j < self.grid.shape[1] and
                self.grid[next_i, next_j] in self.walls
        ):
            found_borders, borders = self.explore(next_i, next_j, visited, direction, initial_position)

        upper_left_corner = min(i, borders[0][0]), min(j, borders[0][1])
        lower_right_corner = max(i, borders[1][0]), max(j, borders[1][1])
        return found_borders, [upper_left_corner, lower_right_corner]


class PacmanEnvironment(GoalAugmentedEnvironment):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            grid_path: Optional[str] = None,
            power_up_steps: int = DIM * 2,
            potential_discount: float = 0.99,
            direction: Optional[Directions] = None,
            normalize_rewards: bool = True,
            adversarial_ghosts_entropy: Optional[float] = None,
            include_pacman_position: bool = False,
            p_cherry: float = 0.,
    ):
        self.grid_path = grid_path
        self.power_up_steps = power_up_steps
        self.grid = None
        self.target_direction = direction
        self._adversarial_ghost_entropy = adversarial_ghosts_entropy
        self.load_grid(self.grid_path)
        self._normalize_reward_fn = normalize_rewards
        self._include_pacman_position = include_pacman_position
        self.p_cherry = p_cherry
        self._move_around_the_rooms = False

        # for rendering
        self.font = None
        self.pil_font = None
        self.font_size = 24
        self.screen_width, self.screen_height = [800, 820]

        observation_spaces = {
            'pacman_observation': spaces.Box(
                low=0,
                high=1,
                shape=(DIM, DIM, len(MapIndices),),
                dtype=np.uint8),
            'pacman_position': spaces.Box(
                low=0,
                high=max(*self.grid.shape),
                shape=(2,),
                dtype=np.int32),
            'goal': spaces.Box(
                low=0,
                high=max(*self.grid.shape),
                shape=(2,),
                dtype=np.int32),
            'power_up': spaces.Box(
                low=0,
                high=power_up_steps,
                # dim 0 is whether Pacman is currently under a power up bonus (1) or not (0),
                # making Pacman invulnerable;
                # dim 1 is the remaining power-up time
                shape=(2,),
                dtype=np.int32),
            'label': spaces.Box(
                low=0,
                high=1,
                shape=(2,),
                dtype=np.uint8),
        }
        if self.target_direction is not None:
            del observation_spaces['goal']
        if not self._include_pacman_position:
            del observation_spaces['pacman_position']
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Discrete(len(Directions) - 1)
        self._power_up = False
        self._power_up_remaining_steps = 0
        self._score = 0
        self.gamma = potential_discount
        self.room_number = 0

    def get_next_room(self, room_number: int, direction: Directions):
        found = False

        assert 0 <= room_number < len(self.rooms)

        for i, [(room_x0, room_y0), (room_x1, room_y1)] in enumerate(self.rooms):
            if i == room_number:
                x, y = (room_x0 + room_x1) // 2, (room_y0 + room_y1) // 2
                found = True

        assert found

        original_room = current_room = room_number

        while original_room == current_room and found:
            found = False
            x, y = direction.next_position(x, y)
            for i, [(room_x0, room_y0), (room_x1, room_y1)] in enumerate(self.rooms):
                if room_x0 <= x <= room_x1 and room_y0 <= y <= room_y1:
                    current_room = i
                    found = True
                    break
            # if not found, then the cursor moved outside the boundaries of the grid
            # in that case, just return the input room

        return current_room

    def _update_room(self):
        self.position_offset = np.array([0, 0])
        # determine which room Pacman is in, and derive the position offset accordingly
        for i, [(room_x0, room_y0), (room_x1, room_y1)] in enumerate(self.rooms):
            if room_x0 <= self.pacman.x <= room_x1 and room_y0 <= self.pacman.y <= room_y1:
                self.position_offset = np.array([room_x0, room_y0])
                self.room_boundaries = np.array([room_x1 - room_x0, room_y1 - room_y0])
                self.room_number = i
                break

        self.goal_positions = np.stack(np.where((self.grid == MapItems.DOOR) | (self.grid == MapItems.GOAL)), axis=-1)
        self.goal_positions = np.array([
            [x, y]
            for x, y in self.goal_positions
            if room_x0 <= x <= room_x1 and room_y0 <= y <= room_y1])

    def load_grid(self, grid_path):
        if grid_path is None:
            import importlib.resources as pkg_resources
            file = pkg_resources.open_text(__package__, 'grid.txt')
        else:
            file = open(grid_path, 'r')

        loader = GridLoader(file=file)
        self.grid = loader.grid
        self.rooms = loader.rooms

        self.initial_positions = np.stack(np.where(self.grid == MapItems.INIT), axis=-1)
        x, y = self.initial_positions[np.random.choice(self.initial_positions.shape[0])]
        self.pacman = Pacman(x, y, self.grid)
        self.agents = [self.pacman]
        self.grid[self.grid == MapItems.INIT] = MapItems.EMPTY

        self._update_room()

        if self.target_direction is None:
            self._current_goal = self.goal_positions[np.random.choice(self.goal_positions.shape[0])]
        else:
            self._current_goal = [
                goal
                for goal in self.goal_positions
                if Directions.relative(self.room_boundaries / 2, goal - self.position_offset) == self.target_direction
            ]
            if not self._current_goal:
                file.close()
                return self.load_grid(grid_path)

        for ghost_type in [MapItems.GHOST_RANDOM, MapItems.GHOST_PATROL, MapItems.GHOST_ADV]:
            self.agents.extend([
                ghosts[ghost_type](x, y, self.grid)
                for x, y in np.stack(np.where(self.grid == ghost_type), axis=-1)
            ])
        for ghost in self.agents:
            if ghost.type == MapItems.GHOST_ADV:
                ghost.entropy_temperature = self._adversarial_ghost_entropy

        return

    def _get_state(self):
        observation = np.zeros(shape=(DIM, DIM, len(MapIndices)), dtype=np.uint8)

        # create the window of size DIMxDIM around Pacman
        for i, x in enumerate(range(self.pacman.x - DIM // 2, self.pacman.x + DIM // 2 + DIM % 2)):
            for j, y in enumerate(range(self.pacman.y - DIM // 2, self.pacman.y + DIM // 2 + DIM % 2)):
                if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                    for map_item, index in MapIndices.items():
                        observation[i, j, index] += int({
                                                            MapItems.GHOST_ADV: MapItems.GHOST,
                                                            MapItems.GHOST_PATROL: MapItems.GHOST,
                                                            MapItems.GHOST_RANDOM: MapItems.GHOST,
                                                            MapItems.DOOR: MapItems.GOAL,
                                                        }.get(self.grid[x, y], self.grid[x, y]) == map_item)

        state = {
            'pacman_observation': observation,
            'power_up': np.array([self._power_up, self._power_up_remaining_steps]),
            'label': self.labeling_fn().astype(int),
        }
        if self.target_direction is None:
            state['goal'] = self.current_goal - self.position_offset
        if self._include_pacman_position:
            state['pacman_position'] = np.array([self.pacman.x, self.pacman.y]) - self.position_offset
        return state

    def reset(self, **kwargs):
        self.load_grid(self.grid_path)
        self._score = 0
        return self._get_state()

    def labeling_fn(self):
        return np.array([
            l1_dist((self.pacman.x, self.pacman.y), self.current_goal) < 1e-6,
            np.any(
                [(self.pacman.x, self.pacman.y) == (ghost.x, ghost.y) for ghost in self.ghosts]
            ) and not self._power_up])

    def state_to_goal(self, state: np.typing.ArrayLike, batched: bool = False, lib=np):
        return state.get('pacman_position', np.array([self.pacman.x, self.pacman.y]) - self.position_offset)

    def is_achieved(self, state: np.typing.ArrayLike, goal: np.typing.ArrayLike, batched: bool = False, lib=np):
        if lib == np:
            all = lambda x: np.all(x, axis=-1 if batched else None)
        elif lib == tf:
            all = lambda x: tf.reduce_all(x, axis=-1 if batched else None)
        else:
            raise ValueError("lib should be either numpy or tensorflow")
        return all(self.state_to_goal(state) == goal)

    @property
    def current_goal(self):
        return self._current_goal

    @property
    def ghosts(self):
        return [agent for agent in self.agents if agent != self.pacman]

    def game_over(self):
        for agent in self.ghosts:
            if agent.x == self.pacman.x and agent.y == self.pacman.y and not self._power_up:
                return True
        return False

    def potential_fn(self, state: ArrayLike, normalized: bool = False):
        if normalized:
            return 1 - l1_dist(state, self.current_goal) / np.sum(self.room_boundaries)
        else:
            return -1. * l1_dist(state, self.current_goal)

    def step(self, action):
        pacman_pos = np.array([self.pacman.x, self.pacman.y])
        if action in self.pacman.get_allowed_actions():
            _x, _y = Directions(action).next_position(self.pacman.x, self.pacman.y)
            if self._move_around_the_rooms or (
                    self.grid[_x, _y] != MapItems.DOOR or l1_dist((_x, _y), self.current_goal) == 0
            ):
                self.pacman.update_position(Directions(action))
        new_pacman_pos = np.array([self.pacman.x, self.pacman.y])

        rew = self.gamma * self.potential_fn(new_pacman_pos, self._normalize_reward_fn) \
              - self.potential_fn(pacman_pos, self._normalize_reward_fn)

        # consumable
        if self.grid[self.pacman.x, self.pacman.y] == MapItems.CHERRY:
            self.grid[self.pacman.x, self.pacman.y] = MapItems.EMPTY
            self._power_up = True
            rew += .5
            self._power_up_remaining_steps += self.power_up_steps
        if self.grid[self.pacman.x, self.pacman.y] in ghosts.keys() and self._power_up:
            self.grid[self.pacman.x, self.pacman.y] = MapItems.EMPTY
            rew += 1.
        if self.p_cherry > 0:
            if np.random.binomial(n=1, p=self.p_cherry):
                candidates = np.stack(np.where(self.grid == MapItems.EMPTY), axis=-1)
                candidates = candidates[np.all(0 <= candidates - self.position_offset, axis=-1)]
                candidates = candidates[np.all(candidates - self.position_offset <= self.room_boundaries, axis=-1)]
                if len(candidates > 0):
                    x, y = candidates[np.random.choice(len(candidates))]
                    self.grid[x, y] = MapItems.CHERRY
        # ghosts
        if not self.game_over():
            for g in self.ghosts:
                if self._power_up and self.pacman.x == g.x and self.pacman.y == g.y:
                    self.agents.remove(g)
                else:
                    act = g.get_act(self.pacman)
                    g.update_position(act)
        done = self.game_over()

        # Goal
        if l1_dist(new_pacman_pos, self.current_goal) < 1e-5:
            rew += np.sum([self.gamma ** i * rew for i in range(200)])
            done = True

        obs = self._get_state()
        if self.is_unsafe(obs):
            deadlock_penalty = -1. * np.abs(
                self.gamma * self.potential_fn(new_pacman_pos, self._normalize_reward_fn)
                - self.potential_fn(new_pacman_pos, self._normalize_reward_fn))
            if not deadlock_penalty:
                deadlock_penalty = -1.
            rew += np.sum([self.gamma ** i * deadlock_penalty for i in range(200)])

        self._power_up_remaining_steps = max(0, self._power_up_remaining_steps - 1)
        self._power_up = self._power_up and self._power_up_remaining_steps > 0
        self._score += rew

        return obs, rew, done, {}

    def is_unsafe(self, state: ArrayLike, lib=np):
        return state['label'][..., 1]

    def __str__(self):
        if self.game_over():
            return u"""
          _____                         ____                 
         / ____|                       / __ \                
        | |  __  __ _ _ __ ___   ___  | |  | |_   _____ _ __ 
        | | |_ |/ _` | '_ ` _ \ / _ \ | |  | \ \ / / _ \ '__|
        | |__| | (_| | | | | | |  __/ | |__| |\ V /  __/ |   
         \_____|\__,_|_| |_| |_|\___|  \____/  \_/ \___|_|   
        """ + '\n' * (self.grid.shape[0] - 6)
        else:
            grid = np.vectorize(str)(self.grid)
            grid[self.pacman.x][self.pacman.y] = str(self.pacman)
            return '\n'.join("".join([char.ljust(2).rjust(2) for char in row]) for row in grid)

    def render(self, mode="human"):
        # os.system('cls' if os.name == 'nt' else 'clear')
        _str = str(self)
        current_observation = self._get_state()
        _str += '\n[win loose]: '
        _str += str(self.labeling_fn().tolist())
        _str += ' | SCORE: '
        _str += '{:.6f}'.format(self._score)
        _str += ' | POS: '
        if self._include_pacman_position is None:
            _str += str(current_observation['pacman_position'])
        else:
            _str += str(np.array([self.pacman.x, self.pacman.y]) - self.position_offset)
        _str += ' | GOAL: '
        if self.target_direction is None and 'goal' in current_observation:
            _str += str(current_observation['goal'])
        else:
            _str += str(self.current_goal - self.position_offset)
        _str += ' | POW: '
        _str += str(self._power_up)
        _str += ' ('
        _str += str(self._power_up_remaining_steps)
        _str += ')'
        _str += f' | #GHOSTS: {len(self.ghosts):d}'
        _str = _str.encode('utf-8').decode('utf-8')
        if mode == 'human':
            CURSOR_UP = '\033[F'
            ERASE_LINE = '\033[K'
            for _ in range(_str.count('\n')):
                print(CURSOR_UP + ERASE_LINE + CURSOR_UP)
            sys.stdout.write('\r')
            sys.stdout.write(_str)
            sys.stdout.flush()
            time.sleep(FPS ** (-1))
            if self.game_over():
                time.sleep(2.)
        elif mode == 'rgb_array':
            lines = _str.split('\n')

            if self.font is None:
                desired_width, desired_height = self.screen_width, self.screen_height
                # Create a white background image
                image = Image.new("RGB", (desired_width, desired_height), "white")
                draw = ImageDraw.Draw(image)  # Initialize the draw object
                self.font = os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    'fonts',
                    'Unifont.ttf'))
                num_chars_per_line = max([len(line) for line in lines])
                num_lines = len(lines)

                while True:
                    self.pil_font = ImageFont.truetype(
                        self.font,
                        size=self.font_size)
                    char_width, char_height = draw.textsize("A", font=self.pil_font)

                    # Calculate the actual width and height based on the font size
                    actual_width = num_chars_per_line * char_width
                    actual_height = num_lines * char_height

                    # Check if the actual dimensions fit within the desired dimensions
                    if actual_width <= desired_width and actual_height <= desired_height:
                        break

                    # Reduce the font size if it's too large
                    self.font_size -= 1

            canvas = Image.new('RGB', (self.screen_width, self.screen_height), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            x, y = 0, 0
            # Loop through each line
            for line_n, line in enumerate(lines):
                # Loop through each character in the line
                char_height = 0
                for char in line:
                    char_width, char_height = draw.textsize(char, font=self.pil_font)
                    draw.text((x, y), char,
                              fill={
                                  u'\u15e4': "#B8860B",
                                  u'\u15e7': "#B8860B",
                                  u'\u15e3': "red",
                              }.get(char, "#000000"),
                              font=self.pil_font, antialias=True)
                    x += (
                        self.screen_width // len(line)
                        if line_n < len(lines) - 1 else
                        char_width
                    )  # Move the drawing position to the right
                # Move to the next line
                x = 0
                y += char_height

            # white = "#000000"
            # draw.text((0, 0), _str, font=pil_font, fill=white, antialias=True)
            return np.asarray(canvas)


class PacmanEnvEval(PacmanEnvironment):

    def __init__(
            self,
            lives: int = 3,
            high_level_goal: bool = False,
            include_goal_in_obs_space: bool = False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._last_action = None
        self._move_around_the_rooms = high_level_goal
        self.input_direction = Directions.NOOP

        if hasattr(self.observation_space, 'spaces'):
            obs_items = self.observation_space.spaces.items()
        else:
            obs_items = self.observation_space.items()

        if not include_goal_in_obs_space:
            self.observation_space = spaces.Dict({
                key: value for key, value in obs_items
                if key != "goal"
            })
        self._previous_initial_position = self.pacman.x, self.pacman.y
        self._lives = lives
        self.lives = self._lives
        self.n_rooms = len(self.rooms)
        self._hard_reset = True

    def reset_life_points(self):
        self.lives = self._lives

    def get_available_directions(self, room_number: int):
        found = False

        assert 0 <= room_number < len(self.rooms)

        for i, [(room_x0, room_y0), (room_x1, room_y1)] in enumerate(self.rooms):
            if i == room_number:
                room_boundaries = np.array([room_x0 + room_x1, room_y0 + room_y1]) / 2
                found = True
                break

        assert found

        goal_positions = np.array([
            [x, y]
            for x, y in np.stack(np.where(
                (self.grid == MapItems.DOOR) | (self.grid == MapItems.GOAL)), axis=-1
            ) if room_x0 <= x <= room_x1 and room_y0 <= y <= room_y1
        ])

        possible_goals = {
            direction: [
                goal
                for goal in goal_positions if Directions.relative(room_boundaries, goal) == direction
            ] for direction in Directions if direction != Directions.NOOP
        }

        return [direction for direction, goals in possible_goals.items() if goals]

    @property
    def available_directions(self):
        possible_goals = {
            direction: [
                goal
                for goal in self.goal_positions
                if Directions.relative(self.room_boundaries / 2, goal - self.position_offset) == direction
            ] for direction in Directions if direction != Directions.NOOP
        }
        return [direction for direction, goals in possible_goals.items() if goals]

    @property
    def initial_directions(self):
        d = dict()
        for i, [(room_x0, room_y0), (room_x1, room_y1)] in enumerate(self.rooms):
            initial_directions = []
            for direction in [dir for dir in Directions if dir != Directions.NOOP]:
                for initial_position in self.initial_positions:
                    x, y = initial_position
                    if room_x0 <= x <= room_x1 and room_y0 <= y <= room_y1 and Directions.relative(
                            np.array((room_x0 + room_x1, room_y0 + room_y1)) / 2,
                            initial_position
                    ) == direction:
                        initial_directions.append(direction)
            if initial_directions:
                d[i] = initial_directions
        return d

    def set_low_level_objective(self, direction: Directions):
        self.target_direction = direction
        self._current_goal = [
            goal
            for goal in self.goal_positions
            if Directions.relative(self.room_boundaries / 2, goal - self.position_offset) == self.target_direction
        ]

    def _get_state(self):
        state = super(PacmanEnvEval, self)._get_state()
        if 'goal' in state:
            del state['goal']
        return state

    def should_hard_reset(self) -> bool:
        return (
                (self.lives <= 1 and (self.game_over() or not self.labeling_fn()[0])) or
                self.grid is None or
                self.grid[self.pacman.x, self.pacman.y] == MapItems.GOAL)

    @property
    def has_been_hard_reset(self) -> bool:
        return self._hard_reset

    def hard_reset(self, **kwargs):
        self.target_direction = None
        s_init = super(PacmanEnvEval, self).reset(**kwargs)
        self._previous_initial_position = self.pacman.x, self.pacman.y
        self.reset_life_points()
        self.n_rooms = len(self.rooms)
        self._hard_reset = True
        self.input_direction = Directions(np.random.choice(self.initial_directions[self.room_number]))
        return s_init

    def reset(self, **kwargs):
        if self.should_hard_reset():
            # hard reset
            return self.hard_reset(**kwargs)

        elif self.game_over():
            self.pacman.x, self.pacman.y = self._previous_initial_position
            self._score = 0.
            self.lives -= 1
            self.input_direction = Directions.NOOP
            return self._get_state()

        elif self.labeling_fn()[0]:
            # the goal is reached, go to the next room
            obs, _, _, _ = self.step(self._last_action)
            self.input_direction = Directions(self.target_direction).opposite()
            self._update_room()
            self.reset_life_points()
            self._previous_initial_position = self.pacman.x, self.pacman.y
            return obs

        else:
            # a time-out occurred, continue
            self.lives -= 1
            self.input_direction = Directions.NOOP
            return self._get_state()

    def get_goal_direction(self, room_number: int):
        goal_positions = np.stack(np.where(self.grid == MapItems.GOAL), axis=-1)
        found = False

        for x, y in goal_positions:
            for i, [(room_x0, room_y0), (room_x1, room_y1)] in enumerate(self.rooms):
                if i == room_number:
                    found = (
                            room_x0 <= x <= room_x1 and
                            room_y0 <= y <= room_y1
                    )
                    v_0 = np.array([(room_x0 + room_x1), (room_y0 + room_y1)]) / 2.
                    goal_direction = Directions.relative(v_0, (x, y))
                    break

        if not found:
            return None
        else:
            return goal_direction

    def step(self, action):
        self._hard_reset = False
        self._last_action = action
        obs, rew, done, info = super(PacmanEnvEval, self).step(action)
        info['goal_reached'] = self.grid[self.pacman.x, self.pacman.y] == MapItems.GOAL
        info['game_over'] = self.game_over()
        info['continue'] = not (info['goal_reached'] or info['game_over'])
        info['room_number'] = self.room_number
        info['score'] = self._score
        return obs, rew, done or not info['continue'], info

    @property
    def continue_simulation(self) -> bool:
        done = self.grid[self.pacman.x, self.pacman.y] == MapItems.GOAL or self.game_over()
        return not done


class HighLevelEnvironment(PacmanEnvEval, TwoLevelEnv):

    def __init__(self, *args, **kwargs):
        super(HighLevelEnvironment, self).__init__(*args, **kwargs)
        self._current_goal = np.stack(np.where(self.grid == MapItems.GOAL), axis=-1)
        self._move_around_the_rooms = True

        if hasattr(self.observation_space, 'spaces'):
            space = self.observation_space.spaces
        else:
            space = self.observation_space

        observation_space = {
            **space,
            'room': spaces.Box(
                low=0,
                high=1,
                shape=(self.n_rooms,),
                dtype=np.uint8)}
        self.observation_space = spaces.Dict(observation_space)

    @property
    def current_room(self):
        return self.room_number

    def _get_state(self):
        state = super()._get_state()
        state['room'] = np.eye(self.n_rooms)[self.room_number]
        return state

    def potential_fn(self, state: ArrayLike, normalized: bool = False):
        high_level_goal_position = np.stack(np.where(self.grid == MapItems.GOAL), axis=-1)
        if normalized:
            return 1 - l1_dist(state, high_level_goal_position) / np.sum(self.room_boundaries)
        else:
            return -1. * l1_dist(state, high_level_goal_position)

    def reset(self, **kwargs):
        # hard reset
        self.target_direction = None
        s_init = super(PacmanEnvEval, self).reset(**kwargs)
        self._current_goal = np.stack(np.where(self.grid == MapItems.GOAL), axis=-1)
        self._previous_initial_position = self.pacman.x, self.pacman.y
        self.lives = 5
        self.n_rooms = len(self.rooms)
        return s_init

    def step(self, action):
        obs, rew, done, info = super(HighLevelEnvironment, self).step(action)

        if info['game_over'] and self.lives > 1:
            self.pacman.x, self.pacman.y = self._previous_initial_position
            self._score = 0.
            self.lives -= 1
        done = (info['game_over'] and self.lives <= 1) or info['goal_reached']

        if not done:
            relative_pos = np.array([self.pacman.x, self.pacman.y]) - self.position_offset
            x, y = relative_pos
            x_max, y_max = self.room_boundaries
            if not (0 <= x <= x_max and 0 <= y <= y_max):
                self._previous_initial_position = self.pacman.x, self.pacman.y
                self._update_room()

        return obs, rew, done, info


if __name__ == '__main__':
    from tf_agents.environments import suite_gym, tf_py_environment
    from tf_agents.policies import random_tf_policy
    from tf_agents.drivers import dynamic_episode_driver
    from tf_agents.trajectories import StepType

    from reinforcement_learning.environments.pacman.metric import LabelMetric

    args = sys.argv[1:]
    if '--human' in args:
        human = True
    else:
        human = False
    env_name = 'PacmanGrid{}-v0'.format(
        'Eval' if '--eval' in args else '', 'HL' if "--HL" in args else '')

    if '--grid_path' in args:
        grid_path = args[args.index('--grid_path') + 1]
    else:
        grid_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'grid_easy{}.txt'.format('Eval' if '--eval' in args else ''))

    if '--direction' in args:
        direction = {
            'south': Directions.DOWN,
            'down': Directions.DOWN,
            'north': Directions.UP,
            'up': Directions.UP,
            'east': Directions.RIGHT,
            'right': Directions.RIGHT,
            'west': Directions.LEFT,
            'left': Directions.LEFT,
        }[args[args.index('--direction') + 1].lower()]
    else:
        direction = None

    episodes = int(args[args.index('--episodes') + 1]) if '--episodes' in args else 1

    with suite_gym.load(env_name, gym_kwargs={'direction': direction, 'grid_path': grid_path}) as py_env:
        if human:
            done = False
            for _ in range(episodes):
                py_env.reset()
                py_env.render(mode='human')
                done = False
                while not done:
                    action = input()
                    action = action[0] if len(action) > 0 else 'X'
                    action = {
                        'z': Directions.UP,
                        's': Directions.DOWN,
                        'q': Directions.LEFT,
                        'd': Directions.RIGHT,
                    }.get(action[0], Directions.NOOP)
                    step = py_env.step(action)
                    done = step.step_type == StepType.LAST
                    py_env.render(mode='human')
                print("Done!")
                py_env.render(mode='human')
        else:
            # py_env = PerturbedEnvironment(env=py_env, perturbation=.1)

            if '--tf' in args:
                tf_env = tf_py_environment.TFPyEnvironment(py_env)
                tf_policy = random_tf_policy.RandomTFPolicy(
                    action_spec=tf_env.action_spec(),
                    time_step_spec=tf_env.time_step_spec())
                label_metric = LabelMetric()

                driver = dynamic_episode_driver.DynamicEpisodeDriver(
                    env=tf_env,
                    policy=tf_policy,
                    observers=[
                        lambda _: py_env.render(mode='human') if '--render' in args else None,
                        label_metric,
                        # lambda _: print(label_metric.result()),
                    ],
                    num_episodes=episodes,
                ).run()
                print(label_metric.result())
                breakpoint()
            else:
                py_env.reset()
                PyDriver(
                    env=py_env,
                    policy=RandomPyPolicy(time_step_spec=py_env.time_step_spec(), action_spec=py_env.action_spec()),
                    observers=[
                        # lambda _: py_env.render(mode='human')
                    ],
                    max_episodes=episodes,
                ).run(time_step=py_env.current_time_step())

            pass
