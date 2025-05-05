# -*- coding: utf-8 -*-

import os
import sys

# from gym.core import ObsType, ActType
from numpy.typing import ArrayLike, DTypeLike

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../../..')

import time
import random

from typing import Union, Collection, List, Tuple, Optional, Dict

import gym
from gym import error, spaces, utils
from gym.utils import seeding
# import Agents
import numpy as np

from tabulate import tabulate
from PIL import Image, ImageDraw, ImageFont

from enum import IntEnum, Enum


# agent actions
class Actions(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NOOP = 4


# Settings
FPS = 10
DEBUG = False
# size of the (square) input
DIM = 32


# Map items
class MapItems(Enum):
    EMPTY = 0
    WALL = '#'
    PACMAN = 'P'
    GHOST_PATROL = 3
    GHOST_ADV = 4
    GHOST = 5
    DOOR = 'X'
    BULLET = 7
    CHERRY = 8
    INIT = 'I'


MapIndices = {e: i for i, e in enumerate(MapItems)}

# Render Config
WALLstr = u'═'
WALLVERTICALstr = u'\u2551'
EMPTYstr = u' '
PACMANstrleft = u'\u15e4'
PACMANstr = u'\u15e7'
GHOSTstr = u'\u15e3'
CHERRYstr = u"\U0001F352"
BULLETstr = u'\u25e6'
DOORstr = u'\u2592'

# for rendering
font = None
pil_font = None


def one_hot_observation(item: int, flatten: bool = False, dtype: DTypeLike = int) -> np.ndarray:
    """
    Encode the observation space in one-hot

    Args:
        item: the item observed in the current observation
        flatten: whether to flatten the observation space or not
        dtype: observation dtype

    Returns:

    """
    one_hot_array = np.eye(len(MapItems), dtype=dtype)[item]
    if flatten:
        return one_hot_array.flatten()
    else:
        return one_hot_array


class Agent:
    def __init__(self, x, y, map):
        self.x = x
        self.y = y
        self.map = map
        self.type = -1

    def get_new_position(self, act):
        x = self.x
        y = self.y
        if act == Actions.LEFT:
            x -= 1
        if act == Actions.RIGHT:
            x += 1
        if act == Actions.UP:
            y += 1
        if act == Actions.DOWN:
            y -= 1
        return x, y

    def update_position(self, act):
        x, y = self.get_new_position(act)
        self.x = x
        self.y = y
        return self.x, self.y

    def get_allowed_actions(self):
        acts = []
        if (
                0 <= self.x - 1 < len(self.map) and
                0 <= self.y < len(self.map[self.x - 1]) and
                self.map[self.x - 1][self.y] != MapItems.WALL
        ):
            acts.append(Actions.LEFT)
        if (
                0 <= self.x + 1 < len(self.map) and
                0 <= self.y < len(self.map[self.x + 1]) and
                self.map[self.x + 1][self.y] != MapItems.WALL
        ):
            acts.append(Actions.RIGHT)
        if (
                0 <= self.x < len(self.map) and
                0 <= self.y + 1 < len(self.map[self.x]) and
                self.map[self.x][self.y + 1] != MapItems.WALL
        ):
            acts.append(Actions.UP)
        if (
                0 <= self.x < len(self.map) and
                0 <= self.y - 1 < len(self.map[self.x]) and
                self.map[self.x][self.y - 1] != MapItems.WALL
        ):
            acts.append(Actions.DOWN)
        return acts

    def __str__(self):
        return str(self.type)


class Pacman(Agent):
    def __init__(self, x, y, map):
        Agent.__init__(self, x, y, map)
        self.type = 'O'
        self._str = PACMANstr

    def update_position(self, act):
        x, y = super().update_position(act)
        if act == Actions.LEFT:
            self._str = PACMANstrleft
        if act == Actions.RIGHT:
            self._str = PACMANstr
        return x, y

    def __str__(self):
        return self._str


class PatrolGhost(Agent):
    def __init__(self, x, y, map, k=10):
        """
        The ghost patrols for k steps right, then k steps left, and so on.
        """
        Agent.__init__(self, x, y, map)
        self.dir = True  # start by walking right
        self.counter = k
        self.k = k
        self.type = 'patrol'

    def get_act(self, pacman):
        if self.counter > 0:
            self.counter -= 1
            act = Actions.RIGHT if self.dir else Actions.LEFT
            if act in self.get_allowed_actions():
                return act

        self.dir = not self.dir
        self.counter = self.k
        return self.get_act(pacman)

    def __str__(self):
        return GHOSTstr


class AdversarialGhost(Agent):
    def __init__(self, x, y, map, k=10, l=1):
        """
        If the pacman is within k steps, it walks a step towards it
        The ghost is allowed to take one step every l time steps to slow it down.
        """
        Agent.__init__(self, x, y, map)
        self.l = l
        self.k = k
        self.counter = 0
        self.type = 'A'

    def get_act(self, pacman):
        if abs(self.x - pacman.x) + abs(self.y - pacman.y) <= self.k:
            # the ghost is only allowed to move when the counter = 0
            if self.counter > 0:
                self.counter -= 1
                return Actions.NOOP

            if pacman.y != self.y:
                act = Actions.UP if pacman.y > self.y else Actions.DOWN
                if act in self.get_allowed_actions():
                    self.counter = self.l
                    return act

            act = Actions.RIGHT if pacman.x > self.x else Actions.LEFT
            if act in self.get_allowed_actions():
                self.counter = self.l
                return act
            else:
                return Actions.NOOP

    def __str__(self):
        return GHOSTstr


ghosts = {MapItems.GHOST_ADV: AdversarialGhost, MapItems.GHOST_PATROL: PatrolGhost}


class PacmanEnv(gym.Env):
    """
    PacMan Room.
    The goal is to escape the room by one of the door by maximizing the score and avoiding being eaten by ghosts.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            flat_observation_space: bool = False,
            reward_shaping: bool = True,
            gamma: float = 1.,
            map_path: Optional[str] = None,
            max_episode_steps: int = 200,
    ):
        self.initial_positions = []
        self._read_map(filename=map_path)
        self._flatten = flat_observation_space
        if flat_observation_space:
            self.observation_space = spaces.Box(
                low=0,
                # one-hot but several (identical) items can be stacked
                # and therefore added up (e.g., several ghosts on the same position)
                high=DIM * DIM,
                shape=(DIM * DIM, len(MapIndices),),
                dtype=np.int32)
        else:
            self.observation_space = spaces.Box(
                low=0,
                # see comment above
                high=DIM * DIM,
                shape=(DIM, DIM, len(MapIndices),),
                dtype=np.int32)

        self.action_space = spaces.Discrete(5)

        self.score = 0.
        self.reward_shaping = reward_shaping
        self.gamma = gamma
        self._max_episode_steps = max_episode_steps
        self._step_counter = 0

        # if goal positions are provided, not all the doors will be considered as goal states,
        # but only those provided in goal position
        self._goal_positions = np.transpose(np.where(self.map == MapItems.DOOR))

    @property
    def goal_positions(self):
        return list(self._goal_positions)

    @goal_positions.setter
    def goal_positions(self, value):
        self._goal_positions = np.array(value)

    def state_to_goal(self, state: np.typing.ArrayLike):
        assert not self._flatten, NotImplemented
        x, y = np.concatenate(state[..., MapItems.PACMAN].nonzero(), axis=-1)
        return x, y

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def distance_fn(
            self,
            x: Union[Tuple[int, int], Collection[Tuple[int, int]]],
            y: Union[Tuple[int, int], Collection[Tuple[int, int]]],
    ) -> float:
        x = np.array(x)
        y = np.array(y)
        dist_to_goal = np.min(np.linalg.norm(x - y, ord=1, axis=-1))
        if self._is_occupied(self.pacman.x, self.pacman.y, self.ghosts):
            if self.max_episode_steps == np.inf:
                return dist_to_goal / (1. - self.gamma)
            else:
                return sum([
                    self.gamma ** i * dist_to_goal
                    for i in range(self.max_episode_steps - self._step_counter)])
        else:
            return dist_to_goal

    def labeling_fn(self, obs: np.typing.ArrayLike, include_safe_flag: bool = True):
        return np.stack(
            ([
                 # safe
                 np.all(np.logical_not(
                     np.logical_and(
                         obs[..., MapIndices[MapItems.PACMAN]],
                         obs[..., MapIndices[MapItems.GHOST]])), ),
             ] if include_safe_flag else []) + [
                # goal
                (self.pacman.x, self.pacman.y) in [tuple(goal) for goal in self.goal_positions]
            ], axis=-1)

    def _read_map(self, filename=None):
        self.ghosts = []

        if filename is None:
            import importlib.resources as pkg_resources
            ls = pkg_resources.open_text(__package__, 'map.txt')
        else:
            ls = open(filename, 'r').readlines()
        ls = [line.strip() for line in ls]
        lines = []
        for line in ls:
            lines.append([x for x in line])

        def get_item(symbol: str):
            try:
                symbol_ = int(symbol)
            except ValueError:
                symbol_ = symbol
            return next(filter(lambda item: item.value == symbol_, MapItems))

        # note that the x and y indices are inverted in the loaded list (lines)
        self.map = np.full(
            shape=(len(lines[0]), len(lines)),
            fill_value=MapItems.EMPTY,
            dtype=object)

        for x in range(self.map.shape[0]):
            for y in range(self.map.shape[1]):
                item = get_item(lines[y][x])
                if item in (MapItems.DOOR, MapItems.CHERRY, MapItems.BULLET, MapItems.WALL):
                    self.map[x][y] = item
                elif item == MapItems.INIT:
                    self.initial_positions.append((x, y))
                elif item not in [MapItems.DOOR, MapItems.EMPTY]:
                    agent = item
                    if agent in ghosts.keys():
                        self.ghosts.append(ghosts[agent](x, y, self.map))
                    elif agent == MapItems.PACMAN:
                        self.pacman = Pacman(x, y, self.map)

        self.agents: List[Agent] = [a for a in self.ghosts]
        self.agents.append(self.pacman)

    def _is_occupied(self, x, y, agents=None):
        """
        is (x,y) occupied by some agent?
        To ask: occupied by ghost, call to "agents=self.ghosts"
        """
        if not agents:
            agents = self.agents

        for a in agents:
            if np.all(a.x == x) and np.all(a.y == y):
                return a

        return False

    def _get_state(self):
        """
        returns the one-hot encoded map in raw format DIM x DIM x N_ITEMS (or flattened if set)
        """
        observation = np.zeros(shape=(DIM, DIM, len(MapIndices)), dtype=np.uint8)

        for x in range(self.map.shape[0]):
            for y in range(self.map.shape[1]):
                observation[x, y, MapIndices[self.map[x, y]]] = 1
                a = self._is_occupied(x, y, agents=self.ghosts)
                if a:
                    observation[x, y, MapIndices[MapItems.GHOST]] += 1
                if x == self.pacman.x and y == self.pacman.y:
                    observation[x, y, MapIndices[MapItems.PACMAN]] = 1

        if self._flatten:
            observation = observation.flatten()

        return observation

    def did_lose(self, x: Optional[int] = None, y: Optional[int] = None):
        if x is None or y is None:
            x = self.pacman.x
            y = self.pacman.y

        return (
            # outside the map
                x < 0 or y < 0 or
                x >= len(self.map) or y >= len(self.map[x]) or
                # on a wall
                self.map[x][y] == MapItems.WALL or
                # eaten by a ghost
                self._is_occupied(x, y, self.ghosts) or
                # time is up
                self._step_counter > self.max_episode_steps
        )

    def step(self, action):
        self._step_counter += 1

        if action in self.pacman.get_allowed_actions():
            self.pacman.update_position(action)
        else:
            self.pacman.update_position(Actions.NOOP)

        done = self.did_lose()

        # position
        x, y = self.pacman.x, self.pacman.y

        if not done:
            for g in self.ghosts:
                act = g.get_act(self.pacman)
                _x, _y = g.get_new_position(act)
                g.update_position(act)
            done = self.did_lose()

        rew = 0.

        # Goal
        if (x, y) in [tuple(goal) for goal in self.goal_positions]:
            rew = 1.
            done = True
        elif self.reward_shaping:
            rew -= self.distance_fn((x, y), self.goal_positions)

        # consumable
        if self.map[x][y] in [MapItems.BULLET, MapItems.CHERRY]:
            rew /= {
                MapItems.BULLET: 1.5,
                MapItems.CHERRY: 3.,
            }[self.map[x][y]]
            self.map[x][y] = MapItems.EMPTY

        self.score += rew

        obs = self._get_state()

        return obs, rew, done, {}

    def reset(self, map_file_name: Optional[str] = None, **kwargs):
        self._read_map(filename=map_file_name)
        self.score = 0.
        self._step_counter = 0

        if self.initial_positions:
            x, y = self.initial_positions[np.random.choice(len(self.initial_positions))]
        else:
            while True:
                x = random.choice(range(len(self.map)))
                y = random.choice(range(len(self.map[x])))
                if not self.did_lose(x, y):
                    break

        self.pacman.x = x
        self.pacman.y = y
        return self._get_state()

    def __str__(self):

        rows = []
        for y in range(len(self.map[0])):
            row = []
            for x in range(len(self.map)):
                if self.map[x][y] == MapItems.WALL:
                    # if x in [0, len(self.map) - 1] and y != 0:
                    #     row += WALLVERTICALstr
                    # else:
                    row += WALLstr
                    continue
                a = self._is_occupied(x, y)
                if a:
                    row += str(a)
                elif self.map[x][y] in [MapItems.DOOR, MapItems.BULLET, MapItems.CHERRY]:
                    row += {
                        MapItems.DOOR: DOORstr,
                        MapItems.BULLET: BULLETstr,
                        MapItems.CHERRY: CHERRYstr,
                    }[self.map[x][y]]
                else:
                    row += EMPTYstr

            rows.append(row)

        if self.did_lose():
            return u"""
          _____                         ____                 
         / ____|                       / __ \                
        | |  __  __ _ _ __ ___   ___  | |  | |_   _____ _ __ 
        | | |_ |/ _` | '_ ` _ \ / _ \ | |  | \ \ / / _ \ '__|
        | |__| | (_| | | | | | |  __/ | |__| |\ V /  __/ |   
         \_____|\__,_|_| |_| |_|\___|  \____/  \_/ \___|_|   
        """ + '\n' * (len(rows) - 6)

        return tabulate(rows, tablefmt='plain', stralign='center', numalign='center')

    def render(self, mode='human', close=False):
        global font
        global pil_font
        _str = str(self)
        _str += f'\nSCORE: {self.score:.3f}:'
        _str += u'  |  [ safe win] :: '
        _str += str(self.labeling_fn(self._get_state(), include_safe_flag=True))
        _str += u'  | goal positions: '
        _str += str(self.goal_positions)
        _str += '\n'
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
            if self.did_lose():
                time.sleep(2.)
        elif mode == 'rgb_array':
            scale = 20
            size_x, size_y = [len(self.map) * scale, len(self.map[0]) * scale]
            font_size = int(scale / 1.7)
            if font is None:
                font = os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    'fonts',
                    'Unifont.ttf'))
            if pil_font is None:
                pil_font = ImageFont.truetype(
                    font,
                    size=font_size)
            canvas = Image.new('RGB', (size_x, size_y), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            white = "#000000"
            draw.text((0, 0), _str, font=pil_font, fill=white)
            return np.asarray(canvas)


if __name__ == '__main__':
    import reinforcement_learning.environments
    from tf_agents.environments import suite_gym, tf_py_environment
    from tf_agents.policies import random_tf_policy
    from tf_agents.drivers import dynamic_episode_driver
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import trajectory

    with suite_gym.load("PacmanRoom-v0") as py_env:
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

        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            env=tf_env,
            policy=tf_policy,
            observers=[
                lambda _: py_env.render(mode='human'),
                lambda trajectory: rb.add_batch(trajectory)
            ],
            num_episodes=2
        ).run()
        pass
