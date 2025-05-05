import abc
import enum
from typing import Collection, Dict, Tuple, Optional, Union

import gym
import numpy as np
from numpy.typing import ArrayLike


class Directions(enum.IntEnum):
    """
    Cardinal directions
    """
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    NOOP = 4

    def opposite(self):
        """
        Get the opposite direction.
        """
        return {
            Directions.RIGHT: Directions.LEFT,
            Directions.DOWN: Directions.UP,
            Directions.LEFT: Directions.RIGHT,
            Directions.UP: Directions.DOWN,
            Directions.NOOP: Directions.NOOP
        }[self]

    def next_clockwise_direction(self):
        """
        Get the next clockwise direction.
        """
        return Directions((self + 1) % 4)

    def next_position(self, i: int, j: int):
        """
        Get the next position in the given direction.
        """
        i += int(self is Directions.DOWN)
        i -= int(self is Directions.UP)
        j += int(self is Directions.RIGHT)
        j -= int(self is Directions.LEFT)
        return i, j

    @staticmethod
    def random():
        """
        Get a random direction.
        """
        return Directions(np.random.choice([dir for dir in Directions]))

    @staticmethod
    def relative(v0: ArrayLike, v: ArrayLike):
        """
        Get the direction from v0 to v.
        """
        v0 = np.array(v0)
        v = np.array(v)
        assert v0.size == v.size == 2, "The vectors v and v0 should be points in a 2D space."
        direction_vector = v - v0
        slope = direction_vector[0] / direction_vector[1] if direction_vector[1] != 0 else np.inf
        if direction_vector[0] <= 0 and np.abs(slope) >= 1:
            return Directions.UP
        elif direction_vector[0] > 0 and np.abs(slope) >= 1:
            return Directions.DOWN
        elif direction_vector[1] >= 0 and np.abs(slope) < 1:
            return Directions.RIGHT
        elif direction_vector[1] < 0 and np.abs(slope) < 1:
            return Directions.LEFT
        else:
            return Directions.NOOP

    @staticmethod
    def from_cardinal(cardinal: str):
        """
        Convert a cardinal direction to a direction.
        """
        return {
            'north': Directions.UP,
            'south': Directions.DOWN,
            'east': Directions.RIGHT,
            'west': Directions.LEFT
        }[cardinal.lower()]

    def to_cardinal(self):
        """
        Convert the direction to a cardinal direction.
        """
        return ('east', 'south', 'west', 'north')[int(self)]


class TwoLevelEnv(gym.Env):

    # direction from which the agent entered the current room
    input_direction: Directions

    # Directions in which the agent can move from the current room.
    # Note: must be equivalent to self.get_available_directions(current_room),
    # where current_room is the room in which the agent is located.
    available_directions: Collection[Directions]

    # directions from which the agent can enter each *initial* room, i.e., the room(s) in which the agent starts
    initial_directions: Dict[int, Collection[Directions]]

    # whether the environment has been hard reset or not
    has_been_hard_reset: bool

    # Set of rooms in the environment
    rooms: Collection

    n_rooms: int
    current_room: int
    between_two_rooms: bool = False
    default_action: Union[int, float] = 0

    @abc.abstractmethod
    def get_next_room(self, room_number: int, direction: Directions) -> int:
        """
        Get the number of the room that is in the given direction from the input room.
        Args:
            room_number: number of the input room
            direction: direction to which the agent wants to move
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_room(self) -> None:
        """
        Update the room in which the agent is located.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_available_directions(self, room_number: int) -> Collection[Directions]:
        """
        Get the directions in which the agent can move from the input room.
        Args:
            room_number: number of the input room

        Returns: a collection of directions in which the agent can move
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_low_level_objective(self, direction: Directions) -> None:
        """
        Set the low-level objective that can be achieved by the agent in the current room to the given direction.
        Args:
            direction: direction to which the agent wants to move
        """
        raise NotImplementedError

    @abc.abstractmethod
    def hard_reset(self, **kwargs) -> ArrayLike:
        """
        Reset the entire two-level environment, and not just the current room.
        This allows to specify different behaviors for resetting the environment: a call to reset() may reset the
        room in which the agent is located, while a call to hard_reset() may reset the entire two-level environment.
        If such a behavior is not needed, this method can be merely implemented as a call to reset().
        """
        raise NotImplementedError

    @abc.abstractmethod
    def should_hard_reset(self) -> bool:
        """
        Check if the environment should be hard reset.
        If this method returns True, the environment will be hard reset when calling hard_reset().
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_goal_direction(self, room_number: int) -> Optional[Directions]:
        """
        Get the direction in which the high-level objective is located.
        Args:
            room_number: number of the input room

        Returns: the goal direction if it is reachable from the input room, None otherwise.
        """
        raise NotImplementedError

    def get_entrance_direction(self, room_0, room_1, direction: Directions) -> Directions:
        """
        Assuming room_1 is entered from room_0, get the direction in which the agent enters room_1.
        """
        return Directions.opposite(direction)
