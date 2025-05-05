import abc

import gym
import numpy as np
from numpy.typing import ArrayLike


class GoalAugmentedEnvironment(gym.Env):

    @abc.abstractmethod
    def state_to_goal(self, state: ArrayLike, batched: bool = False, lib=np):
        return NotImplemented

    @abc.abstractmethod
    def is_achieved(self, state: ArrayLike, goal: ArrayLike, batched: bool = False, lib=np):
        return NotImplemented

    @property
    @abc.abstractmethod
    def current_goal(self):
        return NotImplemented

    @abc.abstractmethod
    def is_unsafe(self, state: ArrayLike, lib=np) -> bool:
        return False
