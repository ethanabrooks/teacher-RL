from typing import Generator

import gym
import numpy as np
from gym.utils import seeding

from bandit import EGreedy


class ChooseBestEnv(gym.Env):
    def __init__(
        self,
        choices: int,
        num_bandits: int,
        data_size: int,
        min_reward=-100,
        max_reward=100,
    ):
        super().__init__()
        self.choices = choices
        self.num_bandits = num_bandits
        self.random, self._seed = seeding.np_random(0)
        self._max_episode_steps = np.inf
        self.iterator = None
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.data_size = data_size
        self.observation_space = gym.spaces.Box(
            low=np.array([0, min_reward], dtype=np.float32),
            high=np.array([1 + self.choices, max_reward], dtype=np.float32),
        )
        self.action_space = gym.spaces.Discrete(2)
        self.bandit = EGreedy(self._seed)
        self.dataset = np.zeros((data_size, self.num_bandits, self.choices))

    def seed(self, seed=None):
        seed = seed or 0
        self.random, self._seed = seeding.np_random(seed)
        self.bandit = EGreedy(self._seed)

    def reset(self):
        self.iterator = self._generator()
        s, _, _, _ = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def _generator(self) -> Generator:
        best = self.random.choice(self.choices)
        for i in range(self.choices):
            yield (i, float(i == best)), 0, False, {}
        yield (0, self.choices), 0, False, {}
        r = 0
        for i in range(self.choices):
            t = i == self.choices - 1
            action = yield (i, 0), r, t, {}
            r = action if i == best else 1 - action

    def render(self, mode="human"):
        pass
