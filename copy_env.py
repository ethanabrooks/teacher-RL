import gym
from gym.spaces import Discrete
from gym.utils import seeding


class CopyEnv(gym.Env):
    def __init__(self, size):
        self.size = size
        self.random = self._seed = self.iterator = None
        self.observation_space = Discrete(3)
        self.action_space = Discrete(2)

    def reset(self):
        self.iterator = self.gen()
        s, _, _, _ = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def seed(self, seed=None):
        seed = seed or 0
        self.random, self._seed = seeding.np_random(seed)

    def gen(self):
        string = self.random.choice(2, size=self.size)
        for s in string:
            yield s, 0, False, {}
        action = yield 2, 0, False, {}
        for i, s in enumerate(string):
            r = float(action == s)
            t = i + 1 == len(string)
            action = 2, r, t, {}

    def render(self, mode="human"):
        pass
