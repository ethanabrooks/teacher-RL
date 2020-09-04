import itertools

import argparse
import gym
from gym.spaces import Discrete


class Env(gym.Env):
    def __init__(self, num_states):
        super().__init__()
        self.iterator = None
        self.num_states = num_states
        self.observation_space = Discrete(num_states)
        self.action_space = Discrete(2)
        self._render = None

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def render(self, mode="human"):
        self._render()

    def generator(self):
        s = 0
        action = 0
        for stage in itertools.count():
            for step in range(stage):
                if action:
                    r = -1 if step else stage
                    t = False or (s + 1) == self.num_states
                else:
                    r = stage
                    t = True

                def render():
                    print("state:", s)
                    print("reward:", r)
                    print("done:", t)
                    print()

                self._render = render
                action = yield s, r, t, {}
                s += 1
