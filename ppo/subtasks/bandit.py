from collections import namedtuple

from gym import spaces
from rl_utils import hierarchical_parse_args
import numpy as np
from ppo.subtasks import control_flow, keyboard_control
from ppo.subtasks.lines import If, Else, EndIf, While, EndWhile, Subtask

Obs = namedtuple("Obs", "condition lines active")


class Bandit(control_flow.Env):
    def __init__(self, seed, n_lines, n_arms, flip_prob):
        super().__init__(seed, n_lines)
        self.flip_prob = flip_prob
        self.n_arms = n_arms
        self.condition_bit = None
        self.arms = None
        self.action_space = spaces.Discrete(n_arms)
        self.observation_space = spaces.Dict(
            dict(
                condition=spaces.Discrete(2),
                lines=spaces.MultiDiscrete(np.array([[6, n_arms]] * n_lines)),
                active=spaces.Discrete(n_lines + 1),
            )
        )
        self.line_types = [If, Else, EndIf, While, EndWhile, Subtask]

    def reset(self):
        self.condition_bit = self.random.randint(0, 2)
        self.arms = self.random.randint(low=0, high=self.n_arms, size=self.n_lines)
        return super().reset()

    def step(self, action):
        if action != self.arms[self.active]:
            return None, -1, True, {}
        if self.random.rand() < self.flip_prob:
            self.condition_bit = 1 - self.condition_bit
        return super().step(action)

    def get_observation(self):
        lines = [self.line_types.index(t) for t in self.lines]
        o = Obs(
            condition=self.condition_bit,
            lines=np.stack([lines, self.arms]).T,
            active=self.n_lines if self.active is None else self.active,
        )._asdict()
        assert self.observation_space.contains(o)
        return o

    def _evaluate_condition(self, i=None):
        return bool(self.condition_bit)

    def done(self):
        return True  # bandits are always done

    def line_strings(self, index, level):
        if index == len(self.lines):
            return
        line = self.lines[index]
        if line in [Else, EndIf, EndWhile]:
            level -= 1
        name = self.arms[index] if line is Subtask else line.__name__
        indent = ("> " if index == self.active else "  ") * level
        yield indent + str(name)
        if line in [If, While, Else]:
            level += 1
        yield from self.line_strings(index + 1, level)

    def render(self, mode="human"):
        print(*self.line_strings(index=0, level=1), sep="\n")
        print("Condition:", self.condition_bit)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-lines", default=6, type=int)
    parser.add_argument("--n-arms", default=3, type=int)
    parser.add_argument("--flip-prob", default=0.5, type=float)
    args = hierarchical_parse_args(parser)
    keyboard_control.run(Bandit(**args), actions="0123")
