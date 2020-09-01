import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from gym.utils import seeding
from tensorboardX import SummaryWriter

"""
Created on Tue Sep 11 17:00:29 2018

@author: Mohammad Doosti Lakhani
"""


class EGreedy:
    def __init__(self, seed=0):
        self.random, _ = seeding.np_random(seed)

    def train_loop(self, dataset: np.ndarray, epsilon=0.1):
        # Implementing Upper Bound Confidence
        T, n, d = dataset.shape

        choices = np.ones((n, d), dtype=np.float16)  # number of selection of ad i
        rewards = np.zeros((n, d))
        arange = np.arange(n)

        # first sample each
        e = 1
        for i in range(d):
            rewards[:, i] = dataset[i, :, i]
            e = yield i * np.ones(n), dataset[i, :, i]

        # implementation in vectorized form
        for i, data in enumerate(dataset[d:], start=d):
            r = rewards / choices
            # delta = (np.log(i + 1) / choices) ** (1 / e)
            # upper_bound = r + e * delta
            # choice = np.argmax(upper_bound, axis=-1)
            if e is None:
                e = epsilon
            greedy = self.random.random(size=n) > e
            random = self.random.choice(d, size=n)
            choice = greedy * np.argmax(r, axis=-1) + (1 - greedy) * random
            choices[arange, choice] += 1
            reward = data[arange, choice]
            rewards[arange, choice] += reward
            e = yield choice, reward


def run(dataset, loc, optimal, path):
    with SummaryWriter(logdir=str(path)) as writer:
        bandit = EGreedy(0).train_loop(dataset)
        next(bandit)
        cumulative_regret = 0
        for t in itertools.count():
            i = yield
            try:
                interaction = bandit.send(i)
            except StopIteration:
                break
            choices, rewards = interaction
            chosen_means = loc[t, 0][choices.astype(int).flatten()].flatten()
            regret = optimal[t : t + 1] - chosen_means
            cumulative_regret = regret.flatten() + cumulative_regret
            writer.add_scalar("reward", np.mean(rewards), t)
            writer.add_scalar("regret", np.mean(regret), t)
            writer.add_scalar("cumulative regret", np.mean(cumulative_regret), t)
            writer.add_scalar("i", i, t)


def main():
    T = 2000
    n = 1000
    d = 10

    np.random.seed(0)

    means = np.zeros((n, d))
    means[:, int(np.random.choice(d))] = 1
    loc = np.tile(means, (T, 1, 1))
    dataset = np.random.normal(loc, 1)
    optimal = loc.max(axis=-1, initial=-np.inf)

    for i in [0.01, 0.05, 0.1, 0.2, 0.25, 0.3]:
        path = Path("/tmp/log/const", str(i))
        runner = run(dataset, loc, optimal, path)
        next(runner)
        print("const i:", i)
        while True:
            try:
                runner.send(i)
            except StopIteration:
                break

    for initial_i in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for i_0_timestep in range(100, n, 100):
            path = Path("/tmp/log/linear", str(initial_i), str(i_0_timestep))
            runner = run(dataset, loc, optimal, path)
            print("initial i:", initial_i)
            print("i_0_timestep:", i_0_timestep)
            next(runner)
            i = initial_i
            while True:
                try:
                    runner.send(i)
                    i -= initial_i / i_0_timestep
                    i = max(i, 0)
                except StopIteration:
                    break

    for initial_i in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for rate in [0.999, 0.99, 0.95, 0.9, 0.8]:
            path = Path("/tmp/log/exp", str(initial_i), str(rate))
            runner = run(dataset, loc, optimal, path)
            print("initial i:", initial_i)
            print("rate:", rate)
            next(runner)
            i = initial_i
            while True:
                try:
                    runner.send(i)
                    i *= rate
                except StopIteration:
                    break


if __name__ == "__main__":
    main()
