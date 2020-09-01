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


def main(log_dir: Path):
    # dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

    T = 1000
    n = 1000
    d = 10

    means = np.zeros((n, d))
    np.random.seed(0)
    means[:, int(np.random.choice(d))] = 1
    loc = np.tile(means, (T, 1, 1))
    dataset = np.random.normal(loc, 2)
    optimal = loc.max(axis=-1, initial=-np.inf)
    cumulative_regret = 0

    # dataset = np.random.binomial(choices, p)
    for i in [0.001, 0.005, 0.01, 0.05, 0.1, 0.015, 0.2]:
        print("i:", i)
        path = Path(log_dir, "fixed", str(i))
        with SummaryWriter(logdir=str(path)) as writer:
            bandit = EGreedy(0).train_loop(dataset)
            next(bandit)
            for t in itertools.count():
                try:
                    interaction = bandit.send(i)
                except StopIteration:
                    print("t", t)
                    print("i", i)
                    break
                choices, rewards = interaction
                chosen_means = loc[t, 0][choices.astype(int).flatten()].flatten()
                regret = optimal[t : t + 1] - chosen_means
                if t > d:
                    cumulative_regret += np.mean(regret)
                writer.add_scalar("reward", np.mean(rewards), t)
                writer.add_scalar("regret", np.mean(regret), t)
                writer.add_scalar("i", i, t)
            writer.add_scalar("cumulative_regret", np.mean(cumulative_regret), t)

    choices, rewards = [np.stack(x) for x in zip(*EGreedy().train_loop(dataset))]

    with Path("results.npz").open("wb") as f:
        np.savez(f, selections=choices, rewards=rewards)


if __name__ == "__main__":
    main(Path("/tmp/log"))
