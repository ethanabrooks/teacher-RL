import itertools
from collections import namedtuple
from pathlib import Path

import gym
import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from tensorboardX import SummaryWriter


class QLearning:
    def __init__(self, seed=0):
        self.random, _ = seeding.np_random(seed)

    def argmax(self, array: np.ndarray):
        max_val = array.max(initial=-np.inf)
        max_indices = np.arange(array.size)[array == max_val]
        return self.random.choice(max_indices)

    def train_loop(
        self,
        env: gym.Env,
        eval_env: gym.Env,
        max_timesteps: int,
        training_iterations: int,
        alpha: float,
    ):
        # Implementing Upper Bound Confidence
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Discrete)
        for _ in range(training_iterations):
            q = np.zeros((env.observation_space.n, env.action_space.n))
            s = env.reset()
            d = False
            r = 0
            t = 0
            states = np.zeros((max_timesteps, env.observation_space.n))
            actions = np.zeros((max_timesteps, env.action_space.n))
            rewards = np.zeros(max_timesteps)
            while True:
                a = yield q, s, 0
                if a is None:
                    a = (
                        self.argmax(q[s])
                        if self.random.random() < 0.1
                        else env.action_space.sample()
                    )

                s, d, r, _ = env.step(a)
                states[t, s] = 1
                actions[t, a] = 1
                rewards[t] = r
                t += 1
                if d:
                    for state, action, reward, next_state, next_action in zip(
                        states, actions, rewards, states[1:], actions[1:]
                    ):
                        q[state, action] += alpha * (
                            reward
                            + 0.99 * q[next_state, next_action]
                            - q[state, action]
                        )
                    q[states[-1], actions[-1]] += alpha * (
                        rewards[-1] - q[states[-1], actions[-1]]
                    )
                    yield q, s, sum(self.evaluate(eval_env, q))

    def act(self, s: int, q: np.ndarray, env: gym.Env):
        return (
            self.argmax(q[s])
            if self.random.random() < 0.1
            else env.action_space.sample()
        )

    def evaluate(self, env: gym.Env, q: np.ndarray):
        d = False
        s = env.reset()
        while not d:
            s, r, t, i = env.step(self.argmax(q[s]))
            yield r


def run(dataset, loc, optimal, path):
    with SummaryWriter(logdir=str(path)) as writer:
        bandit = QLearning(0).train_loop(dataset)
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
    T = 1000
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
