import itertools
from typing import Generator

import gym
import numpy as np
from gym.utils import seeding

from egreedy import EGreedy


class TeacherEnv(gym.Env):
    def __init__(self, choices: int, data_size: int, min_reward=1, max_reward=0):
        super().__init__()
        self.choices = choices
        self.random, self._seed = seeding.np_random(0)
        self._max_episode_steps = np.inf
        self.iterator = None
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.data_size = data_size
        self.observation_space = gym.spaces.Box(
            low=np.array([0, min_reward], dtype=np.float32),
            high=np.array([choices + 1, max_reward], dtype=np.float32),
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros(1, dtype=np.float32), high=np.ones(1, dtype=np.float32)
        )
        self.bandit = EGreedy(self._seed)
        self.dataset = np.zeros((data_size, 1, self.choices))

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

    def _generator(
        self, initial_linear_eps=0.3, initial_exp_eps=0.9, exp_anneal=0.99
    ) -> Generator:
        size = 1, self.choices
        # half = int(len(self.dataset) // 2)
        loc = np.zeros((len(self.dataset), *size))
        loc[:, :, int(self.random.choice(self.choices))] = 1
        # half = len(self.dataset) - half
        # loc2 = np.random.normal(size=(half, *size), scale=1)
        # loc = np.vstack([loc1, loc2])
        self.dataset = self.random.normal(loc, scale=2)
        our_loop = self.bandit.train_loop(dataset=self.dataset)
        linear_loop = self.bandit.train_loop(dataset=self.dataset)
        exp_loop = self.bandit.train_loop(dataset=self.dataset)
        optimal = loc.max(axis=-1, initial=-np.inf)

        linear_return = np.zeros(1)
        exp_return = np.zeros(1)

        next(our_loop)
        next(linear_loop)
        next(exp_loop)
        action = np.ones(1)

        done = False
        interaction = our_loop.send(action)

        linear_eps = initial_linear_eps

        for t in itertools.count():

            def compute_rewards_regret(_choices, _reward):
                chosen_means = loc[t, 0][_choices.astype(int).flatten()]
                _regret = optimal[t : t + 1] - chosen_means
                return np.mean(_reward), np.mean(_regret)

            our_choices, our_rewards = interaction
            linear_reward, linear_regret = compute_rewards_regret(
                *linear_loop.send(linear_eps)
            )
            linear_eps -= initial_linear_eps / len(self.dataset)
            linear_return += linear_reward

            reward, regret = compute_rewards_regret(our_choices, our_rewards)

            s = np.concatenate([our_choices, our_rewards], axis=-1)
            i = dict(
                linear_regret=linear_regret,
                linear_rewards=linear_reward,
                regret=regret,
                rewards=reward,
                coefficient=np.mean(action).item(),
            )
            try:
                interaction = our_loop.send(action)
                self._max_episode_steps = t
            except StopIteration:  # StopIteration
                done = True

            if done:
                i.update(linear_return=linear_return)

            action = yield s, reward, done, i

    def render(self, mode="human"):
        pass
