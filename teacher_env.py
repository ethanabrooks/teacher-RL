import itertools
from typing import Generator

import gym
import numpy as np
from gym.utils import seeding

from egreedy import EGreedy


class TeacherEnv(gym.Env):
    def __init__(
        self, choices: int, num_bandits: int, data_size: int, min_reward=1, max_reward=0
    ):
        super().__init__()
        assert num_bandits == 1
        self.choices = choices
        self.num_bandits = num_bandits
        self.random, self._seed = seeding.np_random(0)
        self._max_episode_steps = np.inf
        self.iterator = None
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.data_size = data_size
        # self.observation_space = gym.spaces.Box(
        #     low=np.tile(
        #         np.array([0] + [min_reward] * self.choices, dtype=np.float32),
        #         (self.num_bandits, 1),
        #     ),
        #     high=np.tile(
        #         np.array([choices - 1] + [max_reward] * self.choices, dtype=np.float32),
        #         (self.num_bandits, 1),
        #     ),
        # )
        self.observation_space = gym.spaces.Box(
            low=np.array([0] + [min_reward] * (1 + self.choices), dtype=np.float32),
            high=np.array(
                [choices + 1] + [max_reward] * (1 + self.choices), dtype=np.float32
            ),
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros(num_bandits, dtype=np.float32),
            high=np.ones(num_bandits, dtype=np.float32),
        )
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
        size = self.num_bandits, self.choices
        # half = int(len(self.dataset) // 2)
        loc = np.zeros((len(self.dataset), *size))
        loc[:, :, int(self.random.choice(self.choices))] = 1
        # half = len(self.dataset) - half
        # loc2 = np.random.normal(size=(half, *size), scale=1)
        # loc = np.vstack([loc1, loc2])
        self.dataset = self.random.normal(loc)
        our_loop = self.bandit.train_loop(dataset=self.dataset)
        base_loop = self.bandit.train_loop(dataset=self.dataset)
        optimal = loc.max(axis=-1, initial=-np.inf)

        baseline_return = np.zeros(self.num_bandits)

        next(our_loop)
        next(base_loop)
        action = np.ones(self.num_bandits)

        done = False
        interaction = our_loop.send(action)

        for t in itertools.count():
            choices, rewards, reward_avg = interaction
            baseline_actions, baseline_rewards, _ = base_loop.send(0.1)
            chosen_means = loc[t][
                np.arange(self.num_bandits), choices.astype(int).flatten()
            ].reshape(self.num_bandits)
            baseline_chosen_means = loc[t][
                np.arange(self.num_bandits), baseline_actions.astype(int).flatten()
            ].reshape(self.num_bandits)
            baseline_return += np.mean(baseline_rewards)

            s = np.concatenate([choices, rewards, reward_avg.flatten()], axis=-1)
            r = np.mean(rewards)
            i = dict(
                baseline_regret=np.mean(optimal[t : t + 1] - baseline_chosen_means),
                baseline_rewards=np.mean(baseline_rewards),
                regret=np.mean(optimal[t : t + 1] - chosen_means),
                rewards=np.mean(rewards),
                coefficient=np.mean(action).item(),
            )
            try:
                interaction = our_loop.send(action)
                self._max_episode_steps = t
            except StopIteration:  # StopIteration
                done = True

            if done:
                i.update(baseline_return=baseline_return)

            action = yield s, r, done, i

    def render(self, mode="human"):
        pass
