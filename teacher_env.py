import itertools
from typing import Generator

import gym
import numpy as np
from gym.utils import seeding

from egreedy import EGreedy


class TeacherEnv(gym.Env):
    def __init__(
        self, choices: int, data_size: int, dataset: str, min_reward=1, max_reward=0
    ):
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
        self.dataset_type = dataset
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
        self, const_eps=0.1, initial_linear_eps=0.3, initial_exp_eps=1, exp_anneal=0.99
    ) -> Generator:
        size = 1, self.choices
        # half = int(len(self.dataset) // 2)
        if self.dataset_type == "01":
            loc = np.zeros((len(self.dataset), *size))
            loc[:, :, int(self.random.choice(self.choices))] = 1
            self.dataset[:] = self.random.normal(loc)
        elif self.dataset_type == "sb":
            loc = np.tile(self.random.normal(size=size), (len(self.dataset), 1, 1))
            self.dataset[:] = self.random.normal(loc)
        else:
            raise NotImplementedError
        # half = len(self.dataset) - half
        # loc2 = np.random.normal(size=(half, *size), scale=1)
        # loc = np.vstack([loc1, loc2])
        our_loop = self.bandit.train_loop(dataset=self.dataset)
        const_loop = self.bandit.train_loop(dataset=self.dataset)
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
        exp_eps = initial_exp_eps

        for t in itertools.count():

            def compute_rewards_regret(_choices, _reward):
                chosen_means = loc[t, 0][_choices.astype(int).flatten()]
                _regret = optimal[t : t + 1] - chosen_means
                return np.mean(_reward), np.mean(_regret)

            our_choices, our_rewards = interaction
            const_reward, const_regret = compute_rewards_regret(
                *const_loop.send(const_eps)
            )
            linear_reward, linear_regret = compute_rewards_regret(
                *linear_loop.send(linear_eps)
            )
            linear_eps -= initial_linear_eps / len(self.dataset)
            linear_return += linear_reward
            exp_reward, exp_regret = compute_rewards_regret(*exp_loop.send(exp_eps))
            exp_eps *= exp_anneal
            exp_return += exp_reward

            reward, regret = compute_rewards_regret(our_choices, our_rewards)

            s = np.concatenate([our_choices, our_rewards], axis=-1)
            i = dict(
                const_regret=const_regret,
                const_rewards=const_reward,
                linear_regret=linear_regret,
                linear_rewards=linear_reward,
                exp_regret=exp_regret,
                exp_rewards=exp_reward,
                our_regret=regret,
                our_rewards=reward,
                our_epsilon=np.mean(action).item(),
                const_epsilon=const_eps,
                linear_epsilon=linear_eps,
                exp_epsilon=exp_eps,
            )
            try:
                interaction = our_loop.send(action)
                self._max_episode_steps = t
            except StopIteration:  # StopIteration
                done = True

            if done:
                i.update(linear_return=linear_return, exp_return=exp_return)

            action = yield s, reward, done, i

    def render(self, mode="human"):
        pass
