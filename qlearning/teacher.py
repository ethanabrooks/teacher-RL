from typing import Generator

import gym
import numpy as np
from gym.spaces import Box

from qlearning.algorithm import QLearning


class TeacherEnv(gym.Env):
    def __init__(
        self,
        seed: int,
        training_iterations: int,
        env_id: str,
        alpha: float,
        gamma: float,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.env_id = env_id
        self.training_iterations = training_iterations
        self.q_learning = QLearning(seed)
        self.iterator = None
        env = gym.make(env_id)
        num_obs = env.observation_space.n
        num_act = env.action_space.n
        self.observation_space = Box(
            low=np.concatenate(
                [np.array([0, 0]), -np.inf * np.ones((num_obs * num_act))]
            ),
            high=np.concatenate(
                [np.array([num_obs, 1]), np.inf * np.ones((num_obs * num_act))]
            ),
        )
        self.action_space = env.action_space

    def seed(self, seed=None):
        pass

    def reset(self):
        self.iterator = self.generator()
        s, _, _, _ = next(self.iterator)
        return s

    def step(self, action):
        return self.iterator.send(action)

    def render(self, mode="human"):
        pass

    def generator(self):
        env = gym.make(self.env_id)
        eval_env = gym.make(self.env_id)
        iterator = self.q_learning.train_loop(
            env, eval_env, alpha=self.alpha, gamma=self.gamma
        )
        q, s, d, r = next(iterator)
        info = dict()
        for i in range(self.training_iterations):
            s = np.concatenate([[s, d], q.flatten()])
            a = yield s, r, False, info
            q, s, d, r = iterator.send(a)
            info = dict(q=q[s, a])
            if d:
                info.update(eval_return=r)
        s = np.concatenate([[s, d], q.flatten()])
        yield s, r, True, {}
