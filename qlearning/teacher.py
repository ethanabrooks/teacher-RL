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
        q_array = np.inf * np.ones((num_obs * num_act), dtype=np.float32)
        self.observation_space = Box(
            low=np.concatenate([np.array([0, 0], dtype=np.float32), -q_array]),
            high=np.concatenate([np.array([num_obs, 1], dtype=np.float32), q_array]),
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
        our_loop = self.q_learning.train_loop(
            gym.make(self.env_id),
            alpha=self.alpha,
            gamma=self.gamma,
        )
        base_loop = self.q_learning.train_loop(
            gym.make(self.env_id),
            alpha=self.alpha,
            gamma=self.gamma,
        )
        eval_env = gym.make(self.env_id)
        q, s, d = next(our_loop)
        bq, _, _ = next(base_loop)
        info = {}
        r = 0
        for i in range(self.training_iterations):
            info = {}
            if d:
                r = sum(self.q_learning.evaluate(eval_env, q))
                info.update(our_return=r, our_final_state=s)
            else:
                r = 0
            s = np.concatenate([[s, d], q.flatten()])
            a = yield s, r, False, info
            q, s, d = our_loop.send(a)
            bq, _, _ = next(base_loop)
        s = np.concatenate([[s, d], q.flatten()])
        info.update(base_return=sum(self.q_learning.evaluate(eval_env, bq)))
        yield s, r, True, info
