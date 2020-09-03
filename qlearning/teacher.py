import gym
import numpy as np
from gym.envs.registration import register
from gym.spaces import Box

from qlearning.algorithm import QLearning

register(
    id="FrozenLakeNotSlippery-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

register(
    id="LocalMinimaEnv-v0",
    entry_point="qlearning.local_maxima_env:Env",
    kwargs=dict(num_states=100),
)


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
            gym.make(self.env_id),
            alpha=self.alpha,
            gamma=self.gamma,
        )
        base_loop = self.q_learning.train_loop(
            gym.make(self.env_id),
            gym.make(self.env_id),
            alpha=self.alpha,
            gamma=self.gamma,
        )
        q, s, d, r = next(our_loop)
        _, _, d2, r2 = next(base_loop)
        info = dict()
        for i in range(self.training_iterations):
            if d:
                info.update(our_return=r)
            if d2:
                info.update(base_return=r2)
            s = np.concatenate([[s, d], q.flatten()])
            a = yield s, r, False, info
            q, s, d, r = our_loop.send(a)
            _, _, d2, r2 = next(base_loop)
            info = dict(q=q[s, a])
        s = np.concatenate([[s, d], q.flatten()])
        yield s, r, True, {}
