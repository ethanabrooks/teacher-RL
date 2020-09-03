import itertools
from collections import namedtuple
from pathlib import Path

import argparse
import gym
import numpy as np
from gym.envs.toy_text import NChainEnv
from gym.spaces import Discrete
from gym.utils import seeding
from gym.wrappers import TimeLimit
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
        training_iterations: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1,
        epsilon_decay: float = 0.9,
        min_epsilon: float = 0.001,
    ):
        # Implementing Upper Bound Confidence
        assert isinstance(env.action_space, Discrete)
        assert isinstance(env.observation_space, Discrete)
        q = np.zeros((env.observation_space.n, env.action_space.n))
        for _ in range(training_iterations):
            states = []
            actions = []
            rewards = []
            s = env.reset()
            t = 0
            d = False
            while not d:
                states.append(s)
                a = yield q, s, d, 0
                if a is None:
                    a = (
                        self.argmax(q[s])
                        if self.random.random() < epsilon
                        else env.action_space.sample()
                    )
                    epsilon *= epsilon_decay
                    epsilon = max(epsilon, min_epsilon)

                s, r, d, _ = env.step(a)

                actions.append(a)
                rewards.append(r)
                t += 1
                if d:
                    for state, action, reward, next_state, next_action in zip(
                        states, actions, rewards, states[1:], actions[1:]
                    ):
                        td_target = reward + gamma * max(q[next_state])
                        q[state, action] += alpha * (td_target - q[state, action])
                    state = states[-1]
                    action = actions[-1]
                    q[state, action] += alpha * (rewards[-1] - q[state, action])
                    evaluation = list(self.evaluate(eval_env, q))
                    yield q, s, d, sum(evaluation)
                    s = env.reset()
                    states = []
                    actions = []
                    rewards = []

    def evaluate(self, env: gym.Env, q: np.ndarray, render: bool = False):
        d = False
        s = env.reset()
        while not d:
            if render:
                env.render()
                input("waiting")
            s, r, d, i = env.step(self.argmax(q[s]))
            yield r


def main(env_id, iterations):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    env.seed(0)
    eval_env.seed(0)
    writer = SummaryWriter("/tmp/qlearning")
    episode = 0
    for i, (q, s, d, r) in enumerate(
        QLearning().train_loop(env, eval_env, training_iterations=iterations)
    ):
        if d:
            writer.add_scalar("return", r, episode)
            episode += 1
            # print(r)
    print(q)
    for _ in range(5):
        list(QLearning().evaluate(eval_env, q, render=True))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("env_id")
    PARSER.add_argument("--iterations", "-i", type=int)
    main(**vars(PARSER.parse_args()))
