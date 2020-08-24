from collections import defaultdict

import numpy as np


class EpochCounter:
    def __init__(self, num_processes):
        self.episode_rewards = []
        self.episode_time_steps = []
        self.rewards = np.zeros(num_processes)
        self.time_steps = np.zeros(num_processes)
        self.infos = defaultdict(list)

    def update(self, reward, done, infos):
        self.rewards += reward.numpy()
        self.time_steps += np.ones_like(done)
        self.episode_rewards += list(self.rewards[done])
        self.episode_time_steps += list(self.time_steps[done])
        self.rewards[done] = 0
        self.time_steps[done] = 0
        for info in infos:
            for k, v in info.items():
                self.infos[k].append(v)

    def reset(self):
        self.episode_rewards = []
        self.episode_time_steps = []
        self.infos = defaultdict(list)

    def items(self, prefix=""):
        if self.episode_rewards:
            yield prefix + "rewards", np.mean(self.episode_rewards)
        if self.episode_time_steps:
            yield prefix + "time_steps", np.mean(self.episode_time_steps)
        for k, vs in self.infos.items():
            yield prefix + k, np.mean(vs)
