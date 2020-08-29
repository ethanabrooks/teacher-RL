import argparse
import pickle
from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
import epoch_counter
from main import add_arguments
from networks import TeacherAgent
from teacher_env import TeacherEnv
from trainer import Trainer
import numpy as np


class EpochCounter(epoch_counter.EpochCounter):
    infos_name = "infos"

    def __init__(self, num_processes):
        super().__init__(num_processes)
        keys = [
            "regret",
            "linear_regret",
            "exp_regret",
            "linear_rewards",
            "exp_rewards",
            "coefficient",
        ]
        self.info_lists = {k: [[] for _ in range(num_processes)] for k in keys}
        self.episode_lists = {k: [None for _ in range(num_processes)] for k in keys}

    def update(self, reward, done, infos):
        for k, lists in self.info_lists.items():
            for i, (info, d) in enumerate(zip(infos, done)):
                lists[i].append(info[k])
                if d:
                    self.episode_lists[k][i] = lists[i]
                    lists[i] = []

        return super().update(reward, done, infos)

    def items(self, prefix=""):
        episode_lists = {
            k: [x for x in v if x is not None] for k, v in self.episode_lists.items()
        }
        yield prefix + EpochCounter.infos_name, episode_lists
        yield from super().items(prefix)


def main(choices, data_size, **kwargs):
    class TeacherTrainer(Trainer, ABC):
        def step(self):
            result = super().step()

            for prefix in ("", "eval_"):
                name = prefix + EpochCounter.infos_name
                for k, v in result.pop(name).items():
                    path = Path(self.logdir, f"{prefix}_{k}")
                    np.save(str(path), np.array(v))

            return result

        def make_env(self, env_id, seed, rank, evaluation):
            env = TeacherEnv(choices=choices, data_size=data_size)
            env.seed(seed + rank)
            return env

        @staticmethod
        def build_agent(envs, **agent_args):
            return TeacherAgent(envs.observation_space, envs.action_space, **agent_args)

        @classmethod
        def build_epoch_counter(cls, num_processes):
            return EpochCounter(num_processes)

    kwargs.update(recurrent=True)
    TeacherTrainer.main(**kwargs)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--choices", "-d", type=int, default=10)
    PARSER.add_argument("--data-size", "-T", type=int, default=1000)
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
