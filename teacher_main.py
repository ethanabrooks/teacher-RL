import argparse
from abc import ABC

from tensorboardX import SummaryWriter

import epoch_counter
from main import add_arguments
from networks import TeacherAgent
from teacher_env import TeacherEnv
from trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt


class EpochCounter(epoch_counter.EpochCounter):
    def __init__(self, num_processes):
        super().__init__(num_processes)
        self.info_lists = dict(
            rewards=[[] for _ in range(num_processes)],
            baseline_rewards=[[] for _ in range(num_processes)],
            coefficient=[[] for _ in range(num_processes)],
        )
        self.info_arrays = {
            k: [None for _ in range(num_processes)] for k in self.info_lists
        }

    def update(self, reward, done, infos):
        for k, lists in self.info_lists.items():
            for i, (info, d) in enumerate(zip(infos, done)):
                lists[i].append(info[k])
                if d:
                    self.info_arrays[k][i] = np.array(lists[i])
                    lists[i] = []

        return super().update(reward, done, infos)

    def items(self, prefix=""):
        yield prefix + "info_arrays", self.info_arrays
        yield from super().items(prefix)


def main(choices, num_bandits, data_size, **kwargs):
    class TeacherTrainer(Trainer, ABC):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.writer = SummaryWriter(self.logdir)

        def log_result(self, result):
            def write_figure(prefix, dictionary):
                for k, arrays in dictionary.items():
                    arrays = [a for a in arrays if a is not None]
                    if arrays:
                        figure = plt.figure(figsize=(10, 10))
                        for array in arrays:
                            if array is not None:
                                plt.plot(array, color="green", alpha=0.2)
                        self.writer.add_figure(prefix + k, figure)

            write_figure("", result.pop("info_arrays"))
            write_figure("eval", result.pop("eval_info_arrays"))

            return result

        def make_env(self, env_id, seed, rank, evaluation):
            return TeacherEnv(
                choices=choices, num_bandits=num_bandits, data_size=data_size
            )

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
    PARSER.add_argument("--num-bandits", "-b", type=int, default=1)
    PARSER.add_argument("--data-size", "-T", type=int, default=1000)
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
