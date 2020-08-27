import argparse
from abc import ABC

from choose_best import ChooseBestEnv
from main import add_arguments
from networks import ChooseBestAgent
from trainer import Trainer


def main(choices, num_bandits, data_size, **kwargs):
    class TeacherTrainer(Trainer, ABC):
        def make_env(self, env_id, seed, rank, evaluation):
            return ChooseBestEnv(
                choices=choices, num_bandits=num_bandits, data_size=data_size
            )

        @staticmethod
        def build_agent(envs, **agent_args):
            return ChooseBestAgent(
                envs.observation_space, envs.action_space, **agent_args
            )

    kwargs.update(recurrent=True)
    TeacherTrainer.main(**kwargs)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--choices", "-d", type=int, default=10)
    PARSER.add_argument("--num-bandits", "-b", type=int, default=1)
    PARSER.add_argument("--data-size", "-T", type=int, default=1000)
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
