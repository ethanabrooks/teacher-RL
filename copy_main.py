import argparse

from copy_env import CopyEnv
from networks import CopyAgent
from trainer import Trainer

from main import add_arguments


def main(size, **kwargs):
    class CopyTrainer(Trainer):
        def make_env(self, env_id, seed, rank, evaluation):
            return CopyEnv(size=size)

        @staticmethod
        def build_agent(envs, **agent_args):
            return CopyAgent(envs.observation_space, envs.action_space, **agent_args)

    return CopyTrainer.main(**kwargs)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--size", "-s", type=int, required=True)
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
