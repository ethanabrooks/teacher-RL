import argparse

import gym

from networks import CopyAgent
from trainer import Trainer

from main import add_arguments
from wrappers import TupleActionWrapper


def main(**kwargs):
    class CopyTrainer(Trainer):
        def make_env(self, env_id, seed, rank, evaluation):
            assert env_id == "Copy-v0"
            return TupleActionWrapper(gym.make(env_id))

        @staticmethod
        def build_agent(envs, **agent_args):
            return CopyAgent(envs.observation_space, envs.action_space, **agent_args)

    return CopyTrainer.main(**kwargs)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    add_arguments(PARSER)
    main(**vars(PARSER.parse_args()))
