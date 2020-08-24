import argparse

from main import add_arguments
from teacher_env import TeacherEnv
from trainer import Trainer
from wrappers import FlattenObs


def main(choices, num_bandits, data_size, **kwargs):
    class TeacherTrainer(Trainer):
        def make_env(self, env_id, seed, rank, evaluation):
            return FlattenObs(
                TeacherEnv(
                    choices=choices, num_bandits=num_bandits, data_size=data_size
                )
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
