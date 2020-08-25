import json
from pathlib import Path

from hyperopt import hp


def get_config(name):
    path = Path("configs", name).with_suffix(".json")
    if path.exists():
        with path.open() as f:
            config = json.load(f)
            del config["use_tune"]
            return config
    return configs[name]


def copy_args(d, prefix):
    for k, v in d.items():
        yield prefix + k, v


def small_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** -i), 5 * (10 ** -i))]


def medium_values(start, stop):
    return [2 ** i for i in range(start, stop)]


def big_values(start, stop):
    return [j for i in range(start, stop) for j in ((10 ** i), 5 * (10 ** i))]


search = dict(
    learning_rate=hp.choice("learning_rate", small_values(2, 5)),
    seed=hp.randint("seed", 20),
    train_steps=hp.choice("train_steps", [5, 10, 20, 25, 30, 35, 40]),
    entropy_coef=hp.choice("entropy_coef", [0.01, 0.02]),
    hidden_size=hp.choice("hidden_size", [64, 128, 256]),
    num_layers=hp.choice("num_layers", [1, 2]),
    use_gae=hp.choice("use_gae", [True, False]),
    clip_param=hp.choice("clip_param", [0.1, 0.2]),
    ppo_epoch=hp.choice("ppo_epoch", [1, 2, 4, 5, 7, 10]),
)
pendulum = dict(
    learning_rate=0.01,
    seed=0,
    train_steps=20,
    entropy_coef=0.01,
    hidden_size=128,
    num_layers=2,
    use_gae=True,
    clip_param=0.2,
    ppo_epoch=2,
)
configs = dict(search=search, pendulum=pendulum)
