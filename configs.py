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


search = dict(
    learning_rate=hp.choice("learning_rate", [7e-4, 3e-4, 2.5e-4, 1e-3]),
    seed=hp.randint("seed", 20),
    train_steps=hp.choice("train_steps", [256, 512, 1024, 2048]),
    entropy_coef=hp.choice("entropy_coef", [0.01, 0.02]),
    hidden_size=hp.choice("hidden_size", [64, 128]),
    num_layers=hp.choice("num_layers", [1, 2, 3]),
    num_batch=hp.choice("num_batch", [1, 2]),
    use_gae=True,
    clip_param=0.2,
    ppo_epoch=hp.choice("ppo_epoch", [5, 10, 15, 20, 25]),
)
ppo_paper_mujoco = dict(
    learning_rate=3e-4,
    seed=hp.randint("seed", 20),
    train_steps=2048,
    entropy_coef=0.01,
    hidden_size=64,
    num_layers=2,
    num_batch=64,
    use_gae=True,
    clip_param=0.2,
    ppo_epoch=10,
    num_processes=128,
)
ppo_paper_roboschool = dict(
    learning_rate=hp.choice("learning_rate", [3e-4, 2.5e-4, 1e-3]),
    seed=hp.randint("seed", 20),
    train_steps=512,
    entropy_coef=0.01,
    hidden_size=64,
    num_layers=2,
    num_batch=4096,
    use_gae=True,
    clip_param=0.2,
    ppo_epoch=15,
    num_processes=128,
)
configs = dict(
    search=search,
    ppo_paper_mujoco=ppo_paper_mujoco,
    ppo_paper_roboschool=ppo_paper_roboschool,
)
