import torch
from torch import nn as nn

from networks import Agent, MLPBase


class TeacherAgent(Agent):
    def build_recurrent_module(
        self, hidden_size, obs_spaces, recurrent, **network_args
    ):
        d = obs_spaces.shape[0]

        class Base(MLPBase):
            def __init__(self):
                super().__init__(
                    hidden_size=hidden_size,
                    num_inputs=hidden_size + d - 1,
                    recurrent=True,
                    **network_args,
                )
                self.embedding = nn.Embedding(int(obs_spaces.high[0]), hidden_size)

            def forward(self, inputs, rnn_hxs, masks):
                choices = inputs[:, 0]
                rest = inputs[:, 1:]
                embedding = self.embedding(choices.flatten().long())
                inputs = torch.cat([embedding, rest], dim=-1)
                return super().forward(inputs, rnn_hxs, masks)

        return Base()
