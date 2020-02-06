import gc
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ppo.control_flow.gridworld.abstract_recurrence as abstract_recurrence
import ppo.control_flow.recurrence as recurrence
from ppo.control_flow.lstm import LSTMCell
from ppo.distributions import FixedCategorical, Categorical
from ppo.utils import init_

RecurrentState = namedtuple(
    "RecurrentState",
    "a d u ag dg p v h hy1 cy1 hy2 cy2 a_probs d_probs ag_probs dg_probs gru_gate",
)


def gate(g, new, old):
    old = torch.zeros_like(new).scatter(1, old.unsqueeze(1), 1)
    return FixedCategorical(probs=g * new + (1 - g) * old)


class Recurrence(abstract_recurrence.Recurrence, recurrence.Recurrence):
    def __init__(
        self, hidden_size, conv_hidden_size, gate_coef, gru_gate_coef, **kwargs
    ):
        self.gru_gate_coef = gru_gate_coef
        self.gate_coef = gate_coef
        self.conv_hidden_size = conv_hidden_size
        recurrence.Recurrence.__init__(self, hidden_size=hidden_size, **kwargs)
        abstract_recurrence.Recurrence.__init__(
            self, conv_hidden_size=self.encoder_hidden_size
        )
        self.zeta = init_(
            nn.Linear(
                hidden_size + self.gru_hidden_size + self.encoder_hidden_size,
                hidden_size,
            )
        )
        gc.collect()
        self.zeta_u = init_(nn.Linear(2 * self.gru_hidden_size, hidden_size))
        self.gru1 = LSTMCell(self.encoder_hidden_size + self.ne, self.gru_hidden_size)
        self.gru2 = LSTMCell(self.encoder_hidden_size, self.gru_hidden_size)
        self.d_gate = Categorical(hidden_size, 2)
        self.a_gate = Categorical(hidden_size, 2)
        state_sizes = self.state_sizes._asdict()
        self.state_sizes = RecurrentState(
            **state_sizes,
            hy1=self.gru_hidden_size,
            cy1=self.gru_hidden_size,
            hy2=self.gru_hidden_size,
            cy2=self.gru_hidden_size,
            ag_probs=2,
            dg_probs=2,
            ag=1,
            dg=1,
            gru_gate=self.gru_hidden_size
        )

    @property
    def gru_in_size(self):
        return self.encoder_hidden_size

    def pack(self, hxs):
        def pack():
            for name, size, hx in zip(
                RecurrentState._fields, self.state_sizes, zip(*hxs)
            ):
                x = torch.stack(hx).float()
                assert np.prod(x.shape[2:]) == size
                yield x.view(*x.shape[:2], -1)

        hx = torch.cat(list(pack()), dim=-1)
        return hx, hx[-1:]

    def parse_hidden(self, hx: torch.Tensor) -> RecurrentState:
        return RecurrentState(*torch.split(hx, self.state_sizes, dim=-1))

    def inner_loop(self, inputs, rnn_hxs):
        T, N, dim = inputs.shape
        inputs, actions = torch.split(
            inputs.detach(), [dim - self.action_size, self.action_size], dim=2
        )

        # parse non-action inputs
        inputs = self.parse_inputs(inputs)
        inputs = inputs._replace(obs=inputs.obs.view(T, N, *self.obs_spaces.obs.shape))

        # build memory
        nl = len(self.obs_spaces.lines.nvec)
        M = self.build_memory(N, T, inputs)

        P = self.build_P(M, N, rnn_hxs.device, nl)
        half = P.size(2) // 2 if self.no_scan else nl
        new_episode = torch.all(rnn_hxs == 0, dim=-1).squeeze(0)
        hx = self.parse_hidden(rnn_hxs)
        for _x in hx:
            _x.squeeze_(0)

        h = hx.h
        hy1 = hx.hy1
        cy1 = hx.cy1
        hy2 = hx.hy2
        cy2 = hx.cy2
        p = hx.p.long().squeeze(-1)
        u = hx.u
        hx.a[new_episode] = self.n_a - 1
        ag_probs = hx.ag_probs
        ag_probs[new_episode, 1] = 1
        R = torch.arange(N, device=rnn_hxs.device)
        ones = self.ones.expand_as(R)
        A = torch.cat([actions[:, :, 0], hx.a.view(1, N)], dim=0).long()
        D = torch.cat([actions[:, :, 1], hx.d.view(1, N)], dim=0).long()
        AG = torch.cat([actions[:, :, 2], hx.ag.view(1, N)], dim=0).long()
        DG = torch.cat([actions[:, :, 3], hx.dg.view(1, N)], dim=0).long()

        for t in range(T):
            self.print("p", p)
            obs = self.preprocess_obs(inputs.obs[t])
            h = self.gru(obs, h)
            zeta_inputs = [h, M[R, p], self.embed_action(A[t - 1].clone())]
            z = F.relu(self.zeta(torch.cat(zeta_inputs, dim=-1)))
            # then put M back in gru
            # then put A back in gru
            d_gate = self.d_gate(z)
            self.sample_new(DG[t], d_gate)
            a_gate = self.a_gate(z)
            self.sample_new(AG[t], a_gate)

            x = [M[R, p], u]
            (hy1_, cy1_), gru_gate = self.gru1(torch.cat(x, dim=-1), (hy1, cy1))
            (hy2_, cy2_), gru_gate = self.gru2(M[R, p] * obs, (hy2, cy2))
            z = F.relu(self.zeta_u(torch.cat([hy1_, hy2_], dim=-1)))
            u = self.upsilon(z).softmax(dim=-1)
            self.print("u", u)
            w = P[p, R]
            d_probs = (w @ u.unsqueeze(-1)).squeeze(-1)
            dg = DG[t].unsqueeze(-1).float()
            self.print("dg prob", d_gate.probs[:, 1])
            self.print("dg", dg)
            d_dist = gate(dg, d_probs, ones * half)
            self.print("d_probs", d_probs[:, half:])
            self.sample_new(D[t], d_dist)
            p = p + D[t].clone() - half
            p = torch.clamp(p, min=0, max=M.size(1) - 1)

            ag = AG[t].unsqueeze(-1).float()
            a_dist = gate(ag, self.actor(z).probs, A[t - 1])
            self.sample_new(A[t], a_dist)
            self.print("ag prob", a_gate.probs[:, 1])
            self.print("ag", ag)
            hy1 = dg * hy1_ + (1 - dg) * hy1
            cy1 = dg * cy1_ + (1 - dg) * cy1
            hy2 = dg * hy2_ + (2 - dg) * hy2
            cy2 = dg * cy2_ + (2 - dg) * cy2
            yield RecurrentState(
                a=A[t],
                v=self.critic(z),
                h=h,
                u=u,
                hy1=hy1,
                cy1=cy1,
                hy2=hy2,
                cy2=cy2,
                p=p,
                a_probs=a_dist.probs,
                d=D[t],
                d_probs=d_dist.probs,
                ag_probs=a_gate.probs,
                dg_probs=d_gate.probs,
                ag=ag,
                dg=dg,
                gru_gate=gru_gate,
            )
