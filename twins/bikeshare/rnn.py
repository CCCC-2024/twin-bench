"""RNNRates: the paper's learned demand forecast (VP-RNN) as a rate model of the station twin (see rates.py).

Gammelli et al. 2022 forecast each hour's rental (or return) rate with a recurrent network that reads the hour's
day of week, hour and precipitation probability, and the previous hour's count. RNNRates runs the authors' trained
networks for one station (one for rentals, one for returns), fed by the twin one hour at a time:
  update(obs)      one step with the hour's counts -> the next hour's rate: the paper's one-step forecast
  ahead(horizon)   the next hours from the current state; after the first, the network reads its own previous
                   forecast in place of the count it has not seen. At 23:00 this uses only what is known that
                   night. The paper's decisions instead used each hour's one-step forecast, which needs that
                   day's earlier counts (README, Notes).
precip(date, hour) is the weather the forecast assumes; by default the recorded one, i.e. a perfect weather forecast.

The network is copied from src/algos/vprnn.py @ abf77f7 (MIT, THIRD_PARTY_NOTICES.md): only the inference part
(VPRNN.q_model, VPRNN.q_h_0 and Encoder), whose output the authors use as the forecast (src/misc/utils.py,
get_results). Parameter names are kept, so their saved weights load as they are. Needs torch: pip install -e ".[rnn]".
"""
import datetime

import numpy as np
import torch
from torch import nn

from . import data
from .rates import next_hour


class _Encoder(nn.Module):                      # vprnn.Encoder
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lin_input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.lin_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.lin_hidden_to_loc = nn.Linear(hidden_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.lin_hidden_to_hidden(torch.relu(self.lin_input_to_hidden(x))))
        return self.lin_hidden_to_loc(h), nn.functional.softplus(self.lin_hidden_to_scale(h))


class _Network(nn.Module):                      # the inference part of vprnn.VPRNN(input_dim=32, output_dim=1)
    def __init__(self, path):
        super().__init__()
        self.q_model = nn.GRU(input_size=33, hidden_size=128, batch_first=True)
        self.q_h_0 = nn.Parameter(torch.zeros(128))
        self.encoder = _Encoder(128, 32, 1)
        saved = torch.load(path, map_location="cpu")
        self.load_state_dict({k: v for k, v in saved.items() if k.split(".")[0] in ("q_model", "q_h_0", "encoder")})

    def step(self, features, count, h):
        """Features of hour t and the count of hour t-1 -> mean and sd of hour t's rate, and the new state."""
        out, h = self.q_model(torch.cat((features, torch.tensor([count], dtype=torch.float32)))[None, None], h)
        loc, scale = self.encoder(out[0, 0])
        mean = torch.exp(loc + scale ** 2 / 2)                       # log-normal mean, as in get_results
        sd = torch.sqrt((torch.exp(scale ** 2) - 1) * torch.exp(2 * loc + scale ** 2))
        return float(mean), float(sd), h


class RNNRates:
    def __init__(self, station=426, precip=None):
        self.precip = data.precip() if precip is None else precip
        self.nets = {k: _Network(data.RNN_MODELS / f"so_st{station}_{k}") for k in ("pickup", "return")}
        self.h = {k: net.q_h_0.detach().view(1, 1, -1) for k, net in self.nets.items()}
        self.next = {}                                               # kind -> (mean, sd) of the next hour
        self.date, self.hour = None, None

    def _features(self, day, hour):
        """The paper's 32 inputs: day of week (one-hot, Monday first), hour (one-hot), precipitation probability."""
        x = torch.zeros(32)
        x[day.weekday()] = 1.0
        x[7 + hour] = 1.0
        x[31] = self.precip(day, hour)
        return x

    @torch.no_grad()
    def update(self, obs):
        self.date, self.hour = datetime.date.fromisoformat(obs["date"]), obs["hour"]
        x = self._features(*next_hour(self.date, self.hour))
        for k, count in (("pickup", obs["pickups"]), ("return", obs["returns"])):
            mean, sd, self.h[k] = self.nets[k].step(x, count, self.h[k])
            self.next[k] = (mean, sd)
        (mp, sp), (mr, sr) = self.next["pickup"], self.next["return"]
        return dict(next_rent_rate=mp, next_return_rate=mr, next_rent_sd=sp, next_return_sd=sr)

    @torch.no_grad()
    def ahead(self, horizon):
        out = {}
        for k, net in self.nets.items():
            (mean, _), h = self.next[k], self.h[k]
            day, hour = next_hour(self.date, self.hour)              # the hour `mean` forecasts
            rates = [mean]
            for _ in range(horizon - 1):
                day, hour = next_hour(day, hour)
                mean, _, h = net.step(self._features(day, hour), mean, h)   # its own forecast as the count
                rates.append(mean)
            out[k] = np.array(rates)
        return out["pickup"], out["return"]
