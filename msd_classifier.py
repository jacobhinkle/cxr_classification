import torch
from torch import nn

from itertools import cycle, islice

import msd

class MSDClassifier2d(nn.Module):
    def __init__(self, in_channels, out_channels, width=1, maxdil=10, depth=100):
        super().__init__()
        dilations = list(islice(cycle(range(1, 1 + maxdil)), depth))
        self.features = msd.MSDBlock2d(
            in_channels=in_channels,
            dilations=dilations,
            width=width,
            padding_mode='zeros',
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        num_features = width * depth + in_channels
        self.classifier = nn.Linear(num_features, out_channels)

    def forward(self, x):
        f = self.features(x)
        p = self.pool(f)
        return self.classifier(p.view(-1, p.shape[1]))
