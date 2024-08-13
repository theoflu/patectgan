import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim, gen_dims, output_dim):
        super(Generator, self).__init__()
        torch.manual_seed(0)
        
        layers = self._build_layers(input_dim, gen_dims, output_dim)
        self.seq = nn.Sequential(*layers)
        
        self._initialize_weights()

    def _build_layers(self, input_dim, gen_dims, output_dim):
        dim = input_dim
        layers = []
        for item in gen_dims:
            layers.extend([
                nn.Linear(dim, item),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(item),
                nn.Dropout(0.3)
            ])
            dim = item
        layers.append(nn.Linear(dim, output_dim))
        layers.append(nn.Tanh())
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input):
        return self.seq(input)
