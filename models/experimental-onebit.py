import math

import numpy as np
import torch
import torch.nn as nn

def clip(x, a, b):
    return torch.max(torch.min(x, b), a)

class BitWeights(nn.Module):
    def __init__(self, layer):
        super(BitWeights, self).__init__()
        self.layer = layer

    def forward(self, input):
        avg_weight = torch.mean(self.layer.weight.data)
        print(f"Average weight: {avg_weight.item()}")
        
        self.layer.weight.data = torch.sign(self.layer.weight.data - avg_weight)
        return self.layer(input)
    
class QuantActivation(nn.Module):
    def __init__(self, bits=8, eps=1e-3):
        super(QuantActivation, self).__init__()
        self.bits = bits
        self.max_val = (2 ** (bits - 1)) - 1

    def forward(self, input):
        gamma = torch.max(torch.abs(input))
        return clip(input * self.max_val / gamma, -self.max_val, self.max_val)