import torch as tr
import torch.nn as nn



class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()