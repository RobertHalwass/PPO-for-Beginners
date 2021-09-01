"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import torch
from torch import nn
from torch.nn import functional as F
from resnet import resnet50

class PointNavNet(nn.Module):
    def __init__(self, num_channels, hidden_size, num_outputs) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder(num_channels, hidden_size)
        self.pointgoal_encoder = PointgoalEncoder()
        self.last_action_encoder = LastActionEncoder()
        self.rnn = RNN(hidden_size+32+32, hidden_size, 2, num_outputs)

    def forward(self, visual, pointgoal, last_action):
        visual_encoded = self.visual_encoder(visual)
        pointgoal_encoded = self.pointgoal_encoder(pointgoal)
        last_action_encoded = self.last_action_encoder(last_action)
        rnn_input = torch.cat((visual_encoded, pointgoal_encoded, last_action_encoded), dim=1).unsqueeze(1)
        output = self.rnn(rnn_input)
        return output

class VisualEncoder(nn.Module):
    def __init__(self, num_channels, base_planes: int = 32, num_groups: int = 32, spatial_size: int = 128):
        self.cnn = resnet50(num_channels, base_planes, num_groups)
        self.num_compression_channels = round(2048 / ((spatial_size * self.cnn.final_spatial_compress) ** 2))
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.cnn.final_channels,
                self.num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, self.num_compression_channels),
            nn.ReLU(True),
        )
    
    def forward(self, input):
        input = F.avg_pool2d(input, 2)
        input = self.cnn(input)
        input = self.compression(input)
        return input

class PointgoalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 32)

    def forward(self, input):
        input = torch.stack((input[:,0], torch.cos(-input[:,1]), torch.sin(-input[:,1])), dim=-1)
        output = self.encoder(input)
        return output

class LastActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 32)

    def forward(self, input):
        output = self.encoder(input)
        return output

class RNN(nn.Module):
    # https://blog.floydhub.com/gru-with-pytorch/
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)
    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        output, _ = self.gru(input, h0)
        output = output[:, -1, :]
        output = self.fc(output)
        return output
    