"""
    This file contains a neural network module for us to
    define our actor and critic networks in PPO.
"""

import torch
from torch import nn
from torchvision import models

class PointNavNet(nn.Module):
    def __init__(self, num_channels, num_outputs, batch_size) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder(num_channels, 512-32-32)
        self.pointgoal_encoder = PointgoalEncoder()
        self.last_action_encoder = LastActionEncoder()
        self.rnn = RNN(512, 512, num_outputs, 1)
        self.hidden = self.rnn.init_hidden(batch_size)

    def forward(self, visual, pointgoal, last_action):
        visual_encoded = self.visual_encoder(visual)
        pointgoal_encoded = self.pointgoal_encoder(pointgoal)
        last_action_encoded = self.last_action_encoder(last_action)
        rnn_input = torch.cat((visual_encoded, pointgoal_encoded, last_action_encoded), dim=1).unsqueeze(dim=0)
        output = self.rnn(rnn_input, self.hidden)
        return output

class VisualEncoder(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super().__init__()
        self.cnn = models.resnet50()

        self.cnn.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_outputs)
        
    def forward(self, input):
        output = self.cnn(input)
        return output

class PointgoalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 32)

    def forward(self, input):
        input = torch.stack((input[:,0], torch.cos(input[:,1]), torch.sin(input[:,1])), dim=-1)
        output = self.encoder(input)
        return output


class LastActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 32)

    def forward(self, input):
        output = self.encoder(input)
        return output

class RNN(nn.Module):
    # https://blog.floydhub.com/gru-with-pytorch/
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size,
                            self.hidden_dim).zero_()
        return hidden.data

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.fc(self.relu(output[:,-1]))
        return output, hidden
