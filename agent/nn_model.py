"""
This file defines the neural network models used by the agent.
"""
import numpy as np
import torch
import torch.nn as nn

# Import global device configuration
from config import DEVICE


class FCNN(nn.Module):
    """
    Fully connected neural network (FCNN).
    """
    def __init__(self, input_size, output_size, middle_size=100):
        super(FCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, middle_size, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(middle_size, output_size, bias=False)

    def forward(self, x):
        """
        Forward pass of the network
        :param x: input tensor of shape (batch_size, input_size)
        :return: output tensor of shape (batch_size, output_size)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_initialization(self, theta1, theta2):
        """
        Initialize weights from given arrays.
        :param theta1: weight array for first layer (output_size, input_size)
        :param theta2: weight array for second layer (1, output_size)
        :return: None
        """
        # Ensure theta1 has correct shape (output_size, input_size)
        theta1 = np.array(theta1, dtype=np.float32)
        # Ensure theta2 has correct shape (1, output_size)
        theta2 = np.array(theta2, dtype=np.float32)
        # Transpose if needed to match PyTorch's weight layout
        self.fc1.weight.data = torch.tensor(data=theta1, dtype=torch.float32, device=DEVICE)
        self.fc2.weight.data = torch.tensor(data=theta2, dtype=torch.float32, device=DEVICE)
        
    def get_feature(self, data):
        """
        Extract features from input data
        :param data: input data
        :return: features
        """
        return data