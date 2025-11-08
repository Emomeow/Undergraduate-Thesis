"""
本文件用于定义神经网络模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


# setting use of the GPU or CPU
USE_CUDA = torch.cuda.is_available()
# if the GPU is available for the server, the device is GPU, otherwise, the device is CPU
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


class FCNN(nn.Module):
    """
    线性模型
    """
    def __init__(self, input_size, output_size, middle_size=100):
        super(FCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, middle_size, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(middle_size, output_size, bias=False)


    def get_initialization(self, theta1, theta2):
        """
        给定array的初始化
        :param theta:
        :return:
        """
        theta1 = np.array(theta1, dtype=np.float32)
        theta2 = np.array(theta2, dtype=np.float32)
        self.fc1.weight.data = torch.tensor(data=theta1, dtype=torch.float32, device=DEVICE)
        self.fc2.weight.data = torch.tensor(data=theta2, dtype=torch.float32, device=DEVICE)
        

    def get_feature(self, data):
        return data
    '''
    def get_action_vector_list(self, idx):
        return self.fc1.weight.detach().cpu().numpy()[idx, :]

    def get_param_std(self):
        return self.fc1.weight.std().detach().cpu().numpy().tolist()
    '''
    def forward(self, data):
        x = self.fc1(data)
        y = self.fc2(x)
        return y
