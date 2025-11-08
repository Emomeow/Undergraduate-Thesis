import os
import numpy as np
import torch
import torch.nn as nn
from agent.nn_model import FCNN
from torch.autograd import Variable

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def loss_with_semi_gradient(fcnn_model, optimizer, input_data, device, gamma):
    """
    Compute loss for a batch using the semi-gradient TD method.

    :param input_data: list of tuples (s_t, a_t, s_t+1, r_t)
    :param fcnn_model: model on device
    :param optimizer: optimizer (unused here but kept for API consistency)
    :param device: torch device
    :param gamma: discount factor
    :return: loss value (float) or None
    """
    # Zero gradients from previous step
    optimizer.zero_grad()

    # Collect all s_t data and move to device
    current_state_list = []
    for data_tuple in input_data:
        # Ensure state is numpy array
        if isinstance(data_tuple[0], (list, tuple)):
            state = np.array(data_tuple[0], dtype=np.float32)
        else:
            state = data_tuple[0]
        current_state_list.append(state)
    # Stack into single numpy array before converting to tensor
    current_state_array = np.stack(current_state_list)
    current_state_tensor = torch.from_numpy(current_state_array).to(device)

    # Collect all s_t+1 data and select non-final next states
    non_final_next_state_list = []
    next_state_list = []
    for data_tuple in input_data:
        next_state = data_tuple[2]
        next_state_list.append(next_state)
        if next_state is not None:
            # Ensure next_state is numpy array
            if isinstance(next_state, (list, tuple)):
                next_state = np.array(next_state, dtype=np.float32)
            non_final_next_state_list.append(next_state)

    if len(non_final_next_state_list) == 0:  # If no non-final states, nothing to train on
        return

    # Stack into single numpy array before converting to tensor
    non_final_next_state_array = np.stack(non_final_next_state_list)
    non_final_next_state_tensor = torch.from_numpy(non_final_next_state_array).to(device)

    # Collect reward data
    reward_list = []
    for data_tuple in input_data:
        reward_list.append(data_tuple[3])
    reward_tensor = Variable(torch.tensor(reward_list, device=device, dtype=torch.float32))

    # Collect action data
    action_list = []
    for data_tuple in input_data:
        action_list.append(data_tuple[1])
    action_tensor = Variable(torch.tensor(action_list, device=device, dtype=torch.int64)).view(-1, 1)

    # Compute Q(s_t+1, a) for non-final states; assume Q=0 for final states
    
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
    )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
    next_state_action_value = torch.ones(len(input_data), device=device)
    # Debug/monitoring variable (uncomment if needed for analysis)
    # next_s_a_value = torch.ones(len(input_data), device=device)
    # 获取maximum state action value，并将为Non-final的state value赋值
    next_state_action_value[non_final_mask] = fcnn_model(non_final_next_state_tensor).detach().max(1)[0]
    # Compute Q(s_t, a_t) and select values for chosen actions
    current_state_action_value_raw = fcnn_model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)
    # Debug/monitoring variable (uncomment if needed for analysis)
    # current_state_value = current_state_action_value_raw.max(1)[0]
    '''
    dist = [[5,7],[2,4,6],[1,3]]
    k = Variable(torch.tensor(0, device=device, dtype=torch.float32))
    for i in range(len(dist)):
        b = torch.index_select(current_state_value, dim=0, index=torch.tensor(dist[i]).to(device))
        k.add_(torch.var(b))
    '''
    # Use MSE between current Q and (reward + gamma * next_Q) as the loss
    criterion = nn.MSELoss()
    loss = criterion(current_state_action_value, (reward_tensor + gamma * next_state_action_value).view(-1, 1)) # 以current state value作为学习对象  # 计算gradient
    # 对当前的模型进行优化

    # 返回当前函数的loss
    return loss.data.to("cpu").numpy().tolist()


def loss_with_true_gradient(fcnn_model, optimizer, input_data, device, gamma):
    """
    Train a model on one batch using the true gradient of the Bellman error.

    :param input_data: list of tuples (s_t, a_t, s_t+1, r_t). Example shapes:
        s_t: (feature_dim,), a_t: scalar, s_t+1: (feature_dim,), r_t: scalar
    :param fcnn_model: model to train (already moved to `device`).
    :param optimizer: optimizer for the model (may contain global state like momentum).
    :param device: torch device (cpu or cuda).
    :param gamma: discount factor.
    :return: loss value (float) or None
    """
    # Zero gradients from the previous step
    optimizer.zero_grad()

    # Duplicate the model for later computation (used as a frozen reference)
    duplicated_model = FCNN(input_size=fcnn_model.input_size, output_size=fcnn_model.output_size)
    duplicated_model.load_state_dict(fcnn_model.state_dict())
    duplicated_model.to(device)

    # 获取所有的s_t数据，处理后放到device上
    current_state_list = []
    for data_tuple in input_data:
        current_state_list.append(data_tuple[0])
    current_state_tensor = Variable(torch.tensor(current_state_list, device=device, dtype=torch.float32))

    # Collect all s_t+1 data and select non-final next states
    non_final_next_state_list = []
    next_state_list = []
    for data_tuple in input_data:
        next_state_list.append(data_tuple[2])
        if data_tuple[2] is not None:
            non_final_next_state_list.append(data_tuple[2])

    if len(non_final_next_state_list) == 0:  # 将没有数据用于训练时，直接退出
        return

    non_final_next_state_tensor = Variable(torch.tensor(
        non_final_next_state_list, device=device, dtype=torch.float32))

    # Collect reward data
    reward_list = []
    for data_tuple in input_data:
        reward_list.append(data_tuple[3])
    reward_tensor = Variable(torch.tensor(reward_list, device=device, dtype=torch.float32))

    # Collect action data
    action_list = []
    for data_tuple in input_data:
        action_list.append(data_tuple[1])
    action_tensor = Variable(torch.tensor(action_list, device=device, dtype=torch.int64)).view(-1, 1)

    # Use the original model to compute Q(s_t, a_t) and select values for chosen actions
    current_state_action_value_raw = fcnn_model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)
    current_state_value = current_state_action_value_raw.max(1)[0]
    dist = [[5,7],[2,4,6],[1,3]]
    k = Variable(torch.tensor(0, device=device, dtype=torch.float32))
    for i in range(len(dist)):
        b = torch.index_select(current_state_value, dim=0, index=torch.tensor(dist[i]).to(device))
        k.add_(torch.var(b))
    # Use the duplicated model to compute Q(s_t+1, a) for non-final next states.
    # For final next states, Q is assumed to be 0.
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
    )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
    next_state_action_value = torch.zeros(len(input_data), device=device)
    next_state_action_value[non_final_mask] = duplicated_model(
        non_final_next_state_tensor).max(1)[0]  # get max state-action value for non-final states

    # Set up two loss functions using detached targets to compute a true Bellman gradient:
    # loss1: [f(x1) - (r + gamma * v(x2))]^2 where v(x2) is detached
    # loss2: [gamma * f(x2) - (-r + v(x1))]^2 where v(x1) is detached
    criterion1 = nn.MSELoss()
    target1 = reward_tensor + gamma * next_state_action_value.detach()
    loss1 = criterion1(current_state_action_value.squeeze(-1),
                       target1) # 以current state value作为学习对象 
    # 计算gradient

    # loss computed for the duplicated-model side
    criterion2 = nn.MSELoss()
    target2 = - reward_tensor + current_state_action_value.detach().squeeze(-1)
    loss2 = criterion2(gamma * next_state_action_value,
                       target2)  # 以next state value function作为学习对象
    # 计算gradient

    # Theoretical note: one could combine gradients from both models to get the true TD gradient.
    # (The loop below is intentionally commented out; adjust if you want to directly mix grads.)

    # 对当前的模型进行优化

    # 返回当前函数的loss
    return loss1.data.to("cpu").numpy().tolist()
