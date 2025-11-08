"""
This module defines training routines for neural networks, including
training with semi-gradient and true-gradient methods.
"""
import os
import torch
import torch.nn as nn
from agent.nn_model import FCNN
from torch.autograd import Variable
import numpy as np  # noqa: F401 - Used in commented out code for tensor conversions

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def training_fcnn_model_with_semi_gradient(*, fcnn_model, optimizer, input_data, device, gamma):
    """
    Train the FCNN model on a single batch using the semi-gradient method.

    :param input_data: list of tuples (s_t, a_t, s_t+1, r_t)
        where s_t/s_t+1 are state representations and a_t, r_t are action and reward
    :param fcnn_model: model to train (on device)
    :param optimizer: optimizer for the model
    :param device: torch device (cpu or cuda)
    :param gamma: discount factor
    :return: None or loss
    """

    # Zero gradients from previous step before computing new ones
    optimizer.zero_grad()

    # Collect and process all s_t data
    current_state_list = []
    for data_tuple in input_data:
        if isinstance(data_tuple[0], (list, tuple)):
            state = np.array(data_tuple[0], dtype=np.float32)
        elif isinstance(data_tuple[0], np.ndarray):
            state = data_tuple[0].astype(np.float32)
        else:
            print(f"Unexpected state type: {type(data_tuple[0])}")
            state = np.array(data_tuple[0], dtype=np.float32)
        current_state_list.append(state)
    
    # Convert list to numpy array then to tensor
    current_state_array = np.stack(current_state_list)
    current_state_tensor = torch.from_numpy(current_state_array).to(device)

    # Collect and process all s_t+1 values and select those that are not final states
    non_final_next_state_list = []
    next_state_list = []
    for data_tuple in input_data:
        next_state = data_tuple[2]
        next_state_list.append(next_state)
        if next_state is not None:
            # Ensure next_state is numpy array
            if isinstance(next_state, (list, tuple)):
                next_state = np.array(next_state, dtype=np.float32)
            else:
                next_state = next_state.astype(np.float32) if isinstance(next_state, np.ndarray) else next_state
            non_final_next_state_list.append(next_state)

    if len(non_final_next_state_list) == 0:  # if there are no next states to train on, exit
        return

    # Convert list to numpy array then to tensor
    non_final_next_state_array = np.stack(non_final_next_state_list)
    non_final_next_state_tensor = torch.from_numpy(non_final_next_state_array).to(device)

    # Collect all rewards
    reward_list = []
    for data_tuple in input_data:
        reward_list.append(data_tuple[3])
    reward_tensor = Variable(torch.tensor(reward_list, device=device, dtype=torch.float32))

    # Collect all actions
    action_list = []
    for data_tuple in input_data:
        action_list.append(data_tuple[1])
    action_tensor = Variable(torch.tensor(action_list, device=device, dtype=torch.int64)).view(-1, 1)

    # Use the model to compute Q(s_t+1, a) and assume Q=0 for final states
    
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
    )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
    next_state_action_value = torch.zeros(len(input_data), device=device)
    # 获取maximum state action value，并将为Non-final的state value赋值
    next_state_action_value[non_final_mask] = fcnn_model(non_final_next_state_tensor).detach().max(1)[0]

    # Use the model to compute Q(s_t, a_t) and select values corresponding to chosen actions
    current_state_action_value_raw = fcnn_model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)
    
    current_state_value = current_state_action_value_raw.max(1)[0]
    '''
    dist = [[5,7],[2,4,6],[1,3]]
    k = Variable(torch.tensor(0, device=device, dtype=torch.float32))
    for i in range(len(dist)):
        b = torch.index_select(current_state_value, dim=0, index=torch.tensor(dist[i]).to(device))
        k.add_(torch.var(b))
    '''

    # Compute loss with MSE between predicted Q and target (reward + gamma * next_Q)
    criterion = nn.MSELoss()
    # 以current state value作为学习对象
    loss = criterion(current_state_action_value, (reward_tensor + gamma * next_state_action_value).view(-1, 1))
    loss.backward()  # 计算gradient

    # 对当前的模型进行优化
    optimizer.step()

    # 返回当前函数的loss
    #return loss.data.to("cpu").numpy().tolist()


def training_fcnn_model_with_true_gradient(fcnn_model, optimizer, input_data, device, gamma):
    """
    本函数用于针对一个模型，一次训练数据，进行一次训练。本函数使用Bellman Error的True Gradient进行训练

    :param input_data: 是一个List of Tuple，每个Tuple里面包含的内容有（s_t, a_t, s_t+1, r_t），其维度分别为为
        s_t: (160, 160, 3) a_t : (3, 1) s_t+1: (160, 160, 3) r_t: (1, )
    :param fcnn_model: 需要训练的模型，已经放到Device上了。这里device指CPU或者GPU。
    :param optimizer: 当前模型的优化器，需要是全局性的，因为里面可能会包含类似Momentum之类的全局信息
    :param device:
    :param gamma: Discount Factor，期望奖励的折扣因子
    :return:
    """
    # 将前一次Tensor中的grad清零，以便于后续的运算
    optimizer.zero_grad()

    # 将model复制一份,用于之后的计算
    duplicated_model = FCNN(input_size=fcnn_model.input_size, output_size=fcnn_model.output_size)
    duplicated_model.load_state_dict(fcnn_model.state_dict())
    duplicated_model.to(device)

    # 获取所有的s_t数据，处理后放到device上
    current_state_list = []
    for data_tuple in input_data:
        current_state_list.append(data_tuple[0])
    current_state_tensor = Variable(torch.tensor(current_state_list, device=device, dtype=torch.float32))

    # 获取所有的s_t+1数据，并挑选出那些不为final state的状态作为函数的输入
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

    # 获取所有的reward数据
    reward_list = []
    for data_tuple in input_data:
        reward_list.append(data_tuple[3])
    reward_tensor = Variable(torch.tensor(reward_list, device=device, dtype=torch.float32))

    # 获取所有的action数据
    action_list = []
    for data_tuple in input_data:
        action_list.append(data_tuple[1])
    action_tensor = Variable(torch.tensor(action_list, device=device, dtype=torch.int64)).view(-1, 1)

    # 使用原始的model计算所有的Q(s_t, a_t)，并且按照选择的action，将对应的action value选出来
    current_state_action_value_raw = fcnn_model(current_state_tensor)
    current_state_action_value = current_state_action_value_raw.gather(1, action_tensor)
    
    current_state_value = current_state_action_value_raw.max(1)[0]
    '''
    dist = [[5,7],[2,4,6],[1,3]]
    k = Variable(torch.tensor(0, device=device, dtype=torch.float32))
    for i in range(len(dist)):
        b = torch.index_select(current_state_value, dim=0, index=torch.tensor(dist[i]).to(device))
        k.add_(torch.var(b))
    '''
    # 使用复制的model计算所有的Q(s_t+1, a_t+1)，并且假设在s_t+1为最终的state的时候，Q(s_t+1, a_t+1)=0
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.bool
    )  # 本函数的意义为筛选出哪些tuple的next state是final state，因为根据假设final state的值为0，最终的值为[True, False,....]
    next_state_action_value = torch.zeros(len(input_data), device=device)
    next_state_action_value[non_final_mask] = duplicated_model(
        non_final_next_state_tensor).max(1)[0]  # 获取maximum state action value，并将为Non-final的state value赋值

    # 给两个model分别设置一个Loss Function，将Model1的输出detach，当做Model2的label，计算Loss和Gradient，
    # 将Model2的输出detach，当做Model1的label，计算Loss和True Gradient
    # 这里计算 [f(x1) - (r + a * v(x2))]^2的loss和gradient
    criterion1 = nn.MSELoss()
    target1 = reward_tensor + gamma * next_state_action_value.detach()
    loss1 = criterion1(current_state_action_value.squeeze(-1),
                       target1) # 以current state value作为学习对象
    loss1.backward()  # 计算gradient

    # 这里计算[a * f(x2) - ( -r + v(x1) )]^2的loss和gradient
    criterion2 = nn.MSELoss()
    target2 = - reward_tensor + current_state_action_value.detach().squeeze(-1)
    loss2 = criterion2(gamma * next_state_action_value,
                       target2)  # 以next state value function作为学习对象
    loss2.backward()  # 计算gradient

    # 对原来Model中的每一个Parameter（Tensor），进行Gradient的修改，获得Temporal Difference Error的真正的gradient
    # 表示为 [f(x1) - r - a * v(x2)]grad(f(x1)) + [a * f(x2) - ( -r + v(x1) )]grad(f(x2))
    # = [v(x1) - r - a * v(x2)]*[grad(f(x1)) - grad(f(x2))]
    for param1, param2 in zip(fcnn_model.parameters(), duplicated_model.parameters()):
        param1.grad += param2.grad

    # 对当前的模型进行优化
    optimizer.step()

    # 返回当前函数的loss
    #return loss1.data.to("cpu").numpy().tolist()


def save_model(model, optimizer, lr_scheduler, filename):
    torch.save({"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler},
               CURRENT_PATH + "/../model/" + filename + ".pl")


def load_model(filename):
    if not os.path.exists(CURRENT_PATH + "/../model/" + filename + ".pl"):
        print("can't load model")
        return None, None, None

    from torch.serialization import safe_globals
    from agent.nn_model import FCNN
    
    # Safely allow FCNN class for deserialization
    with safe_globals([FCNN]):
        state_dict = torch.load(CURRENT_PATH + "/../model/" + filename + ".pl", weights_only=False)
        return state_dict["model"], state_dict["optimizer"], state_dict["lr_scheduler"]
