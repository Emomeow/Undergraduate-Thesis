"""
本文件用于将Environment和Agent两个模块串联起来并且训练Agent
"""
import os
import json
import argparse
import torch
import pickle
import random
import numpy as np
import time
from agent.agent import MDPAgent
from rl_enviroment.env_simple import DeterministicMDPSimple
from visialization.plot_loss import plot_loss
from visialization.plot_policies import plot_policy_in_env
from visialization.plot_value_function import plot_current_value_function
from visialization.plot_value_function import plot_state_action_value_function
from visialization.plot_accumulate_reward import plt_accumulate_reward
from visialization.plot_value_change import plot_value_change
from visialization.plot_value_change import plot_var
from visialization.plot_value_change import generate_dist
from visialization.plot_value_change import get_value_mean_err

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# setting use of the GPU or CPU
USE_CUDA = torch.cuda.is_available()
# if the GPU is available for the server, the device is GPU, otherwise, the device is CPU
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main_func(action_size, state_representation_size, horizon_length, gamma, optimizer_type,
              init_learning_rate, training_num, epoch_num, gradient_type,
              lr_discount_factor, lr_discount_epoch, random_seed, env_name, init_value_list, varepsilon):
    # 控制模型的初始化
    setup_seed(random_seed)

    # 根据模型的名称选择超参数
    if env_name == "env_simple":
        env_class = DeterministicMDPSimple
    else:
        raise Exception("")

    # 从给定文件夹内读取已有的MDP环境
    deterministic_env = env_class(state_representation_size=state_representation_size, max_step=horizon_length,
                                  state_representation_random_seed=random_seed, varepsilon=varepsilon)

    training_data, non_repeat_sample_data_length = deterministic_env.get_state_action_next_state_tuple()
    training_data_dict_list = []
    reward_list = []
    next_state_list = []
    action_list = []
    current_state_list = []
    termination_set_none = False
    for data in training_data:
        current_state_list.append(data[0])
        action_list.append(data[1])
        next_state_list.append(data[2])
        reward_list.append(data[3])
        if termination_set_none is False:
            training_data_dict_list.append({"current_state": data[0], "next_state": data[2],
                                            "current_action": data[1], "reward": data[3]})
        else:
            if data[4] is False:
                training_data_dict_list.append({"current_state": data[0], "next_state": data[2],
                                            "current_action": data[1], "reward": data[3]})
            else:
                training_data_dict_list.append({"current_state": data[0], "next_state": None,
                                            "current_action": data[1], "reward": data[3]})

    # state_index_dict = deterministic_env.get_state_next_state_index_dict()  # 用于展示函数拟合Reward Function
    # reward_function = deterministic_env.get_reward_function()  # 获取每个点的reward

    loss_list = []
    policy_list = []
    state_value_function_list = []
    total_accumulated_reward_list = []
    target_policy_list = []
    target_value_function_list = []
    total_Q_value_list = []
    #调整batch大小
    buffer_size = len(training_data)
    batch_size = len(training_data)

    mdp_agent = MDPAgent(buffer_size=buffer_size, state_representation_size=state_representation_size,
                         action_size=action_size, optimizer_type=optimizer_type,
                         init_learning_rate=init_learning_rate, gradient_type=gradient_type,
                         lr_discount_factor=lr_discount_factor, lr_discount_epoch=lr_discount_epoch,
                         training_num=training_num)
    mdp_agent.replay_buffer.insert_data_tuple_list(training_data_dict_list)

    # 根据给定的state value来计算线性参数初始值应该是多少
    '''
    solved_theta = np.linalg.solve(deterministic_env.state_representation, init_value_list)
    theta = [solved_theta.tolist(), solved_theta.tolist()]
    mdp_agent.model.get_initialization(theta1=theta)
    '''
    #自定义参数初始化
    
    theta1 = np.random.normal(loc = 0, scale = 0.1, size=(100,3))
    theta2 = np.random.normal(loc = 0, scale = 0.1, size=(2,100))
    mdp_agent.model.get_initialization(theta1 = theta1, theta2 = theta2)
    
    

    begin_time = time.time()
    for current_epoch in range(epoch_num):

        current_state_value, policy, state_value_function, Q_value_list = mdp_agent.get_current_policy_and_value(
                current_state_data=current_state_list,
                action_data=action_list,
                state_representation=deterministic_env.state_representation)
        policy_list.append(policy)
        #target_policy_list.append(target_policy)
        state_value_function_list.append(state_value_function)
        total_Q_value_list.append(Q_value_list)
        #target_value_function_list.append(target_value_function)
        if current_epoch % 10 == 0:
            print("current mean value is " + str(current_state_value))
            print("current policy is " + str(policy))
            #print("target policy is "+ str(target_policy))
       
        accumulated_reward = deterministic_env.get_mean_reward_from_all_state_given_policy(policy_list=policy)
        total_accumulated_reward_list.append(accumulated_reward)

        temp_loss_list = mdp_agent.offline_learning(batch_size=batch_size, gamma=gamma, epoch_num=current_epoch)
        loss_list += temp_loss_list
        if current_epoch % 10 == 0:
            print("finish one epoch of training ********** " + str(temp_loss_list) + ", current epoch is " + str(current_epoch))
    #跳出模块
    '''
    mdp_agent = MDPAgent(buffer_size=buffer_size, state_representation_size=state_representation_size,
                         action_size=action_size, optimizer_type=optimizer_type,
                         init_learning_rate=init_learning_rate, gradient_type="true",
                         lr_discount_factor=lr_discount_factor, lr_discount_epoch=lr_discount_epoch,
                         training_num=training_num)
    mdp_agent.replay_buffer.insert_data_tuple_list(training_data_dict_list)

    for current_epoch in range(int(epoch_num/2), epoch_num):

        current_state_value, policy, state_value_function = mdp_agent.get_current_policy_and_value(
                current_state_data=current_state_list,
                action_data=action_list,
                state_representation=deterministic_env.state_representation)
        policy_list.append(policy)
        state_value_function_list.append(state_value_function)
        if current_epoch % 10 == 0:
            print("current mean value is " + str(current_state_value))
            print("current policy is " + str(policy))
       
        accumulated_reward = deterministic_env.get_mean_reward_from_all_state_given_policy(policy_list=policy)
        total_accumulated_reward_list.append(accumulated_reward)

        temp_loss_list = mdp_agent.offline_learning(batch_size=batch_size, gamma=gamma)
        loss_list += temp_loss_list
        if current_epoch % 10 == 0:
            print("finish one epoch of training ********** " + str(temp_loss_list) + ", current epoch is " + str(current_epoch))
    '''
    #线性拟合
    '''
    dist = generate_dist()
    means, err = get_value_mean_err(dist, state_value_function_list, len(loss_list) - 1)
    x = range(1,len(means))
    y = means[1:]
    A = np.stack((x,np.ones(len(x))),axis = 1)
    b = np.array(y).reshape((len(x),1))
    fitting_value_list = list()
    fitting_theta, _, _, _ = np.linalg.lstsq(A,b,rcond = None)
    a_ = fitting_theta[0]
    b_ = fitting_theta[1]
    fitting_value_list = list(map(lambda x: a_*x + b_, range(len(means))))
    value_list = np.zeros(state_representation_size)
    for i in range(1,len(dist)):
        for j in range(len(dist[i])):
            value_list[dist[i][j]] = fitting_value_list[i]
    value_list[dist[0][0]] = state_value_function_list[len(loss_list)-1][dist[0][0]]
    solved_theta = np.linalg.solve(deterministic_env.state_representation, value_list)
    fitting_theta = [solved_theta.tolist(), solved_theta.tolist(), solved_theta.tolist(), solved_theta.tolist()]
    mdp_agent.model.get_initialization(theta=fitting_theta)

    for current_epoch in range(int(epoch_num/2),epoch_num):

        current_state_value, policy, state_value_function = mdp_agent.get_current_policy_and_value(
                current_state_data=current_state_list,
                action_data=action_list,
                state_representation=deterministic_env.state_representation)
        policy_list.append(policy)
        state_value_function_list.append(state_value_function)
        if current_epoch % 10 == 0:
            print("current mean value is " + str(current_state_value))
            print("current policy is " + str(policy))
       
        accumulated_reward = deterministic_env.get_mean_reward_from_all_state_given_policy(policy_list=policy)
        total_accumulated_reward_list.append(accumulated_reward)

        temp_loss_list = mdp_agent.offline_learning(batch_size=batch_size, gamma=gamma)
        loss_list += temp_loss_list
        if current_epoch % 10 == 0:
            print("finish one epoch of training ********** " + str(temp_loss_list) + ", current epoch is " + str(current_epoch))
    '''
    end_time = time.time()
    print("running time is " + str(end_time - begin_time) + "s")
    # 将输出结果画图展示出来
    plt_accumulate_reward(reward_list=total_accumulated_reward_list, gradient_type=gradient_type)
    plot_loss(loss_list, gradient_type=gradient_type)
    plot_policy_in_env(env=deterministic_env, last_policy_list=policy_list[-1], gradient_type=gradient_type)
    plot_current_value_function(value_function_list=state_value_function_list, state_size=deterministic_env.state_size,
                                gradient_type=gradient_type)
    plot_state_action_value_function(state_action_value_function_list=total_Q_value_list,state_size=deterministic_env.state_size,action_size=deterministic_env.action_size,
                                gradient_type=gradient_type)
    #plot_var(value_function_list=state_value_function_list, gradient_type=gradient_type)
    
    #plot_value_change(loss_list=loss_list, value_function_list=state_value_function_list, state_size=deterministic_env.state_size, gradient_type=gradient_type)
    
    
    # 将输出的数据存入pickle中
    pickle.dump({"loss_list": loss_list, "policy_list": policy_list},
                open(CURRENT_PATH + "/../" + gradient_type + "_output_result.pl", "wb"))

    deterministic_env.close()

    return buffer_size, batch_size


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--env-name", type=str, default='env_simple', help="环境名称")
    PARSER.add_argument("--action-size", type=int, default=2, help="MDP模型中Action Space的大小")
    PARSER.add_argument("--state-representation-size", default=3, type=int, help="表示MDP模型每个状态的Dense Vector的维度")
    PARSER.add_argument("--training-num", type=int, default=1, help="每次采样完成后的训练次数")
    PARSER.add_argument("--epoch-num", type=int, default=5000, help="采样并训练的轮数")
    PARSER.add_argument("--horizon_size", type=int, default=20, help="用于指示当前一次探索应该采样多少次")
    PARSER.add_argument("--optimizer-type", type=str, default='sgd', help="规定优化器类型，现在支持“adam”和“SGD”")
    PARSER.add_argument("--init-learning-rate", type=float, default=0.01, help="规定初始学习率")
    PARSER.add_argument("--gamma", type=float, default=0.9, help="奖励折扣因子")
    PARSER.add_argument("--lr-discount-epoch", type=float, default=3000, help="学习率下降步数")
    PARSER.add_argument("--lr-discount-factor", type=float, default=1, help="学习率下降折扣数")
    PARSER.add_argument("--random-seed", type=int, default=110, help="随机种子")
    PARSER.add_argument("--init-value-list", type=json.loads, default='[1,1,1]', help="线性拟合初始化参数")
    PARSER.add_argument("--varepsilon", type=float, default=0.05, help="feature表示的小误差")

    ARGS = PARSER.parse_args()

    # 获得超参
    ENV_NAME = ARGS.env_name
    ACTION_SIZE = ARGS.action_size  # MDP模型中Action Space的大小
    STATE_REPRESENTATION_SIZE = ARGS.state_representation_size  # 表示MDP模型每个状态的Dense Vector的维度
    TRAINING_NUM = ARGS.training_num # 每次采样完成后的训练次数
    EPOCH_NUM = ARGS.epoch_num  # 采样并训练的轮数

    # 用于指示当前一次探索应该采样多少次
    HORIZON_LENGTH = ARGS.horizon_size

    OPTIMIZER_TYPE = ARGS.optimizer_type  # 规定优化器类型，现在支持“adam”和“sgd”

    INIT_LEARNING_RATE = ARGS.init_learning_rate # 规定学习率
    INIT_VALUE_LIST = ARGS.init_value_list  # 规定
    VAREPSILON = ARGS.varepsilon  # 规定初始化标准差

    GAMMA = ARGS.gamma  # 奖励折扣因子
    RANDOM_SEED = ARGS.random_seed  # 随机种子

    LR_DISCOUNT_EPOCH = ARGS.lr_discount_epoch  # 学习率下降步长
    LR_DISCOUNT_FACTOR = ARGS.lr_discount_factor  # 学习率下降比例

    BUFFER_SIZE = None
    BATCH_SIZE = None

    for GRADIENT_TYPE in ["semi"]:
        BUFFER_SIZE, BATCH_SIZE = main_func(action_size=ACTION_SIZE, state_representation_size=STATE_REPRESENTATION_SIZE, epoch_num=EPOCH_NUM,
                  init_learning_rate=INIT_LEARNING_RATE, gamma=GAMMA, optimizer_type=OPTIMIZER_TYPE,
                  horizon_length=HORIZON_LENGTH, training_num=TRAINING_NUM,
                  gradient_type=GRADIENT_TYPE, lr_discount_epoch=LR_DISCOUNT_EPOCH, lr_discount_factor=LR_DISCOUNT_FACTOR,
                  random_seed=RANDOM_SEED, env_name=ENV_NAME, varepsilon=VAREPSILON, init_value_list=INIT_VALUE_LIST)

    # 将传入的超参写入txt中
    HYPER_PARAM_DICT = {
        "action space size": ACTION_SIZE,
        "state representation size": STATE_REPRESENTATION_SIZE,
        "horizon length": HORIZON_LENGTH,
        "gamma": GAMMA,
        "optimizer type": OPTIMIZER_TYPE,
        "initial learning rate": INIT_LEARNING_RATE,
        "training number in each epoch": TRAINING_NUM,
        "exploration number": EPOCH_NUM,
        "learning rate discount epoch": LR_DISCOUNT_EPOCH,
        "learning rate discount factor": LR_DISCOUNT_FACTOR,
        "random seed": RANDOM_SEED,
        "buffer size": BUFFER_SIZE,
        "batch size": BATCH_SIZE,
        "varepsilon": VAREPSILON,
        "init_value_list": INIT_VALUE_LIST
    }

    with open(CURRENT_PATH + "/../hyper_param.json", "w") as f:
        f.write(str(HYPER_PARAM_DICT))
