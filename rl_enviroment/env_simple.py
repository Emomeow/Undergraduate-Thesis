"""
本模块主要包含MDP(Stochastic MDP)和MiniGrid(Deterministic MDP)两个可用的环境，若没有自定义的部分可以留空
"""
import os
import gym
import copy
import pickle
import numpy as np
import mdptoolbox
from gym.envs.classic_control import rendering
from itertools import product

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def _transition_rule(state, action):
    """
    传入当前状态和动作，给出下一个状态
    :param state:
    :param action:
    :return:
    """
    if action == 0:
        if state in [0]:
            return state
            # return 4  # 去掉self recurrent的情况
        else:
            return state - 1
    elif action == 1:
        if state in [2]:
            return state
            # return 0  # 去掉self recurrent的情况
        else:
            return state + 1
    else:
        raise Exception("")
    '''
    elif action == 2:
        if state in [2,5,8]:
            return state
            # return 0  # 去掉self recurrent的情况
        else:
            return state + 1
    elif action == 3:
        if state in [6,7,8]:
            return state
            # return 0  # 去掉self recurrent的情况
        else:
            return state + 3
    '''
    


def _reward_rule(next_state, step_penalty, win_reward):
    #奖励规则，若到达terminal给win，若未到达则有惩罚
    if next_state in [1]:
        return win_reward
    else:
        return step_penalty


def _generate_one_hot_array(total_length, pos, value):
    #生成独热码，总长为状态数，在2处有3值
    one_hot_array = [0] * total_length
    one_hot_array[pos] = value
    return one_hot_array


def _generate_transition_tensor(state_size, action_size):
    # 根据移动规则，形成转移概率
    transition_tensor = []
    for action in range(action_size):
        transition_tensor.append([])
        for state in range(state_size):
            next_state = _transition_rule(state, action)
            transition_tensor[action].append(_generate_one_hot_array(state_size, next_state, 1))
    return transition_tensor


def _generate_reward_tensor(state_size, action_size, step_penalty, win_reward):
    #生成奖励张量，有2个(左右)
    reward_tensor = []
    for action in range(action_size):
        #先生成空list
        reward_tensor.append([])
        for state in range(state_size):
            #对每个状态得出向左或向右之后得到的奖励
            next_state = _transition_rule(state, action)
            next_reward = _reward_rule(next_state=next_state, step_penalty=step_penalty, win_reward=win_reward)
            reward_tensor[action].append(_generate_one_hot_array(state_size, next_state, next_reward))
    return reward_tensor


class DeterministicMDPSimple(gym.Env):

    # 我们在初始函数中定义一个 viewer ，即画板
    def __init__(self, state_representation_size, max_step, state_representation_random_seed, varepsilon):
        super(DeterministicMDPSimple, self).__init__()
        self.current_state = 0
        self.action_size = 2
        self.state_size = 3
        self.state_representation_size = state_representation_size
        self.viewer = None

        self.max_step = max_step
        self.step_count = 0

        # action的顺序为上下左右
        self.transition_matrix = _generate_transition_tensor(self.state_size, self.action_size)
        self.transition_matrix = np.array(self.transition_matrix)

        self.step_penalty = -0.05
        self.win_reward = -0.05
        self.reward_tensor = _generate_reward_tensor(self.state_size, self.action_size, step_penalty=self.step_penalty,
                                                     win_reward=self.win_reward)
        self.reward_tensor = np.array(self.reward_tensor)

        # 创建新的MDP模型
        if state_representation_size != 1 and state_representation_size != 3:
            # 控制状态的初始化
            np.random.seed(state_representation_random_seed)
            # 构造状态的dense表示和状态和表示向量之间的映射
            full_rank = False
            while not full_rank:
                self.state_representation = np.random.rand(self.state_size, self.state_representation_size)

                if np.linalg.matrix_rank(self.state_representation) == min(self.state_size,
                                                                           self.state_representation_size):
                    full_rank = True

            # 每一个行向量表示一个状态，让每一个状态的L2 norm都为1
            for i in range(self.state_representation.shape[0]):
                self.state_representation[i] /= np.linalg.norm(self.state_representation[i], ord=2)
        if state_representation_size == 3:
            
            self.state_representation = []
            #坐标
            '''
            for i in range(1,6):
                for j in range(1,6):
                    self.state_representation.append([0.1*j,0.1*i])
                    
            del self.state_representation[12]
            self.state_representation = np.array(self.state_representation)
            '''
            #one-hot
            for i in range(state_representation_size):
                k = varepsilon*np.ones(state_representation_size)
                k[i] = np.sqrt(1 - (state_representation_size - 1)*varepsilon**2)
                self.state_representation.append(k)
            self.state_representation = np.array(self.state_representation)
            self.state_representation = self.state_representation.reshape([3,3])
            

        else:
            self.state_representation = np.arange(start=0, stop=1, step=1 / self.state_size).reshape(self.state_size, 1)

        # 构造从状态的encoding到状态下标的映射
        self.state_encode_to_index_mapping = {}
        for i in range(self.state_size):
            self.state_encode_to_index_mapping[str(self.state_representation[i].tolist())] = i

    def reset(self, state=None):
        if state is None:
            self.current_state = 0
        else:
            self.current_state = state
        self.step_count = 0

    def get_index_by_obs(self, obs):
        return self.state_encode_to_index_mapping[str(list(obs))]

    def get_termination_state(self):
        #终止状态
        return [1]

    def get_positive_pretermination_state(self):
        return [1]

    def get_negative_pretermination_state(self):
        return []

    def close(self):
        self.viewer.close()
    #返回该状态对应的state representation
    def obs(self):
        return self.state_representation[self.current_state]

    def step(self, action):
    # 走一步之后的奖励与状态
        if int(action) >= self.action_size:
            raise Exception("")

        next_state = np.argmax(self.transition_matrix[action][self.current_state])

        reward = self.reward_tensor[action][self.current_state][next_state]

        # if next_state in [11, 12, 15] or self.step_count >= self.max_step:
        if next_state in self.get_termination_state() or self.step_count >= self.max_step:
            done = True
        else:
            done = False

        self.current_state = next_state
        self.step_count += 1

        return self.current_state, reward, done

    def render(self, mode='human', close=False, policy_list=None):
        """
        继承Env render函数
        :param mode:
        :param close:
        :return:
        """

        # 创建网格世界，一共包括23条直线，事先算好每条直线的起点和终点坐标，然后绘制这些直线，代码如下：
        screen_width = 1500
        screen_height = 800

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)  # 调用rendering中的画图函数，#创建600*600的窗口
            # 横线
            self.line1 = rendering.Line((100, 100), (600, 100))
            self.line2 = rendering.Line((100, 150), (600, 150))
            self.line3 = rendering.Line((100, 200), (600, 200))
            self.line4 = rendering.Line((100, 250), (600, 250))
            self.line5 = rendering.Line((100, 300), (600, 300))
            self.line6 = rendering.Line((100, 350), (600, 350))
            self.line7 = rendering.Line((100, 400), (600, 400))
            self.line8 = rendering.Line((100, 450), (600, 450))
            self.line9 = rendering.Line((100, 500), (600, 500))
            self.line10 = rendering.Line((100, 550), (600, 550))
            self.line11 = rendering.Line((100, 600), (600, 600))
            # 竖线
            self.line12 = rendering.Line((100, 100), (100, 600))
            self.line13 = rendering.Line((150, 100), (150, 600))
            self.line14 = rendering.Line((200, 100), (200, 600))
            self.line15 = rendering.Line((250, 100), (250, 600))
            self.line16 = rendering.Line((300, 100), (300, 600))
            self.line17 = rendering.Line((350, 100), (350, 600))
            self.line18 = rendering.Line((400, 100), (400, 600))
            self.line19 = rendering.Line((450, 100), (450, 600))
            self.line20 = rendering.Line((500, 100), (500, 600))
            self.line21 = rendering.Line((550, 100), (550, 600))
            self.line22 = rendering.Line((600, 100), (600, 600))

            # 创建目标点
            self.mountain = rendering.make_polygon(v=[(100, 100), (100, 150), (150, 150), (150, 100), (100, 100)], filled=True)
            self.circletrans = rendering.Transform(translation=(50, 0))
            self.mountain.add_attr(self.circletrans)
            self.mountain.set_color(0, 1, 1)

            # 创建机器人
            self.robot = rendering.make_circle(15)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(1, 0.8, 0)

            # 创建完之后，给11条直线设置颜色，并将这些创建的对象添加到几何中代码如下：
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0, 0, 0)
            self.line13.set_color(0, 0, 0)
            self.line14.set_color(0, 0, 0)
            self.line15.set_color(0, 0, 0)
            self.line16.set_color(0, 0, 0)
            self.line17.set_color(0, 0, 0)
            self.line18.set_color(0, 0, 0)
            self.line19.set_color(0, 0, 0)
            self.line20.set_color(0, 0, 0)
            self.line21.set_color(0, 0, 0)
            self.line22.set_color(0, 0, 0)

            # 添加组件到Viewer中
            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.line13)
            self.viewer.add_geom(self.line14)
            self.viewer.add_geom(self.line15)
            self.viewer.add_geom(self.line16)
            self.viewer.add_geom(self.line17)
            self.viewer.add_geom(self.line18)
            self.viewer.add_geom(self.line19)
            self.viewer.add_geom(self.line20)
            self.viewer.add_geom(self.line21)
            self.viewer.add_geom(self.line22)
            self.viewer.add_geom(self.mountain)
            self.viewer.add_geom(self.robot)

        """为了让结果可视化，我们需要自己渲染结果，比如我打算设置一个1000 * 50的窗口，并且使用它们的中心点"""
        state_position_mapping_dict = {}
        for state_num in range(3):
            state_position_mapping_dict[state_num] = [125 + 50 * state_num, 125]



        self.robotrans.set_translation(state_position_mapping_dict[self.current_state][0],
                                       state_position_mapping_dict[self.current_state][1])

        # 渲染每一个格子的Policy是什么
        if policy_list is not None:
            for i in range(self.state_size):
                current_x = state_position_mapping_dict[i][0]
                current_y = state_position_mapping_dict[i][1]
                action_mapping_dict = {
                    0: [(current_x + 15, current_y - 10), (current_x + 15, current_y + 10), (current_x - 15, current_y)],
                    #1: [(current_x, current_y + 15), (current_x - 10, current_y - 15), (current_x + 10, current_y - 15)],
                    1: [(current_x - 15, current_y - 10), (current_x - 15, current_y + 10), (current_x + 15, current_y)],
                    #3: [(current_x, current_y - 15), (current_x - 10, current_y + 15), (current_x + 10, current_y + 15)]
                }
                current_action = policy_list[i]
                temp = rendering.FilledPolygon(action_mapping_dict[current_action])
                temp.set_color(1, 0, 0)
                self.viewer.add_onetime(temp)  # 只在最开始的时候加上Policy的箭头

        return_rgb_array_flag = mode == 'rgb_array'
        return self.viewer.render(return_rgb_array=return_rgb_array_flag)

    def brute_force_solver(self, gamma, computing_period):
        """
        本函数用于直接使用DP方法求解MDP模型，并返回value function，policy和state Distribution
        :return:
        """

        fh = mdptoolbox.mdp.QLearning(self.transition_matrix, self.reward_tensor, gamma, computing_period)
        fh.run()
        # Policy Matrix的维度为(S, N)，其中[:,0]列为最新的计算得出的Policy
        optimal_policy = fh.policy[:, 0]

        # 使用optimal policy进行运算，获得最终的state dict
        state_count = np.zeros(self.state_size)
        accumulate_reward = 0
        self.reset()
        # self.render()
        while True:
            current_action = optimal_policy[self.current_state]
            obs, reward, done = self.step(current_action)
            # self.render()
            # time.sleep(0.2)
            state_count[obs] += 1
            accumulate_reward += reward

            if done:
                break

        total_step = self.step_count
        self.reset()
        if self.viewer is not None:
            self.close()

        return optimal_policy, fh.V[:, 0], state_count/total_step

    def get_state_action_next_state_tuple(self):

        sas_tuple_list = []
        sample_data_index = 0

        for current_state in range(self.state_size):

            if current_state in self.get_termination_state():
                continue

            self.reset(current_state)

            current_state_obs = self.obs()
            for action in range(self.action_size):
                tmp_sas_tuple = [current_state_obs, action]
                obs, reward, done = self.step(action)

                tmp_sas_tuple.append(self.obs())
                tmp_sas_tuple.append(reward)
                tmp_sas_tuple.append(done)
                sas_tuple_list.append(tmp_sas_tuple)
                self.current_state = current_state
                sample_data_index += 1

        non_repeat_sample_data_length = len(sas_tuple_list)

        return sas_tuple_list, non_repeat_sample_data_length

    def given_policy_and_td_error_generate_trajectory_td_error(self, policy_list):

        trajectory_dict = {}

        for current_state in range(self.state_size):
            trajectory_dict[current_state] = []

            if current_state in self.get_termination_state():
                continue

            self.reset(current_state)

            done = False
            while not done:
                action = policy_list[self.current_state]
                temp_state = self.current_state
                next_state, reward, done = self.step(action)
                trajectory_dict[current_state].append((temp_state, action))

                # 如果撞墙了就停止
                if next_state == temp_state:
                    break
                # 如果循环了就停止
                if len(trajectory_dict[current_state]) >= 2 and next_state == trajectory_dict[current_state][-2][0]:
                    break

        return trajectory_dict

    def get_full_state_action_state_state_tuple(self):

        sas_tuple_list = []

        for current_state in range(self.state_size):

            self.reset(current_state)

            for action in range(self.action_size):
                tmp_sas_tuple = [self.obs(), action]
                obs, reward, done = self.step(action)
                tmp_sas_tuple.append(self.obs())
                tmp_sas_tuple.append(reward)
                tmp_sas_tuple.append(done)
                sas_tuple_list.append(tmp_sas_tuple)

                self.current_state = current_state

        return sas_tuple_list

    def get_reward_function(self):
        reward_list = []
        for next_state in range(self.state_size):
            reward = _reward_rule(next_state=next_state, win_reward=self.win_reward,
                                  step_penalty=self.step_penalty)
            reward_list.append(reward)
        return reward_list


    def get_mean_reward_from_all_state_given_policy(self, policy_list):

        total_reward = 0

        for current_state in range(self.state_size):

            if current_state in self.get_termination_state():
                continue

            self.reset(current_state)

            done = False
            temp_reward = 0
            while not done:
                _, reward, done = self.step(policy_list[self.current_state])
                temp_reward += reward

            total_reward += temp_reward

        return total_reward/(self.state_size - len(self.get_termination_state()))

# env = DeterministicMDPSimple(1, 20, 1)
# data, _ = env.get_state_action_next_state_tuple()
# print("")
# policy, _, _ = env.brute_force_solver(gamma=0.9, computing_period=100)
# env.get_mean_reward_from_all_state_given_policy([1, 1, 0, 0, 0])
# env.render(policy_list=policy)
# env.close()
