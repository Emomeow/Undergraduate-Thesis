"""
定义用于存储从环境中采样的数据的结构，在Offline Learning中是非常重要的结构，在训练神经网络的时候要从其中采样作为训练数据
"""
import random

class SimpleReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_tuple = []
        self.buffer_size = buffer_size
        self.data_num_count = 0

    def insert_data_tuple(self, current_state, current_action, next_state, reward):
        """
        本函数采用循环链表插入的方法，即当Buffer满了之后，会从index0处开始进行数据覆盖
        :param current_state:
        :param current_action:
        :param next_state:
        :param reward:
        :return:
        """
        if self.get_buffer_length() < self.buffer_size:
            self.buffer_tuple.append((current_state, current_action, next_state, reward))
        else:
            self.buffer_tuple[self.data_num_count % self.buffer_size] = \
                (current_state, current_action, next_state, reward)
        self.data_num_count += 1

    def insert_data_tuple_list(self, data_tuple_list):
        for data_tuple in data_tuple_list:
            self.insert_data_tuple(data_tuple["current_state"], data_tuple["current_action"], data_tuple["next_state"],
                                   data_tuple["reward"])

    def get_sequential_batch_data(self, batch_size):
        if batch_size > self.buffer_size:
            raise Exception("请求数据量过大")

        if self.data_num_count < batch_size:
            batch_size = self.data_num_count

        return self.buffer_tuple[0: batch_size]

    def get_shuffle_batch_data(self, batch_size):
        if batch_size > self.buffer_size:
            raise Exception("请求数据量过大")

        if batch_size < len(self.buffer_tuple):
            result = random.sample(self.buffer_tuple, batch_size)
        else:
            result = random.sample(self.buffer_tuple, len(self.buffer_tuple))

        return result

    def get_buffer_length(self):
        return len(self.buffer_tuple)
