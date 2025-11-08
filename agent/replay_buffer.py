"""
定义用于存储从环境中采样的数据的结构，在Offline Learning中是非常重要的结构，在训练神经网络的时候要从其中采样作为训练数据
"""
import random
import numpy as np

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
        batch_data = self.buffer_tuple[0: batch_size]
        # Return as list of tuples (s, a, s', r) which training functions expect
        return batch_data

    def get_shuffle_batch_data(self, batch_size):
        if batch_size > self.buffer_size:
            raise Exception("请求数据量过大")

        if batch_size < len(self.buffer_tuple):
            batch_data = random.sample(self.buffer_tuple, batch_size)
        else:
            batch_data = random.sample(self.buffer_tuple, len(self.buffer_tuple))
        
        # Return as list of tuples (s, a, s', r) which training functions expect
        return batch_data

    def get_buffer_length(self):
        return len(self.buffer_tuple)

    def _format_batch_data(self, batch_data):
        """
        Convert list of tuples into dictionary format with proper numpy array conversion
        """
        batch_dict = {
            "current_states": [],
            "actions": [],
            "next_states": [],
            "rewards": []
        }
        
        for current_state, action, next_state, reward in batch_data:
            # Make sure current_state is a numpy array
            if isinstance(current_state, (list, tuple)):
                current_state = np.array(current_state, dtype=np.float32)
            batch_dict["current_states"].append(current_state)
            batch_dict["actions"].append(action)
            
            # Handle next_state which might be None for terminal states
            if next_state is not None:
                if isinstance(next_state, (list, tuple)):
                    next_state = np.array(next_state, dtype=np.float32)
            batch_dict["next_states"].append(next_state)
            
            batch_dict["rewards"].append(reward)
        
        # Convert lists to numpy arrays, handling None values in next_states
        batch_dict["current_states"] = np.stack(batch_dict["current_states"]).astype(np.float32)
        batch_dict["actions"] = np.array(batch_dict["actions"], dtype=np.int64)
        
        # Handle next_states separately due to potential None values
        valid_next_states = [s for s in batch_dict["next_states"] if s is not None]
        if valid_next_states:
            batch_dict["next_states"] = np.stack(valid_next_states).astype(np.float32)
        else:
            batch_dict["next_states"] = np.array([])
            
        batch_dict["rewards"] = np.array(batch_dict["rewards"], dtype=np.float32)
        
        return batch_dict