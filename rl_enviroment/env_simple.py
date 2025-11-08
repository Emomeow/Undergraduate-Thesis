"""
This module provides simple RL environments: a basic Deterministic MDP
and utilities to interact with it. Customize as needed.
"""
import os
import gymnasium as gym
# import copy  # Unused but may be needed for future deep copying
# import pickle  # Unused but may be needed for serialization
import numpy as np
import mdptoolbox
import pygame
# from itertools import product  # Unused but may be needed for state space generation

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def _transition_rule(state, action):
    """
    Given the current state and action, return the next state.
    :param state: current state index
    :param action: action index
    :return: next state index
    """
    if action == 0:
        if state in [0]:
            return state
            # return 4  # removed self-recurrent case
        else:
            return state - 1
    elif action == 1:
        if state in [2]:
            return state
            # return 0  # removed self-recurrent case
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
    # Reward rule: give win_reward when reaching a terminal state; otherwise apply step_penalty
    if next_state in [1]:
        return win_reward
    else:
        return step_penalty


def _generate_one_hot_array(total_length, pos, value):
    # Generate an array of length total_length with `value` placed at index pos
    one_hot_array = [0] * total_length
    one_hot_array[pos] = value
    return one_hot_array


def _generate_transition_tensor(state_size, action_size):
    # Generate transition tensor according to movement rules (deterministic transitions)
    transition_tensor = []
    for action in range(action_size):
        transition_tensor.append([])
        for state in range(state_size):
            next_state = _transition_rule(state, action)
            transition_tensor[action].append(_generate_one_hot_array(state_size, next_state, 1))
    return transition_tensor


def _generate_reward_tensor(state_size, action_size, step_penalty, win_reward):
    # Generate reward tensor for each (action, state) pair
    reward_tensor = []
    for action in range(action_size):
        # first create an empty list
        reward_tensor.append([])
        for state in range(state_size):
            # compute the reward for each state after moving left or right
            next_state = _transition_rule(state, action)
            next_reward = _reward_rule(next_state=next_state, step_penalty=step_penalty, win_reward=win_reward)
            reward_tensor[action].append(_generate_one_hot_array(state_size, next_state, next_reward))
    return reward_tensor


class DeterministicMDPSimple(gym.Env):

    # Define a viewer (rendering canvas) in the initializer
    def __init__(self, state_representation_size, max_step, state_representation_random_seed, varepsilon):
        super(DeterministicMDPSimple, self).__init__()
        self.current_state = 0
        self.viewer = None
        self.robot_pos = [125, 125]  # Initial position
        self.robot_radius = 15
        self.action_size = 2
        self.state_size = 3
        self.state_representation_size = state_representation_size
        self.viewer = None

        self.max_step = max_step
        self.step_count = 0

    # Action order: left/right (two actions)
        self.transition_matrix = _generate_transition_tensor(self.state_size, self.action_size)
        self.transition_matrix = np.array(self.transition_matrix)

        self.step_penalty = -0.05
        self.win_reward = -0.05
        self.reward_tensor = _generate_reward_tensor(self.state_size, self.action_size, step_penalty=self.step_penalty,
                                                     win_reward=self.win_reward)
        self.reward_tensor = np.array(self.reward_tensor)

    # create new MDP model
        if state_representation_size != 1 and state_representation_size != 3:
            # Control initialization of state representations
            np.random.seed(state_representation_random_seed)
            # Construct dense state representations and mapping between encodings and indices
            full_rank = False
            while not full_rank:
                self.state_representation = np.random.rand(self.state_size, self.state_representation_size)

                if np.linalg.matrix_rank(self.state_representation) == min(self.state_size,
                                                                           self.state_representation_size):
                    full_rank = True

            # Each row vector represents a state; normalize each to unit L2 norm
            for i in range(self.state_representation.shape[0]):
                self.state_representation[i] /= np.linalg.norm(self.state_representation[i], ord=2)
        if state_representation_size == 3:
            
            self.state_representation = []
            # coordinate-style representations (unused here)
            '''
            for i in range(1,6):
                for j in range(1,6):
                    self.state_representation.append([0.1*j,0.1*i])
                    
            del self.state_representation[12]
            self.state_representation = np.array(self.state_representation)
            '''
            # one-hot style representations with small perturbations
            states = []
            for i in range(state_representation_size):
                k = varepsilon * np.ones(state_representation_size, dtype=np.float32)
                k[i] = np.sqrt(1 - (state_representation_size - 1) * varepsilon**2)
                states.append(k)
            self.state_representation = np.array(states, dtype=np.float32).reshape([3, 3])
        else:
            self.state_representation = np.arange(start=0, stop=1, step=1 / self.state_size, dtype=np.float32).reshape(self.state_size, 1)

    # Build mapping from state encoding (list) to state index
        self.state_encode_to_index_mapping = {}
        for i in range(self.state_size):
            # Convert to tuple for hashable key
            state_key = tuple(self.state_representation[i].tolist())
            self.state_encode_to_index_mapping[state_key] = i

    def reset(self, state=None):
        if state is None:
            self.current_state = 0
        else:
            self.current_state = state
        self.step_count = 0

    def get_index_by_obs(self, obs):
        # Convert observation to tuple for dictionary lookup
        return self.state_encode_to_index_mapping[tuple(obs.tolist())]

    def get_termination_state(self):
        # Return terminal state(s)
        return [1]

    def get_positive_pretermination_state(self):
        return [1]

    def get_negative_pretermination_state(self):
        return []

    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None

    # Return the state representation corresponding to the current state
    def obs(self):
        # Return state representation directly
        state = self.state_representation[self.current_state]
        if not isinstance(state, np.ndarray):
            print(f"Warning: state is not ndarray but {type(state)}, value: {state}")
            state = np.array(state, dtype=np.float32)
        return state.astype(np.float32)

    def step(self, action):
        # Take one step and return (next_state, reward, done)
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
        Override Env.render to provide simple visualization of the environment.
        :param mode: render mode ('human' or 'rgb_array')
        :param close: unused
        :param policy_list: optional policy to render arrows for each state
        :return: rgb array if mode == 'rgb_array', otherwise None
        """
        if close:
            if self.viewer is not None:
                pygame.quit()
                self.viewer = None
            return

        screen_width = 1500
        screen_height = 800

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption('Simple MDP Environment')

        self.viewer.fill((255, 255, 255))  # White background
        
        # Draw horizontal and vertical lines
        for y in range(100, 601, 50):  # Horizontal lines
            pygame.draw.line(self.viewer, (0, 0, 0), (100, y), (600, y))
        
        for x in range(100, 601, 50):  # Vertical lines
            pygame.draw.line(self.viewer, (0, 0, 0), (x, 100), (x, 600))

            # Draw goal state
            pygame.draw.polygon(self.viewer, (0, 255, 255), [
                (100, 100), (100, 150), (150, 150), (150, 100)
            ])
            
            # Initialize robot position (will be updated in render)
            self.robot_pos = [125, 125]  # Initial position
            self.robot_radius = 15

        # Map states to screen positions
        state_position_mapping_dict = {}
        for state_num in range(3):
            state_position_mapping_dict[state_num] = [125 + 50 * state_num, 125]
        
        # Update robot position
        self.robot_pos = [state_position_mapping_dict[self.current_state][0],
                         state_position_mapping_dict[self.current_state][1]]
        
        # Draw robot
        pygame.draw.circle(self.viewer, (255, 204, 0), self.robot_pos, self.robot_radius)
        
        # Draw policy arrows if provided
        if policy_list is not None:
            for i in range(self.state_size):
                current_x = state_position_mapping_dict[i][0]
                current_y = state_position_mapping_dict[i][1]
                action_mapping_dict = {
                    0: [(current_x + 15, current_y - 10), (current_x + 15, current_y + 10), (current_x - 15, current_y)],
                    1: [(current_x - 15, current_y - 10), (current_x - 15, current_y + 10), (current_x + 15, current_y)]
                }
                current_action = policy_list[i]
                pygame.draw.polygon(self.viewer, (255, 0, 0), action_mapping_dict[current_action])
        
        # Update display
        pygame.display.flip()
        
        if mode == 'rgb_array':
            data = pygame.surfarray.array3d(self.viewer)
            return data
        return None
        return self.viewer.render(return_rgb_array=return_rgb_array_flag)

    def brute_force_solver(self, gamma, computing_period):
        """
        Solve the MDP using dynamic programming / Q-learning and return
        the optimal policy, value function, and state distribution.
        :return: (optimal_policy, value_function, state_distribution)
        """

        fh = mdptoolbox.mdp.QLearning(self.transition_matrix, self.reward_tensor, gamma, computing_period)
        fh.run()
    # The policy matrix has shape (S, N); column [:,0] is the most recently computed policy
        optimal_policy = fh.policy[:, 0]

    # Simulate using the optimal policy to obtain the final state distribution
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

                # stop if the action causes no movement (hit wall)
                if next_state == temp_state:
                    break
                # stop if a loop is detected in the trajectory
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
