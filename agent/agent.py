"""
Define the reinforcement learning Agent. In RL, the two main components
are the Agent and the Environment. This module implements the Agent logic
and training routines; environments are typically provided separately.
"""
# import copy  # Unused but may be needed for future deep copying
import torch
import numpy as np
from agent.replay_buffer import SimpleReplayBuffer
from agent.nn_model import FCNN
from agent.model_operation import save_model, \
    load_model, training_fcnn_model_with_true_gradient, training_fcnn_model_with_semi_gradient
from agent.model_compute_loss import loss_with_semi_gradient, loss_with_true_gradient

# setting use of the GPU or CPU
USE_CUDA = torch.cuda.is_available()
# if the GPU is available for the server, the device is GPU, otherwise, the device is CPU
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")


class MDPAgent(object):
    def __init__(self, buffer_size, state_representation_size, action_size, optimizer_type,
                 init_learning_rate, gradient_type, lr_discount_factor, lr_discount_epoch, training_num):
        self.replay_buffer = SimpleReplayBuffer(buffer_size=buffer_size)
        self.buffer_size = buffer_size
        self.gradient_type = gradient_type
        self.training_num = training_num

    # If a trained model exists, load it; otherwise create a new model
        model, optimizer, lr_scheduler = load_model("fcnn_" + gradient_type)

        if model is None:
            self.model = FCNN(input_size=state_representation_size, output_size=action_size)
        else:
            self.model = model
        # Make sure model is on the correct device
        self.model = self.model.to(DEVICE)
        #self.target = FCNN(input_size=self.model.input_size, output_size=self.model.output_size)
        #self.target.load_state_dict(self.model.state_dict())
        #self.target.to(DEVICE)

        self.learning_rate = init_learning_rate

        if optimizer is None:
            if optimizer_type == "adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif optimizer_type == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            else:
                raise Exception("Unrecognized optimizer type: " + optimizer_type)
        else:
            self.optimizer = optimizer
        
        if lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer, step_size=int(lr_discount_epoch/self.training_num), gamma=lr_discount_factor)
        else:
            self.lr_scheduler = lr_scheduler

    # Track training and exploration counts to adjust learning rate and exploration randomness
        self.training_count = 0
        self.exploration_count = 0

    def get_current_policy_and_value(self, current_state_data, action_data, state_representation):
        # Convert inputs to numpy arrays first
        state_rep = np.array(state_representation, dtype=np.float32)
        current_state = np.array(current_state_data, dtype=np.float32)
        action_data = np.array(action_data, dtype=np.int64)
        
        # Create tensors from numpy arrays
        state_representation_tensor = torch.from_numpy(state_rep).to(DEVICE)
        current_state_representation_tensor = torch.from_numpy(current_state).to(DEVICE)
        action_tensor = torch.from_numpy(action_data).view(-1, 1).to(DEVICE)
        
        # Ensure model is on the correct device
        self.model = self.model.to(DEVICE)
        current_state_value_tensor = self.model(current_state_representation_tensor).gather(1, action_tensor).squeeze(-1)

        current_state_value_tensor = current_state_value_tensor.cpu().detach().numpy().tolist()

        state_value_tensor = self.model(state_representation_tensor)
        # fix terminal state's value (example placeholder)
        '''
        state_value_tensor[1,0] = 1
        state_value_tensor[1,1] = 1
        '''
        current_value, current_policy = state_value_tensor.max(1)
        current_action_list = current_policy.cpu().detach().numpy().tolist()
        state_value_function = current_value.cpu().detach().numpy().tolist()
        Q_value_list = state_value_tensor.cpu().detach().numpy().tolist()
        #target_value_tensor = self.target(state_representation_tensor)
        #target_value, target_policy = target_value_tensor.max(1)
        #target_action_list = target_policy.cpu().detach().numpy().tolist()
        #target_value_function = target_value.cpu().detach().numpy().tolist()

        return np.mean(current_state_value_tensor), current_action_list, state_value_function, Q_value_list

    def offline_learning(self, batch_size, gamma, epoch_num):
        loss_value_list = []
        for _ in range(self.training_num):
            buffer_size = self.buffer_size
            sample_data = self.replay_buffer.get_shuffle_batch_data(batch_size)
            loss_data = self.replay_buffer.get_sequential_batch_data(buffer_size)


            if self.gradient_type == "true":
                training_fcnn_model_with_true_gradient(fcnn_model=self.model, device=DEVICE,
                                                                      input_data=sample_data, optimizer=self.optimizer,
                                                                      gamma=gamma)

                current_loss = loss_with_true_gradient(fcnn_model=self.model, device=DEVICE,
                                                                      input_data=loss_data, optimizer=self.optimizer,
                                                                      gamma=gamma)                                           
            elif self.gradient_type == "semi":
                #if epoch_num % 1 == 0:
                    #self.target.load_state_dict(self.model.state_dict())
                training_fcnn_model_with_semi_gradient(fcnn_model=self.model, device=DEVICE,
                                                                      input_data=sample_data, optimizer=self.optimizer,
                                                                      gamma=gamma)
                current_loss = loss_with_semi_gradient(fcnn_model=self.model, device=DEVICE,
                                                                      input_data=loss_data, optimizer=self.optimizer,
                                                                      gamma=gamma)
            else:
                raise Exception("can't recognize gradient_type: " + self.gradient_type)

            loss_value_list.append(current_loss)

        self.lr_scheduler.step()
        save_model(model=self.model, optimizer=self.optimizer, filename="fcnn_" + self.gradient_type,
                   lr_scheduler=self.lr_scheduler)

        return loss_value_list
