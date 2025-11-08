import os
# import imageio  # Currently unused
# import shutil  # Currently unused
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl  # Used for colors
# from itertools import product  # Currently unused
# from datetime import datetime  # Currently unused


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(CURRENT_PATH, "../img")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_current_value_function(value_function_list, state_size, gradient_type):
    colors = pl.cm.jet(np.linspace(0, 1, state_size))

    for state_num in range(state_size):
        data_dict_list = []

        for i in range(len(value_function_list)):
            tmp_dict = {}
            tmp_dict["training count"] = i
            tmp_dict["value"] = value_function_list[i][state_num]
            data_dict_list.append(tmp_dict)

        df = pd.DataFrame().from_dict(data_dict_list)
        sns.lineplot(data=df, x="training count", y="value", legend='brief', label="state_" + str(state_num), color=colors[state_num])
    plt.legend(bbox_to_anchor=(1.05, 1.45), ncol=int(state_size/3))
    plt.title(gradient_type + "_mean_temporal_difference")
    plt.savefig(os.path.join(OUT_DIR, f"{gradient_type}_value_function_for_each_state.png"), bbox_inches='tight', pad_inches=1)
    plt.close()

def plot_state_action_value_function(state_action_value_function_list, state_size, action_size, gradient_type):
    colors = pl.cm.jet(np.linspace(0, 1, state_size * action_size))

    for state_num in range(state_size):
        for action_num in range(action_size):
            data_dict_list = []

            for i in range(len(state_action_value_function_list)):
                tmp_dict = {}
                tmp_dict["training count"] = i
                tmp_dict["value"] = state_action_value_function_list[i][state_num][action_num]
                data_dict_list.append(tmp_dict)

            df = pd.DataFrame().from_dict(data_dict_list)
            sns.lineplot(data=df, x="training count", y="value", legend='brief',
                         label="state_" + str(state_num) + "_action_" + str(action_num), color=colors[state_num * 2 + action_num])
    # plt.legend(bbox_to_anchor=(1.05, 1.45), ncol=int(state_size/4))
    plt.title(gradient_type + "_state_action_value")
    plt.savefig(os.path.join(OUT_DIR, f"{gradient_type}_value_function_for_each_state_action.png"), bbox_inches='tight', pad_inches=1)
    plt.close()

    plt.figure(figsize=(45, 20))
    for state_num in range(state_size):
        for action_num in range(action_size):
            data_dict_list = []

            for i in range(len(state_action_value_function_list)):
                tmp_dict = {}
                tmp_dict["training count"] = i
                tmp_dict["value"] = state_action_value_function_list[i][state_num][action_num]
                data_dict_list.append(tmp_dict)

            df = pd.DataFrame().from_dict(data_dict_list)
            plt.subplot(action_size, state_size, action_size * state_num + action_num + 1)
            sns.lineplot(data=df, x="training count", y="value", legend='brief',
                     label="state_" + str(state_num) + "_action_" + str(action_num), color=colors[state_num * 2 + action_num])
    plt.title(gradient_type + "_state_action_value_subplot")
    plt.savefig(os.path.join(OUT_DIR, f"{gradient_type}_value_function_for_each_state_action_subplot.png"),
        bbox_inches='tight', pad_inches=1)
    plt.close()
# plot_value_function([[-0.2, 0.2, 0.7, -0.7, 0.4, -0.5], [-0.7, 0.2, 0.3, -0.3, 0.4, -0.5], [-0.3, 0.2, 0.1, -0.7, 0.4, -0.6]], "true")
