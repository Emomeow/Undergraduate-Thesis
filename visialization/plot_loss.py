import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
# import numpy as np  # Currently unused

sns.set_theme(style="darkgrid")

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def plot_loss(loss_list, gradient_type):

    data_dict_list = []
    increase_count = 0

    increase_data_point = []
    increase_data_loss = []

    for i in range(len(loss_list)):
        tmp_dict = {}
        tmp_dict["training count"] = i
        tmp_dict["loss"] = loss_list[i]
        data_dict_list.append(tmp_dict)

        if i < len(loss_list) - 1:
            if loss_list[i] < loss_list[i + 1]:
                increase_count += 1
                increase_data_loss.append(loss_list[i])
                increase_data_point.append(i)

    df = pd.DataFrame().from_dict(data_dict_list)
    sns.lineplot(data=df, x="training count", y="loss")
    plt.scatter(x=increase_data_point, y=increase_data_loss, c="r")
    plt.title(gradient_type + "_loss, loss increase count=" + str(increase_count))
    out_dir = os.path.join(CURRENT_PATH, "../img")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{gradient_type}_loss.png"))
    plt.close()

    data_dict_list = []

    for i in range(len(loss_list)):
        tmp_dict = {}
        tmp_dict["training count"] = i
        tmp_dict["log scale loss"] = math.log10(loss_list[i])
        data_dict_list.append(tmp_dict)

    df = pd.DataFrame().from_dict(data_dict_list)
    sns.lineplot(data=df, x="training count", y="log scale loss")
    plt.title(gradient_type + "_loss")
    plt.savefig(os.path.join(out_dir, f"{gradient_type}_loss_log_scale.png"))
    plt.close()


# plt_accumulate_reward([[1.1, 2,3.7, 5.4,1.1, 2,3.7, 5.4,1.1, 2,3.7, 5.4], [1.0, 2,3.2, 5.4,1.5, 2,3.7, 5.1,1.1, 2,3.7, 5.2]])


