import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid")

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def plt_accumulate_reward(reward_list, gradient_type):

    data_dict_list = []

    for i in range(len(reward_list)):
        tmp_dict = {}
        tmp_dict["trajectory count"] = i
        tmp_dict["accumulate reward"] = reward_list[i]
        data_dict_list.append(tmp_dict)

    df = pd.DataFrame().from_dict(data_dict_list)
    sns.lineplot(data=df, x="trajectory count", y="accumulate reward")
    plt.title(gradient_type + "_accumulate_reward")
    plt.savefig(CURRENT_PATH + "/../img/" + gradient_type + "_accumulate_reward.png")
    plt.close()

# plt_accumulate_reward([[1.1, 2,3.7, 5.4,1.1, 2,3.7, 5.4,1.1, 2,3.7, 5.4], [1.0, 2,3.2, 5.4,1.5, 2,3.7, 5.1,1.1, 2,3.7, 5.2]], "true", "11")
