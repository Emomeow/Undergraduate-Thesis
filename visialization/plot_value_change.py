import os
import imageio
import shutil
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from itertools import product
from datetime import datetime

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

def generate_dist():
    dist = []
    dist.append([8])
    dist.append([5,7])
    dist.append([2,4,6])
    dist.append([1,3])
    dist.append([0])
    return dist

#得到按不同距离分类的价值函数并计算均值与振幅
def get_value_mean_err(dist, value_function_list, index):
    value = []
    means = []
    err = []
    for k in range(len(dist)):
        value.append([])
        for j in range(len(dist[k])):
            value[k].append(value_function_list[index][dist[k][j]])
        means.append(np.mean(value[k]))
        err.append(np.var(value[k]))
    return means, err

def plot_value_change(loss_list, value_function_list, state_size, gradient_type):
    dist = generate_dist()
    #得到前期的增长点，并把第一个与最后一个列为phase
    means, err = get_value_mean_err(dist, value_function_list, len(loss_list) - 1)
    x = range(len(dist))
    plt.errorbar(x, means, yerr = err)
    #plt.legend(bbox_to_anchor=(1.05, 1.45), ncol=int(state_size/3))
    plt.title(gradient_type + "_mean_temporal_difference")
    plt.savefig(CURRENT_PATH + "/../img/" + gradient_type + "_value_change_for_each_state.png", bbox_inches='tight', pad_inches=1)
    plt.close()


def plot_var(value_function_list, gradient_type):
    data_dict_list = []
    dist = generate_dist()
    for i in range(len(value_function_list)):
        means, err = get_value_mean_err(dist, value_function_list, i)
        a = 0
        c=0
        for j in range(len(err)):
            a += len(dist[j]) * err[j]
        if a > 0:
            b = math.log10(a)
        else :
            b = -14
        tmp_dict = {}
        tmp_dict["training count"] = i
        tmp_dict["kappa"] = a
        tmp_dict["kappa_log_scale"] = b
        data_dict_list.append(tmp_dict)
    df = pd.DataFrame().from_dict(data_dict_list)
    sns.lineplot(data=df, x="training count", y="kappa")
    plt.title(gradient_type + "_kappa_change")
    plt.savefig(CURRENT_PATH + "/../img/" + gradient_type + "_var_change.png", bbox_inches='tight', pad_inches=1)
    plt.close()
    
    sns.lineplot(data=df, x="training count", y="kappa_log_scale")
    plt.title(gradient_type + "_kappa_change_log_scale")
    plt.savefig(CURRENT_PATH + "/../img/" + gradient_type + "_var_change_log_scale.png", bbox_inches='tight', pad_inches=1)
    plt.close()
    
    

