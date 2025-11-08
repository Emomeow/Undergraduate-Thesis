import os
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

def plot_policy_in_env(env, last_policy_list, gradient_type):
    env.reset()
    policy_graph = env.render(mode="rgb_array", policy_list=last_policy_list)
    cv2.imwrite(CURRENT_PATH + "/../img/" + gradient_type + "_policy.png",
                cv2.cvtColor(policy_graph, cv2.COLOR_RGB2BGR))

# plot_policies([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,], [1,1,1,2,3,1,1,1,1,1,2,3,1,1,1,1,1,2,3,1,1,], [1,1,3,3,3,3,1,1,1,3,3,3,3,1,1,1,3,3,3,3,1,], [1,2,2,2,2,2,1,1,2,2,2,2,2,1,1,2,2,2,2,2,1,]], "semi")
