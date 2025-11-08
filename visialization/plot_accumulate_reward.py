"""
Plot the accumulated reward during training.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def plt_accumulate_reward(reward_list, gradient_type):
    # Robustly convert reward_list to a 1-D numeric series for plotting.
    try:
        reward_array = np.array(reward_list)
    except Exception:
        # fallback: coerce element-wise
        reward_array = np.array([r for r in reward_list], dtype=object)

    # If elements are lists/arrays (object or 2-D), reduce to a scalar per epoch by taking the mean
    if reward_array.dtype == object:
        scalar_list = []
        for r in reward_array:
            if r is None:
                scalar_list.append(np.nan)
            elif isinstance(r, (list, tuple, np.ndarray)):
                try:
                    scalar_list.append(float(np.mean(r)))
                except Exception:
                    scalar_list.append(np.nan)
            else:
                try:
                    scalar_list.append(float(r))
                except Exception:
                    scalar_list.append(np.nan)
        reward_array = np.array(scalar_list, dtype=np.float32)
    else:
        # numeric dtype: if multi-dimensional, average across trailing axes to get a per-epoch scalar
        if reward_array.ndim > 1:
            reward_array = reward_array.mean(axis=1)

    x = range(len(reward_array))
    plt.plot(x, reward_array)
    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    plt.title(f'Training Progress - {gradient_type}')
    plt.grid(True)
    # ensure output directory exists and save the figure (headless-safe)
    out_dir = os.path.join(CURRENT_PATH, "../img")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gradient_type}_accumulated_reward.png")
    plt.savefig(out_path, bbox_inches='tight')
    # no CSV output required; only save PNG
    plt.close()