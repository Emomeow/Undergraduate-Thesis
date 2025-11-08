import os
import cv2
# import pandas as pd  # Currently unused
# import seaborn as sns  # Currently unused 
# import matplotlib.pyplot as plt  # Currently unused


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

def plot_policy_in_env(env, last_policy_list, gradient_type):
    env.reset()
    policy_graph = env.render(mode="rgb_array", policy_list=last_policy_list)
    out_dir = os.path.join(CURRENT_PATH, "../img")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gradient_type}_policy.png")
    # If environment render returned an image array, try to save it using OpenCV
    if policy_graph is not None:
        try:
            # cv2.imwrite returns True/False; check success explicitly
            img_to_save = cv2.cvtColor(policy_graph, cv2.COLOR_RGB2BGR)
            write_success = cv2.imwrite(out_path, img_to_save)
            # no CSV output required; only save PNG
            if write_success:
                return
            # if write failed (False), fall through to matplotlib fallback
        except Exception:
            # Fall through to matplotlib fallback
            pass

    # Fallback for headless environments or when pygame can't render:
    # draw a small 3-cell horizontal grid and arrows showing the action per state
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrow

        fig, ax = plt.subplots(figsize=(6, 2))
        # grid parameters to match env layout roughly
        cell_width = 1.0
        cell_height = 1.0
        spacing = 0.2
        n_states = len(last_policy_list)
        total_width = n_states * (cell_width + spacing)

        # draw cells
        for i in range(n_states):
            x0 = i * (cell_width + spacing)
            rect = Rectangle((x0, 0), cell_width, cell_height, fill=False, edgecolor='black')
            ax.add_patch(rect)
            # draw goal cell (state 1 in env_simple is terminal) with cyan fill if present
            try:
                if i in getattr(env, 'get_termination_state', lambda: [])():
                    rect_goal = Rectangle((x0, 0), cell_width, cell_height, facecolor=(0.0, 1.0, 1.0, 0.5))
                    ax.add_patch(rect_goal)
            except Exception:
                pass

            # draw arrow indicating policy action: 0 -> left, 1 -> right
            mid_x = x0 + cell_width / 2
            mid_y = 0.5 * cell_height
            action = int(last_policy_list[i])
            arrow_dx = -0.4 * cell_width if action == 0 else 0.4 * cell_width
            # arrow head length
            arrow = FancyArrow(mid_x, mid_y, arrow_dx, 0, width=0.08, length_includes_head=True, color='red')
            ax.add_patch(arrow)

        ax.set_xlim(-0.2, total_width)
        ax.set_ylim(-0.1, cell_height + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Policy ({gradient_type})')
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

        # no CSV output required; only save PNG
    except Exception:
        # give up silently (don't crash the training run)
        return

# plot_policies([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,], [1,1,1,2,3,1,1,1,1,1,2,3,1,1,1,1,1,2,3,1,1,], [1,1,3,3,3,3,1,1,1,3,3,3,3,1,1,1,3,3,3,3,1,], [1,2,2,2,2,2,1,1,2,2,2,2,2,1,1,2,2,2,2,2,1,]], "semi")
