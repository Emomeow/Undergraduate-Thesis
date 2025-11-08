## Developer cheat sheet — quick entry points

This short cheat sheet lists the main files and entry points you will most likely modify when working on training, models, replay buffer, environment rules, and visualization. Each entry includes the file path and the line number where the symbol is defined in the current workspace.

Keep this file small — use it as a table-of-contents for deeper edits.

---

1) Top-level runner
- File: `main_func.py` — def `main_func(...)` at line 40 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/main_func.py#L40)
  - Purpose: main training loop, hyper-parameters parsing, instantiates `DeterministicMDPSimple`, `MDPAgent`, runs epochs and calls all visualization helpers.
  - Key plotting call locations (near end of function):
    - `plt_accumulate_reward(...)` at line 219
    - `plot_loss(...)` at line 220
    - `plot_policy_in_env(...)` at line 221
    - `plot_current_value_function(...)` at line 222
    - `plot_state_action_value_function(...)` at line 224
  - Typical edits: experiment orchestration (epoch count, learning rate schedule, which plots to produce), data collection before plotting.

2) Agent and training
2) Agent and training
-- File: `agent/agent.py` — `class MDPAgent` at line 21 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/agent.py#L21)
  - `get_current_policy_and_value(...)` at line 64 — computes policy & value from the model given state representations.
  - `offline_learning(...)` at line 98 — the per-epoch training loop that samples the replay buffer and calls training/loss functions.
  - Typical edits: change how the model is called, alter training loops or logging frequency, add hooks for target networks.

3) Network definition
  - `get_initialization(...)` at line 30 — directly sets layer weights from numpy arrays.
  - `forward(...)` — standard feed-forward (modify activation sizes, add layers, etc.).
  - Typical edits: model architecture, initialization, forward pass behavior.
-- File: `agent/nn_model.py` — `class FCNN` at line 16 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/nn_model.py#L16)

4) Training routines (semi / true gradient)
-- File: `agent/model_operation.py` — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/model_operation.py)
  - `training_fcnn_model_with_semi_gradient(...)` at line 15 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/model_operation.py#L15)
  - `training_fcnn_model_with_true_gradient(...)` at line 115 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/model_operation.py#L115)
  - Typical edits: change batching, loss computation, or optimizer scheduling here.

5) Loss-computation helpers
-- File: `agent/model_compute_loss.py` — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/model_compute_loss.py)
  - `loss_with_semi_gradient(...)` at line 11 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/model_compute_loss.py#L11)
  - `loss_with_true_gradient(...)` at line 100 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/model_compute_loss.py#L100)
  - Typical edits: return type formatting, metrics collection, or different loss definitions.

6) Replay buffer
  - `insert_data_tuple(...)` at line 13
  - `insert_data_tuple_list(...)` at line 29
  - `get_sequential_batch_data(...)` at line 34
  - `get_shuffle_batch_data(...)` at line 44
-- File: `agent/replay_buffer.py` — `class SimpleReplayBuffer` at line 7 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/agent/replay_buffer.py#L7)
  - Typical edits: change sampling scheme, add prioritized replay, or adjust stored tuple format.

7) Environment (deterministic 3-state MDP)
-- File: `rl_enviroment/env_simple.py` — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/rl_enviroment/env_simple.py)
  - `def render(self, mode='human', close=False, policy_list=None):` at line 222 — drawing/rendering code (pygame + matplotlib fallback logic lives in visualizer). [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/rl_enviroment/env_simple.py#L222)
  - `def get_state_action_next_state_tuple(self):` at line 330 — returns the (s, a, s', r, done)-style tuples used to populate the replay buffer. [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/rl_enviroment/env_simple.py#L330)
  - Typical edits: change transition rules in `_transition_rule`, reward logic in `_reward_rule`, or representation generation.

8) Visualization helpers (save PNGs to `img/`)
-- File: `visialization/plot_accumulate_reward.py` — `plt_accumulate_reward(...)` at line 11 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/visialization/plot_accumulate_reward.py#L11)
-- File: `visialization/plot_loss.py` — `plot_loss(...)` at line 13 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/visialization/plot_loss.py#L13)
-- File: `visialization/plot_policies.py` — `plot_policy_in_env(...)` at line 10 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/visialization/plot_policies.py#L10)
-- File: `visialization/plot_value_function.py` — `plot_current_value_function(...)` at line 18 and `plot_state_action_value_function(...)` at line 37 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/visialization/plot_value_function.py#L18)
-- File: `visialization/plot_value_change.py` — `plot_value_change(...)` at line 41 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/visialization/plot_value_change.py#L41)
  - Typical edits: change plot styling, add annotations, or swap saving format (all current plots write PNGs to `img/` — change `IMG_DIR` in `config.py` or the `OUT_DIR` constant in each module if needed).

9) Shared configuration
-- File: `config.py` — `DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")` at line 13 — [open on GitHub](https://github.com/Emomeow/Undergraduate-Thesis/blob/main/config.py#L13)
  - Purpose: central place for `DEVICE` and `IMG_DIR` constants; import `config.DEVICE` in new modules for consistent device handling.

10) Quick dev tips
- To change default hyperparameters used by the command-line runner edit the defaults in the `if __name__ == "__main__":` block inside `main_func.py` (arguments and default values start near the bottom of that file).
- When editing model/training code, always ensure tensors and model are on the same device — prefer `from config import DEVICE` and `tensor.to(DEVICE)`.
- When you change visualization filenames, check `img/` after a short run to confirm plots are being overwritten as expected.

---

If you'd like, I can:
- add quick navigation links (VS Code-friendly) — done in this file.
- create a small `.vscode/tasks.json` with shortcuts to open these locations from the VS Code Tasks menu (I added this to the workspace — see `.vscode/tasks.json`).

File generated automatically on: 2025-11-08
