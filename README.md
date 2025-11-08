# Q-learning
# Offline RL: FCNN-based MDP experiment

This repository implements a compact offline reinforcement-learning experiment using a fully-connected neural network (FCNN) to represent value / Q functions on a tiny deterministic MDP. The project includes a simple environment, an agent, replay buffer, training/loss routines, and visualization scripts that save PNGs into `img/`.

## Goals
- Provide a small reproducible example for semi-gradient and true-gradient TD learning with a neural function approximator.
- Keep the dataflow explicit and easy to inspect: environment -> replay buffer -> training -> visualize.

## Quick facts
- Main entrypoint: `main_func.py` (invokes training and saves visualizations to `img/`)
- Environment: `rl_enviroment/DeterministicMDPSimple` (3-state deterministic MDP)
- Agent: `agent/MDPAgent` using `agent/FCNN` (`agent/nn_model.py`)
- Replay buffer: `agent/SimpleReplayBuffer` (`agent/replay_buffer.py`)
- Training & loss: `agent/model_operation.py`, `agent/model_compute_loss.py`
- Visualizations: PNG files saved under `img/` by modules in `visialization/`

## Project layout
```
config.py                # global constants (DEVICE, IMG_DIR)
main_func.py             # top-level training / runner
agent/                   # agent implementation + model / replay buffer
	agent.py
	nn_model.py
	model_operation.py
	model_compute_loss.py
	replay_buffer.py
rl_enviroment/           # simple deterministic MDP environment
	env_simple.py
visialization/           # plotting helpers (save PNGs into img/)
	plot_accumulate_reward.py
	plot_loss.py
	plot_policies.py
	plot_value_change.py
	plot_value_function.py
img/                     # generated PNG visualizations (output)
model/                   # saved model checkpoints
tests/                   # quick smoke tests
README.md
```

## Per-file explanations

Below are short notes and the primary entry points to edit when you work on a specific aspect of the project. These are the same per-file pointers found in `DEVELOPER_CHEATSHEET.md` and are intended to help you quickly find where to make changes.

1) Top-level runner
- File: `main_func.py` — function `main_func(...)`
	- Purpose: main training loop, argument parsing, instantiates `DeterministicMDPSimple` and `MDPAgent`, runs epochs and calls all visualization helpers.
	- Key plotting calls (near the end of `main_func`):
		- `plt_accumulate_reward(...)`
		- `plot_loss(...)`
		- `plot_policy_in_env(...)`
		- `plot_current_value_function(...)`
		- `plot_state_action_value_function(...)`
	- Typical edits: experiment orchestration (epoch count, LR schedule, which plots to produce), and data collection for plotting.

2) Agent and training
- File: `agent/agent.py` — class `MDPAgent`
	- `get_current_policy_and_value(...)` — computes policy & value from the model given state representations.
	- `offline_learning(...)` — the per-epoch training loop that samples the replay buffer and calls training/loss functions.
	- Typical edits: change how the model is called, alter training loops or logging frequency, add hooks for target networks or metrics.

3) Network definition
- File: `agent/nn_model.py` — class `FCNN`
	- `get_initialization(...)` — directly sets layer weights from numpy arrays.
	- `forward(...)` — feed-forward; modify this to change architecture, layers, or activations.

4) Training routines (semi / true gradient)
- File: `agent/model_operation.py`
	- `training_fcnn_model_with_semi_gradient(...)`
	- `training_fcnn_model_with_true_gradient(...)`
	- Typical edits: change batch organization, loss usage, or optimization steps here.

5) Loss-computation helpers
- File: `agent/model_compute_loss.py`
	- `loss_with_semi_gradient(...)`
	- `loss_with_true_gradient(...)`
	- Typical edits: return formatting, metric collection, or alternative loss definitions.

6) Replay buffer
- File: `agent/replay_buffer.py` — class `SimpleReplayBuffer`
	- Methods to note: `insert_data_tuple(...)`, `insert_data_tuple_list(...)`, `get_sequential_batch_data(...)`, `get_shuffle_batch_data(...)`.
	- Typical edits: change sampling strategy, add prioritization, or adapt stored tuple content.

7) Environment (deterministic 3-state MDP)
- File: `rl_enviroment/env_simple.py`
	- `render(self, mode='human', close=False, policy_list=None)` — pygame rendering and RGB-array return.
	- `get_state_action_next_state_tuple(...)` — returns the (s, a, s', r, done)-style tuples used to populate the replay buffer.
	- Typical edits: modify `_transition_rule`, `_reward_rule`, or the state representation generation.

8) Visualization helpers (save PNGs to `img/`)
- Files: `visialization/plot_accumulate_reward.py`, `visialization/plot_loss.py`, `visialization/plot_policies.py`, `visialization/plot_value_function.py`, `visialization/plot_value_change.py`
	- Typical edits: styling, additional annotations, or alternate output paths (plots currently save PNGs into `img/`).

9) Shared configuration
- File: `config.py` — contains `DEVICE` and `IMG_DIR` configuration constants.
	- Purpose: centralize device selection and image output directory; prefer importing `DEVICE` from `config` when adding new modules.

10) Quick dev tips
- Change default CLI hyperparameters in the `if __name__ == "__main__":` block of `main_func.py`.
- When editing model/training code, ensure tensors and model live on the same device (use `config.DEVICE`).


## Requirements
- Python 3.8+ recommended.
- Core Python packages: numpy, torch, matplotlib, seaborn, pandas, opencv-python, pillow, pygame, gymnasium, pymdptoolbox.

Example install (PowerShell):
```powershell
pip install numpy matplotlib seaborn pandas opencv-python pillow pygame gymnasium pymdptoolbox
# Install torch according to your CUDA version from https://pytorch.org/get-started/locally/
```

## Configuration
- `config.py` exposes `DEVICE` (cpu or cuda:0) and `IMG_DIR` (default `img`). The code detects CUDA automatically.
- Visualization modules save PNGs into `img/`. `plot_policies.py` contains a matplotlib fallback so a policy PNG is produced even on headless systems.

## How to run
- Quick demo (short run, produces PNGs in `img/`):

```powershell
python main_func.py --epoch-num 10
```

- Full training (default 5000 epochs):

```powershell
python main_func.py
```

Run `python main_func.py -h` to see all arguments (learning rate, optimizer, gamma, etc.).

## Output files
- PNG visualizations (saved into `img/`):
	- `{gradient}_accumulated_reward.png`
	- `{gradient}_loss.png` and `_loss_log_scale.png`
	- `{gradient}_policy.png`
	- `{gradient}_value_function_for_each_state.png`
	- `{gradient}_value_function_for_each_state_action.png` (and subplot version)
- Model checkpoints in `model/` as `fcnn_{gradient}.pl`.

## Troubleshooting & notes
- If you see device mismatch errors, ensure PyTorch/CUDA are installed correctly. `config.py` centralizes `DEVICE` detection.
- The plotting functions close matplotlib figures to be headless-friendly. If you prefer interactive plots, call `plt.show()` in the plotting module or run locally in a Jupyter environment.
- Old CSV debug files may exist from earlier runs; the current code writes PNGs only.

## Testing
- Small smoke tests live in `tests/`. Run with:

```powershell
pytest -q
```

## Development notes
- Replay buffer returns a list of tuples `(s_t, a_t, s_{t+1}, r_t)`.
- Training code converts numpy arrays to PyTorch tensors and moves them to `DEVICE`.


