"""
Global configuration constants used across the project.

This file provides a single source of truth for device selection so other
modules can safely import `DEVICE` from `config`.

Keep this file minimal to avoid heavy imports at module-import time.
"""
import torch

# detect whether CUDA is available; use cpu otherwise
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

# Default directory for saved visualization outputs (relative to project root)
IMG_DIR = "img"
