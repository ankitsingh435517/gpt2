import subprocess
import sys
from pathlib import Path
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_checkpoints",
    action="store_true",
    help="Load an existing checkpoint",
)

args = parser.parse_args()

# Paths
volume_path = Path("/mnt/gpt2-checkpoints")
checkpoint_path = volume_path / "checkpoints"
src = Path("/mnt/gpt2/gpt2.pt")
dst = checkpoint_path / "gpt2.pt"

# Validate checkpoint volume
if not volume_path.is_dir():
    raise FileNotFoundError(
        f"Checkpoint volume not found at {volume_path}. "
        "Make sure the Modal volume is mounted when starting the shell."
    )

# Create checkpoints directory if it doesn't exist
if not checkpoint_path.exists():
    checkpoint_path.mkdir()
    print(f"Created directory: {checkpoint_path}")

# Handle checkpoint
if args.load_checkpoints:
    if dst.exists():
        print(f"Checkpoint already exists: {dst}. Skipping move.")
    elif src.exists():
        shutil.move(src, dst)
        print(f"Moved checkpoint: {src} -> {dst}")
    else:
        raise FileNotFoundError(
            f"No checkpoint found at {src} to move."
        )

# Change directory to GPT-2 project
gpt2_path = Path("/mnt/gpt2")

if not gpt2_path.is_dir():
    raise FileNotFoundError(f"GPT-2 directory not found: {gpt2_path}")

os.chdir(gpt2_path)

# Install dependencies
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    check=True,
)

# Prepare HellaSwag
subprocess.run(
    [sys.executable, "prepare_hellaswag.py"],
    check=True,
)