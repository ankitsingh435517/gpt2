import subprocess
import sys
from pathlib import Path
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--nproc_node", type=int, default=2)
parser.add_argument(
    "--load_checkpoints", type=bool, default=False
)  # defaults to a fresh run

args = parser.parse_args()

checkpoint_path = "/mnt/gpt2-checkpoints/checkpoints"
# create a directory for checkpoints in the volume, ephemeral volumes
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# move the gpt2.pt into checkpoints if needed
if args.load_checkpoints:
    src = Path("/mnt/gpt2/gpt2.pt")
    dst = Path(checkpoint_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)

# change directory to gpt2 code
path = Path("mnt/gpt2")
os.chdir(path)

# execute
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
)
subprocess.run([sys.executable, "python", "prepare_hellaswag.py"], check=True)
subprocess.run(
    [
        sys.executable,
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_nodes}",
        "train_gpt2.py",
    ],
    check=True,
)
