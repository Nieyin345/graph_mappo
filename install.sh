#!/usr/bin/env bash
# CloudLab setup script for the QKD-SAGIN RL project.
set -euxo pipefail

REPO_DIR="${REPO_DIR:-/local/repository}"
DATASET_DIR="$REPO_DIR/dataset/global"
DATASET_URL="${DATASET_URL:-}"

apt-get update
apt-get install -y git curl ca-certificates

# Miniconda
if [ ! -d /opt/conda ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda create -y -n pytorch python=3.10
conda activate pytorch

# GPU nodes install CUDA-enabled PyTorch; CPU-only nodes stay lean.
if command -v nvidia-smi >/dev/null 2>&1; then
  conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
else
  conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
fi

pip install -e "$REPO_DIR"

# The H5 dataset is intentionally not tracked by Git. Either set DATASET_URL
# in the CloudLab profile parameters before instantiation, or upload the
# files manually after the node boots.
mkdir -p "$DATASET_DIR"
if [ -n "$DATASET_URL" ]; then
  curl -fsSL "$DATASET_URL" -o "$DATASET_DIR/link_data.h5"
fi
if [ ! -f "$DATASET_DIR/link_data.h5" ]; then
  echo "WARNING: dataset/global/link_data.h5 not found."
  echo "Upload it to $DATASET_DIR after boot, or set DATASET_URL."
fi

echo "Setup complete."
echo "Run training with: conda run -n pytorch python $REPO_DIR/scripts/train_graph_mappo.py --mode random_episode"
