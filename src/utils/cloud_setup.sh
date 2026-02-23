#!/usr/bin/env bash

set -e  # 에러 발생 시 즉시 종료

echo "===== RunPod setup start ====="
cd /workspace

# --------------------------------------------------
# repo clone
# --------------------------------------------------
if [ ! -d "CTMP_GIN" ]; then
    git clone https://github.com/loveiscomplicated/CTMP_GIN.git
fi

cd CTMP_GIN

# --------------------------------------------------
# install miniconda 
# --------------------------------------------------
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

source $HOME/miniconda3/etc/profile.d/conda.sh

# --------------------------------------------------
# create conda env
# --------------------------------------------------
if ! conda env list | grep -q pyg_2; then
    conda create -y -n pyg_2 python=3.12
fi

conda activate pyg_2

# --------------------------------------------------
# python packages
# --------------------------------------------------
pip install --upgrade pip
pip install torch torchvision
pip install torch_geometric
pip install -r requirements.txt
pip install requests

# --------------------------------------------------
# install tmux + rclone
# --------------------------------------------------
sudo apt update
sudo apt install -y tmux rclone

# --------------------------------------------------
# data download
# --------------------------------------------------
cd src/data/raw

if [ ! -f "dataset_downloaded.flag" ]; then
    pip install gdown
    gdown 1T1oYAsdYDcdqUckd7CBzBWj9RnwGrEZg
    touch dataset_downloaded.flag
fi

cd /workspace/CTMP_GIN

echo "===== RunPod setup complete ====="