cd workspace

git clone https://github.com/loveiscomplicated/CTMP_GIN.git
git checkout runpod
git pull origin runpod

cd CTMP_GIN

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/miniconda3/etc/profile.d/conda.sh

conda create -n pyg_2 python=3.12

conda activate pyg_2

pip3 install torch torchvision

pip install torch_geometric

pip install -r requirements.txt

pip install requests

# --------------------------------------------------
# install tmux + rclone
# --------------------------------------------------
apt update
apt install -y tmux
echo "set -g mouse on" >> ~/.tmux.conf && tmux source-file ~/.tmux.conf 
cd src/data/raw

gdown 1T1oYAsdYDcdqUckd7CBzBWj9RnwGrEZg

cd ..

cd ..

cd ..