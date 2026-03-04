#!/bin/bash
set -e
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
python3 /home/abhinav/Data_Drive/roboracer/update_repo.py
sudo mv /home/abhinav/Data_Drive/roboracer/nvidia-container-toolkit.list /etc/apt/sources.list.d/
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
