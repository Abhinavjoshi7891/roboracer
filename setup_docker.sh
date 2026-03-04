#!/bin/bash
sudo systemctl stop docker
sudo systemctl stop docker.socket
sudo mkdir -p /home/abhinav/Data_Drive/roboracer/docker_data
sudo mkdir -p /etc/docker
sudo mv /home/abhinav/Data_Drive/roboracer/daemon.json /etc/docker/daemon.json
sudo systemctl start docker
