#!/bin/bash

echo "========================================="
echo " Starting AutoDRIVE Gap Follower V2"
echo "========================================="

# Stop any accidentally running background container
docker stop autodrive_devkit 2>/dev/null
docker rm -f autodrive_devkit 2>/dev/null

echo "Booting Docker Container, Compiling, and Launching ROS 2 Nodes..."

docker run --name autodrive_devkit --rm -it \
  --network=host --ipc=host \
  -v /home/abhinav/Data_Drive/roboracer/autodrive_devkit_workspace:/home/autodrive_devkit/src/autodrive_devkit \
  --entrypoint /bin/bash \
  autodriveecosystem/autodrive_roboracer_api:practice \
  -c "cd /home/autodrive_devkit && rm -rf build/autodrive_roboracer install/autodrive_roboracer && source /opt/ros/humble/setup.bash && colcon build --packages-select autodrive_roboracer --symlink-install && source install/setup.bash && ros2 launch autodrive_roboracer gap_follower.launch.py"
