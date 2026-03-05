#!/bin/bash
echo "========================================="
echo " Starting LocalMapPP Racing Algorithm"
echo "========================================="
docker stop autodrive_devkit 2>/dev/null
docker rm -f autodrive_devkit 2>/dev/null

echo "Booting Docker Container..."
docker run --name autodrive_devkit --rm -it \
  --network=host --ipc=host \
  -v /home/abhinav/Data_Drive/roboracer/autodrive_devkit_workspace:/home/autodrive_devkit/src/autodrive_devkit \
  -v /home/abhinav/Data_Drive/roboracer/telemetry:/tmp/telemetry \
  --entrypoint /bin/bash \
  roboracer-localmap:latest \
  -c "cd /home/autodrive_devkit && \
      rm -rf build/autodrive_roboracer install/autodrive_roboracer && \
      source /opt/ros/humble/setup.bash && \
      colcon build --packages-select autodrive_roboracer --symlink-install && \
      source install/setup.bash && \
      ros2 launch autodrive_roboracer local_map_pp.launch.py; \
      cp /tmp/race_telemetry.csv /tmp/telemetry/race_telemetry.csv 2>/dev/null; \
      echo '=== Telemetry saved to telemetry/race_telemetry.csv ==='"
