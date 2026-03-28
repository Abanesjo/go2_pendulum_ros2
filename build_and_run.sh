docker build -t unitree_ros2:v1 .
xhost +local:docker
docker run -it --rm --name unitree_ros2 --network host \
 --gpus all \
 --runtime=nvidia \
 -e DISPLAY=$DISPLAY \
 -e NVIDIA_DRIVER_CAPABILITIES=all \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v "$(dirname "$(readlink -f "$0")"):/workspace/src/unitree_ros2" \
 unitree_ros2:v1
