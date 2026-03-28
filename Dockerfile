FROM osrf/ros:humble-desktop-full

RUN apt update && apt upgrade -y

# ROS2 and build deps
RUN apt install -y \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-rosbag2-cpp \
    libyaml-cpp-dev \
    libeigen3-dev \
    python3-pip \
    libboost-all-dev \
    libspdlog-dev \
    libfmt-dev

# Build & install CycloneDDS from source (needed by unitree_sdk2_python)
COPY dependencies/cyclonedds /tmp/cyclonedds
RUN cd /tmp/cyclonedds && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF && \
    cmake --build . --parallel $(nproc) --target install && \
    rm -rf /tmp/cyclonedds

# Build & install unitree_sdk2 (C++ SDK)
COPY dependencies/unitree_sdk2 /tmp/unitree_sdk2
RUN cd /tmp/unitree_sdk2 && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics -DBUILD_EXAMPLES=OFF && \
    cmake --build . --parallel $(nproc) --target install && \
    rm -rf /tmp/unitree_sdk2

# Install unitree_sdk2_python
COPY dependencies/unitree_sdk2_python /tmp/unitree_sdk2_python
RUN CYCLONEDDS_HOME=/opt/cyclonedds pip3 install /tmp/unitree_sdk2_python && \
    rm -rf /tmp/unitree_sdk2_python

ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV CMAKE_PREFIX_PATH=/opt/unitree_robotics:/opt/cyclonedds
ENV LD_LIBRARY_PATH=/opt/unitree_robotics/lib:/opt/cyclonedds/lib

# Workspace setup
WORKDIR /workspace

SHELL ["/bin/bash", "-c"]

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
