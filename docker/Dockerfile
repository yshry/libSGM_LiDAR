FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update && apt -y upgrade
RUN apt install -y cmake libopencv-dev zstd
RUN apt install -y wget
RUN apt install -y cmake-curses-gui

RUN apt install -y vim

RUN version=3.18 && build=4 \
 && mkdir ~/temp-cmake \
 && cd ~/temp-cmake \
 && wget https://cmake.org/files/v$version/cmake-$version.$build-Linux-x86_64.sh \
 && mkdir /opt/cmake \
 && sh cmake-$version.$build-Linux-x86_64.sh --prefix=/opt/cmake --skip-license \
 && for filename in /opt/cmake/bin/*; do echo Registering $filename; ln -fs $filename /usr/local/bin/`basename $filename`; done \
 && rm cmake-$version.$build-Linux-x86_64.sh

RUN apt install -y python3 python3-pip python-is-python3
RUN pip install pykitti opencv-python
RUN pip install scipy

RUN apt install -y libpng-dev zlib1g-dev
RUN apt install -y libpng++-dev
RUN apt install -y libboost-filesystem-dev libboost-program-options-dev

CMD bash
