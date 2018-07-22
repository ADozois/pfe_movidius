FROM ros:kinetic


RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install lightdm -y
RUN apt-get install -y  libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
RUN apt-get install -y --no-install-recommends libboost-all-dev
RUN apt-get install -y bash git make wget sudo python3-pip libopencv-dev python-opencv cmake
RUN apt-get install cython3
RUN apt-get install python-matplotlib python-numpy python-pil python-scipy
RUN apt-get install build-essential cython
RUN apt-get install python-skimage
RUN pip3 install -U pip
RUN python3 -m pip install tensorflow==1.6.0

RUN mkdir -p /workspace
WORKDIR /workspace
RUN git clone https://github.com/movidius/ncsdk.git
WORKDIR /workspace/ncsdk
RUN make install

CMD bash