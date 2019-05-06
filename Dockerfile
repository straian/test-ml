FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y pciutils

RUN pip install cython
RUN pip install pycocotools
RUN pip install scikit-image
RUN pip install matplotlib

COPY ml ml

