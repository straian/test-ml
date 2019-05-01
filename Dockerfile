FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y pciutils

RUN pip install matplotlib

RUN mkdir /root/.keras
COPY datasets /root/.keras/datasets

COPY ml ml

RUN pip install cython
RUN pip install pycocotools
RUN pip3 install scikit-image

