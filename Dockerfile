FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y pciutils

RUN pip install matplotlib

# TODO: Preload data in image.

COPY ml ml

