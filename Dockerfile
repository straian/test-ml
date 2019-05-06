FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y pciutils

#COPY download-datasets.sh download-datasets.sh
#RUN bash download-datasets.sh

COPY datasets datasets
#RUN mkdir /root/.keras
#COPY datasets /root/.keras/datasets

RUN pip install cython
RUN pip install pycocotools
RUN pip install scikit-image
RUN pip install matplotlib

COPY ml ml

