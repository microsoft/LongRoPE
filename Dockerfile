FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app

COPY ./requirements.txt /install/requirements.txt

RUN nvcc --version

RUN cd /install && pip install -r requirements.txt

# RUN rm -rf /install

RUN apt-get update
RUN apt-get install git
RUN apt-get install vim -y
RUN apt-get install git -y
