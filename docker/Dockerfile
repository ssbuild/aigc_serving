#FROM nvcr.io/nvidia/pytorch:22.12-py3
FROM xbuild/cuda:11.8.0-cudnn870-devel-ubuntu20.04
#FROM xbuild/cuda:11.8.0-cudnn870-devel-ubuntu22.04

WORKDIR /as/
ENV PYTHONPATH /as/
COPY requirements.txt /as/
COPY . /as


RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r /as/requirements.txt -i https://mirror.baidu.com/pypi/simple
RUN pip3 install bitsandbytes --upgrade -i https://mirror.baidu.com/pypi/simple