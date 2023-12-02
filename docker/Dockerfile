#FROM nvcr.io/nvidia/pytorch:22.12-py3
#FROM xbuild/cuda:11.8.0-cudnn870-cudnn8-devel-ubuntu22.04
#FROM xbuild/cuda:11.8.0-cudnn870-cudnn8-devel-ubuntu22.04
#FROM xbuild/cuda:11.8.0-cudnn8-devel-ubuntu20.04
#FROM xbuild/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 安装当前最新版 torch 2.1
FROM xbuild/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential python3.10 python3.10-distutils curl git vim &&  \
    rm -rf /var/lib/apt/lists/* &&  \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&  \
    python3.10 get-pip.py

RUN ln -sf /usr/bin/python3 /usr/bin/python && /usr/bin/pip3 /usr/bin/pip

ENV LANG=C.UTF-8
# torch 2.1
RUN pip3 install torch torchvision torchaudio

WORKDIR /aigc_serving/
ENV PYTHONPATH /aigc_serving/
COPY aigc_serving /aigc_serving
RUN pip3 install --no-cache-dir -r /aigc_serving/requirements.txt -i https://mirror.baidu.com/pypi/simple