#FROM nvcr.io/nvidia/pytorch:22.12-py3
FROM xbuild/cuda:11.8.0-cudnn870-devel-ubuntu20.04
#FROM xbuild/cuda:11.8.0-cudnn870-devel-ubuntu22.04
env DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl git && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

WORKDIR /aigc_serving/
ENV PYTHONPATH /aigc_serving/
COPY aigc_serving /aigc_serving


RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r /aigc_serving/requirements.txt -i https://mirror.baidu.com/pypi/simple && \
    pip3 install bitsandbytes --upgrade -i https://mirror.baidu.com/pypi/simple