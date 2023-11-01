#!/usr/bin/env bash


base_path=$(cd `dirname $0`; pwd)

# 配置文件路径
export AS_CONFIG_PATH="${base_path}/config.yaml"

python3 ${base_path}/../serving/main.py