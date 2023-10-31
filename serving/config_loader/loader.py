# -*- coding: utf-8 -*-
# @Time:  18:46
# @File：loader

import os
from serving.config_loader.utils.env_check import *

__all__ = [
    'global_models_info_args',
    'global_serve_args'
]

#优先从环境变量导入
AS_CONFIG_PATH = os.environ.get("AS_CONFIG_PATH",None)
if AS_CONFIG_PATH is None:
    # 从本地文件导入配置
    AS_CONFIG_PATH = os.path.join(os.path.join(os.path.dirname(__file__),'../../config/config.yaml'))

assert os.path.exists(AS_CONFIG_PATH)

global_serve_args,global_models_info_args = load_config_from_yaml(AS_CONFIG_PATH)

assert global_serve_args is not None and global_models_info_args is not None

check_config(global_models_info_args)


