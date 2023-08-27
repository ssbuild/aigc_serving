# -*- coding: utf-8 -*-
# @Time:  18:46
# @Author: tk
# @File：main
import copy
import os
from config.model_config import model_config
from config.serve_config import serve_config
from serving.config_parser.utils.env_check import check_config, load_config_yaml

__all__ = [
    'global_models_info_args',
    'global_serve_args'
]


global_serve_args = copy.deepcopy(serve_config)

# 所有模型配置
global_models_info_args = {
    **model_config
}

#从自定义的路径 导入model config
AS_CONFIG_PATH = os.environ.get("AS_CONFIG_PATH",None)
if AS_CONFIG_PATH is not None:
    global_models_info_args.update(load_config_yaml(AS_CONFIG_PATH))

check_config(global_models_info_args)
