# -*- coding: utf-8 -*-
# @Time:  18:46
# @Author: tk
# @File：constant_map
import os
from config.pyconfig import load_config as load_config_python
from config.yamlconfig import load_config as load_config_yaml



__all__ = [
    'global_models_info_args',
    'global_serve_args'
]

from config.utils.env_check import check_config

global_serve_args = {
    "host": '0.0.0.0',
    "port": 8081,
    "workers": 4
}

# 资源充足可以全部启用 , 并导入 global_models_info_args
global_models_info_args = {

}
#从pyconfig 导入model config
global_models_info_args.update(load_config_python())

#从 yamlconfig 导入model config
global_models_info_args.update(load_config_yaml())


#从自定义的路径 导入model config
AS_CONFIG_PATH = os.environ.get("AS_CONFIG_PATH",None)
if AS_CONFIG_PATH is not None:
    global_models_info_args.update(load_config_yaml(AS_CONFIG_PATH))

check_config(global_models_info_args)
