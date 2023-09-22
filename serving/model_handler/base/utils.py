# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/22 14:55
from transformers import PretrainedConfig
def is_quantization_bnb(config: PretrainedConfig):
    quantization_config = getattr(config,"quantization_config",None)
    if not quantization_config:
        return False
    if isinstance(quantization_config,dict):
        is_quart = quantization_config.get("load_in_8bit",False) or quantization_config.get("load_in_4bit",False)
    else:
        is_quart = quantization_config.load_in_8bit or  quantization_config.load_in_4bit

    return is_quart