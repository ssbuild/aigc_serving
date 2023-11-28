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



def is_quantization_awq(config: PretrainedConfig):
    quantization_config = getattr(config, "quantization_config", None)
    if not quantization_config:
        return False

    if isinstance(quantization_config, dict):
        bits = quantization_config.get("bits", 0)
        is_quart = quantization_config.get("quant_method", "") == "awq" and bits > 0 and bits < 16
    else:
        is_quart = quantization_config.quant_method == "awq" and quantization_config.bits > 0 and quantization_config.bits < 16

    return is_quart