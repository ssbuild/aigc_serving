# -*- coding: utf-8 -*-
# @Time:  18:46
# @Author: tk
# @File：constant_map
from config.utils.env_check import check_config
from config.baichuan_conf import baichuan_config
from config.baichuan2_conf import baichuan2_config
from config.bloom_conf import bloom_conf
from config.chatglm_conf import chatglm_conf
from config.chatglm2_conf import chatglm2_conf
from config.internlm_conf import internlm_conf
from config.llama_conf import llama_conf
from config.moss_conf import moss_conf
from config.opt_conf import opt_conf
from config.rwkv_conf import rwkv_conf
from config.qwen_conf import qwen_conf


__all__ = [
    'global_models_info_args',
    'global_serve_args'
]

# 资源充足可以全部启用 , 并导入 global_models_info_args
global_models_info_args = {
    # **baichuan_config,
    # **baichuan2_config,
    # **bloom_conf,
    # **chatglm_conf,
    # **chatglm2_conf,
    # **internlm_conf,
    # **llama_conf,
    # **moss_conf,
    # **opt_conf,
    # **rwkv_conf,
     **qwen_conf,

}

global_serve_args = {
    "host": '0.0.0.0',
    "port":8081,
    "workers":4
}


check_config(global_models_info_args)




