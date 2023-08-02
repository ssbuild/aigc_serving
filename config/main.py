# -*- coding: utf-8 -*-
# @Time:  18:46
# @Author: tk
# @File：constant_map
import socket
from config.baichuan_conf import baichuan_config
from config.bloom_conf import bloom_conf
from config.chatglm_conf import chatglm_conf
from config.internlm_conf import internlm_conf
from config.llama_conf import llama_conf
from config.moss_conf import moss_conf
from config.opt_conf import opt_conf
from config.rwkv_conf import rwkv_conf


__all__ = [
    'global_models_info_args'
]

# 资源充足可以全部启用 , 并导入 global_models_info_args
global_models_info_args = {
    # **baichuan_config,
    # **bloom_conf,
    **chatglm_conf,
    # **internlm_conf,
    # **llama_conf,
    # **moss_conf,
    # **opt_conf,
    # **rwkv_conf,

}


def get_free_tcp_port():
    """获取可用的端口"""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port



_is_check_deepseed = False
for model_name,model_config in global_models_info_args.items():
    if not model_config['enable']:
        continue

    if model_config['work_mode'] != 'deepspeed':
        continue
    workers = model_config['workers']
    flag = False
    for worker in workers:
        if len(worker['device_id']) > 1:
            flag = True
            break
    model_config['deepspeed'] = {}
    conf = model_config['deepspeed']
    if flag:
        _is_check_deepseed = True
        port = get_free_tcp_port()
        conf["MASTER_ADDR"] = "127.0.0.1"
        conf["MASTER_PORT"] = str(port)
        conf["TORCH_CPP_LOG_LEVEL"] = "INFO"

