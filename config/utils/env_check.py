# -*- coding: utf-8 -*-
# @Time:  14:13
# @Author: tk
# @File：env_check
import socket

__all__ = [
    'check_config'
]

def get_free_tcp_port():
    """获取可用的端口"""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def check_config(global_models_info_args):

    for model_name,model_config in global_models_info_args.items():
        if not model_config['enable']:
            continue

        if model_config['work_mode'] != 'deepspeed':
            continue
        workers = model_config['workers']

        for worker in workers:
            worker['deepspeed'] = {}
            conf = worker['deepspeed']
            if len(worker['device_id']) > 1:
                port = get_free_tcp_port()
                conf["MASTER_ADDR"] = "127.0.0.1"
                conf["MASTER_PORT"] = str(port)
                conf["TORCH_CPP_LOG_LEVEL"] = "INFO"
                conf["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
                conf["TORCH_SHOW_CPP_STACKTRACES"] = "1"