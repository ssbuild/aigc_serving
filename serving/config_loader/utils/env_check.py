# -*- coding: utf-8 -*-
# @Time:  14:13
# @Author: tk
# @File：env_check
import os
import socket

__all__ = [
    "load_config_from_yaml",
    "check_config",
]

import yaml


def get_free_tcp_port():
    """获取可用的端口"""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def check_config(global_models_info_args):

    name_set = set()
    for model_name,model_config in global_models_info_args.items():
        if not model_config['enable']:
            continue
        alias = model_config.get('alias')
        if alias is not None:
            if isinstance(alias,str):
                alias = [alias]
            assert isinstance(alias,list)
            for name in alias:
                assert name is not None and isinstance(name,str)
                if name in name_set:
                    raise ValueError("{} exists".format(name))
                name_set.add(name)

        if model_config['work_mode'] != 'deepspeed':
            continue
        workers = model_config['workers']

        for worker in workers:
            worker['deepspeed'] = {}
            conf = worker['deepspeed']
            if len(worker['device_id']) >= 1:
                port = get_free_tcp_port()
                conf["MASTER_ADDR"] = "127.0.0.1"
                conf["MASTER_PORT"] = str(port)
                conf["TORCH_CPP_LOG_LEVEL"] = "INFO"
                conf["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
                conf["TORCH_SHOW_CPP_STACKTRACES"] = "1"


def load_config_from_yaml(path_file):
    serve_args = None
    model_config = {}
    if os.path.isfile(path_file):
        path_file = [path_file]
    else:
        fs = os.listdir(path_file)
        path_file = [os.path.join(path_file,_) for _ in fs if _.lower().endswith('.yaml')]

    for file in path_file:
        with open(file,mode='r',encoding='utf-8') as f:
            c = yaml.full_load(f)
        serve_args = serve_args or c.pop("serve_args",None)
        for model_name,conf in c.items():
            if not isinstance(conf,dict):
                continue
            if 'model_config' not in conf:
                # invalid config
                continue
            if not conf.get('enable',False):
                continue
            model_config[model_name] = conf

    return serve_args,model_config