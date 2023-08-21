# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/21 10:21
import os
import yaml


def load_config(path_dir = None):
    if path_dir is None:
        path_dir = os.path.abspath(os.path.dirname(__file__))

    yaml_config = {}
    fs = os.listdir(path_dir)
    fs = [os.path.join(path_dir,_) for _ in fs if _.lower().endswith('.yaml')]
    for file in fs:
        with open(file,mode='r',encoding='utf-8') as f:
            c = yaml.full_load(f)
        for conf in c.values():
            if not isinstance(conf,dict):
                continue
            if 'model_config' not in conf:
                # invalid config
                continue
            if not conf.get('enable',False):
                continue
            yaml_config.update(c)

    return yaml_config