# -*- coding: utf-8 -*-
# @Time:  23:28
# @Author: tk
# @File：serving

import os
import multiprocessing

from serving.serve.http_serving_openai import HTTP_Serving
from serving.workers import llm_worker
from ipc_worker.ipc_zmq_loader import IPC_zmq, ZMQ_process_worker # noqa
from config.main import global_models_info_args

def main():
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    os.environ['ZEROMQ_SOCK_TMP_DIR'] = tmp_dir

    evt_quit = multiprocessing.Manager().Event()
    queue_mapper = {}
    process_list = []

    for model_name, config in global_models_info_args.items():
        if not config["enable"]:
            continue
        group_name = 'serving_group_{}_1'.format(model_name)
        # group_name
        # manager is an agent  and act as a load balancing
        # worker is real doing your work
        instance = IPC_zmq(
            CLS_worker=llm_worker.My_worker,
            worker_args=(model_name, config,),  # must be tuple
            worker_num=len(config['workers']),  # number of worker Process  大模型 建议使用1个 worker
            group_name=group_name,  # share memory name
            evt_quit=evt_quit,
            queue_size=20,  # recv queue size
            is_log_time=True,  # whether log compute time
        )
        process_list.append(instance)
        queue_mapper[model_name] = instance
        instance.start()

    http_ = HTTP_Serving(queue_mapper,
                         http_ip='0.0.0.0',
                         http_port=8081, )
    http_.start()
    process_list.append(http_)
    try:
        for p in process_list:
            p.join()
    except Exception as e: # noqa
        evt_quit.set()
        for p in process_list:
            p.terminate()
    del evt_quit