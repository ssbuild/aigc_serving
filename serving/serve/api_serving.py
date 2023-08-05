# -*- coding: utf-8 -*-
# @Time:  23:28
# @Author: tk
# @File：serving
import os
import multiprocessing
import shutil
from serving.workers import llm_worker
from ipc_worker.ipc_zmq_loader import IPC_zmq, ZMQ_process_worker # noqa
from config.main import global_models_info_args
from serving.utils import logger

class WokerLoader:
    def __init__(self,queue_mapper):
        self.queue_mapper = queue_mapper
        self.evt_quit = multiprocessing.Manager().Event()
        self.process_list = []

    def create(self):
        logger.info('WokerLoader create...')
        queue_mapper = self.queue_mapper
        process_list = self.process_list
        for model_name, config in global_models_info_args.items():
            if not config["enable"]:
                continue
            group_name = 'ai_group_{}'.format(model_name)
            # group_name
            # manager is an agent  and act as a load balancing
            # worker is real doing your work
            instance = IPC_zmq(
                CLS_worker=llm_worker.My_worker,
                worker_args=(model_name, config,),  # must be tuple
                worker_num=len(config['workers']),  # number of worker Process  大模型 建议使用1个 worker
                group_name=group_name,  # share memory name
                evt_quit=self.evt_quit,
                queue_size=20,  # recv queue size
                is_log_time=True,  # whether log compute time
            )
            process_list.append(instance)
            queue_mapper[model_name] = instance
            instance.start()

    def release(self):
        logger.info('WokerLoader release ...')
        try:
            self.evt_quit.set()
            for p in self.process_list:
                p.terminate()
            del self.evt_quit
        except Exception as e:  # noqa
            print(e)
        logger.info('WokerLoader release end')
