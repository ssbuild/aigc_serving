# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 13:33
# @Author  : tk
import os
import numpy as np
from ipc_worker.ipc_zmq_loader import IPC_zmq,ZMQ_process_worker
import copy


class My_worker(ZMQ_process_worker):
    def __init__(self,config,device_num,*args,**kwargs):
        super(My_worker,self).__init__(*args,**kwargs)
        #config info , use by yourself
        self._logger.info('Process id {}, group name {} , identity {}'.format(self._idx,self._group_name,self._identity))
        # config['model_config']['device_id'] = self._idx % device_num if device_num > 0 else 0
        self._logger.info(config)
        self.config = copy.deepcopy(config)

    #Process begin trigger this func
    def run_begin(self):
        self._logger.info('worker pid {}...'.format(os.getpid()))

        print('process started....')

    # Process end trigger this func
    def run_end(self):
        print('process ended....')

    #any data put will trigger this func
    def run_once(self,request_data):
        print('request data',request_data)
        return request_data