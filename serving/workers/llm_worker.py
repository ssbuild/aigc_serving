# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 13:33
# @Author  : tk
import os
import sys
import time
import traceback
import typing
from ipc_worker.ipc_zmq_loader import IPC_zmq,ZMQ_process_worker  # noqa
import copy


def get_worker_instance(model_name,config,group_name,worker_idx):
    model_name: str = model_name.lower()
    if model_name.startswith("baichuan"):
        if model_name.find('13b') != -1:
            from serving.model_handler.baichuan2.infer import EngineAPI
            api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        else:
            from serving.model_handler.baichuan.infer import EngineAPI
            api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
            
    elif model_name.startswith("chatglm2"):
        from serving.model_handler.chatglm2.infer import EngineAPI
        api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        
    elif model_name.startswith("chatglm"):
        from serving.model_handler.chatglm.infer import EngineAPI
        api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        
    elif model_name.startswith("llama") or model_name.startswith("opt") or model_name.startswith("bloom"):
        from serving.model_handler.llm.infer import EngineAPI
        api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        
    elif model_name.startswith("internlm"):
        from serving.model_handler.internlm.infer import EngineAPI
        api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        
    elif model_name.startswith("moss"):
        from serving.model_handler.moss.infer import EngineAPI
        api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        
    elif model_name.startswith("rwkv"):
        from serving.model_handler.rwkv.infer import EngineAPI
        api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)

    else:
        raise ValueError('not support yet')
    return api_client

class My_worker(ZMQ_process_worker):
    def __init__(self,model_name,config,*args,**kwargs):
        super(My_worker,self).__init__(*args,**kwargs)
        self._logger.info('group name {} ,worker id {}'.format(self._group_name,self._idx))
        self._logger.info(config)
        self.config = copy.deepcopy(config)
        self.model_name = model_name
        self.api_client = None
        self.initial_error = None

    #Process begin trigger this func
    def run_begin(self):
        try:
            device_id = self.config['workers'][self._idx]['device_id']
            if device_id is not None:
                os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(_) for _ in device_id])
            self._logger.info('{} worker pid {}...'.format(self.model_name, os.getpid()))
            self.api_client = get_worker_instance(self.model_name, self.config,self._group_name, self._idx)
            self.api_client.init()
        except Exception as e:
            traceback.print_exc()
            self._logger.error(e)
            self.api_client = None
            self.initial_error = str(e)

    # Process end trigger this func
    def run_end(self):
        print(self.model_name,' process ended....')

    #any data put will trigger this func
    def run_once(self,request_data):
        r = request_data
        result = None
        start_time = time.time()
        try:
            if self.initial_error is None:
                method = r.get('method', "generate")
                if method == 'chat_stream':
                    gen = self.api_client.trigger_generator(r)
                    for node_result in gen:
                        result, code, msg, complte_flag = node_result
                        end_time = time.time()
                        ret = {
                            "code": code,
                            "runtime": (end_time - start_time) * 1000,
                            "msg": msg,
                            "complete": complte_flag
                        }
                        if code == 0:
                            if not isinstance(result, tuple):
                                ret["result"] = result
                            else:
                                ret["result"] = result[0]
                                ret["history"] = result[1]
                        yield ret

                    return None

                else:
                    result,code,msg,complte_flag = self.api_client.trigger(r)
            else:
                code = -1
                msg = self.initial_error
        except Exception as e:
            traceback.print_exc()
            code = -1
            msg = str(e)
            self._logger.info(e)
        end_time = time.time()

        ret = {
            "code": code,
            "runtime": (end_time - start_time) * 1000,
            "msg": msg,
            "complete": True
        }
        if code == 0:
            if not isinstance(result, tuple):
                ret["result"] = result
            else:
                ret["result"] = result[0]
                ret["history"] = result[1]
        yield ret

