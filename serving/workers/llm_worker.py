# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 13:33
# @Author  : tk
import os
import time
import typing
from ipc_worker.ipc_zmq_loader import IPC_zmq,ZMQ_process_worker  # noqa
import copy


def get_worker_instance(model_name,config):
    model_name: str = model_name.lower()
    if model_name.startswith("baichuan"):
        if model_name.find('13b') != -1:
            from serving.model_handler.baichuan2.infer import EngineAPI
            api_client = EngineAPI(config)
        else:
            from serving.model_handler.baichuan.infer import EngineAPI
            api_client = EngineAPI(config)
            
    elif model_name.startswith("chatglm2"):
        from serving.model_handler.chatglm2.infer import EngineAPI
        api_client = EngineAPI(config)
        
    elif model_name.startswith("chatglm"):
        from serving.model_handler.chatglm.infer import EngineAPI
        api_client = EngineAPI(config)
        
    elif model_name.startswith("llama") or model_name.startswith("opt") or model_name.startswith("bloom"):
        from serving.model_handler.llm.infer import EngineAPI
        api_client = EngineAPI(config)
        
    elif model_name.startswith("internlm"):
        from serving.model_handler.internlm.infer import EngineAPI
        api_client = EngineAPI(config)
        
    elif model_name.startswith("moss"):
        from serving.model_handler.moss.infer import EngineAPI
        api_client = EngineAPI(config)
        
    elif model_name.startswith("rwkv"):
        from serving.model_handler.rwkv.infer import EngineAPI
        api_client = EngineAPI(config)

    else:
        raise ValueError('not support yet')
    return api_client

class My_worker(ZMQ_process_worker):
    def __init__(self,model_name,config,*args,**kwargs):
        super(My_worker,self).__init__(*args,**kwargs)
        self._logger.info('Process id {}, group name {} , identity {}'.format(self._idx,self._group_name,self._identity))
        self._logger.info(config)
        self.config = copy.deepcopy(config)
        self.model_name = model_name
        self.api_client = None
        self.initial_error = None

    #Process begin trigger this func
    def run_begin(self):
        try:
            device_id = self.config.get("device_id", None)
            if device_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_id) if isinstance(device_id,(list,tuple)) else str(device_id)
            self._logger.info('{} worker pid {}...'.format(self.model_name, os.getpid()))
            self.api_client = get_worker_instance(self.model_name, self.config)
            self.api_client.init()
        except Exception as e:
            self.api_client = None
            self.initial_error = str(e)

    # Process end trigger this func
    def run_end(self):
        print(self.model_name,' process ended....')

    #any data put will trigger this func
    def run_once(self,request_data):
        r = request_data
        result = []
        code = 0
        start_time = time.time()
        msg = "ok"
        try:
            if self.initial_error is None:
                texts = r.get('texts', [])
                params = r.get('params', {})
                method = r.get('method', "generate")
                method_fn = getattr(self.api_client, method)
                if method_fn is not None:
                    if isinstance(params,dict):
                        for text in texts:
                            result.append(method_fn(text,**params))
                    else:
                        code = -1
                        msg = "params error"
                else:
                    code = -1
                    msg = "{} not exist method {}".format(self.model_name,method)
            else:
                code = -1
                msg = self.initial_error
        except Exception as e:
            msg = str(e)
            self._logger.info(e)
            code = -1
        end_time = time.time()
        return {
            "code": code,
            "runtime": (end_time - start_time) * 1000,
            "result": result,
            "msg": msg
        }