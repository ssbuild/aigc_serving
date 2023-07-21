# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 13:33
# @Author  : tk
import os
import time
import typing

from ipc_worker.ipc_zmq_loader import IPC_zmq,ZMQ_process_worker
import copy


def get_worker_instance(model_name,config):
    model_name: str = model_name.lower()
    if model_name.startswith("baichuan"):
        if model_name.find('13b') != -1:
            from serving.model_handler.baichuan2.infer import Engine_API
            api_client = Engine_API()
        else:
            from serving.model_handler.baichuan.infer import Engine_API
            api_client = Engine_API()
            
    elif model_name.startswith("chatglm2"):
        from serving.model_handler.chatglm2.infer import Engine_API
        api_client = Engine_API()
        
    elif model_name.startswith("chatglm"):
        from serving.model_handler.chatglm.infer import Engine_API
        api_client = Engine_API()
        
    elif model_name.startswith("llama") or model_name.startswith("opt") or model_name.startswith("bloom"):
        from serving.model_handler.llm.infer import Engine_API
        api_client = Engine_API()
        
    elif model_name.startswith("internlm"):
        from serving.model_handler.internlm.infer import Engine_API
        api_client = Engine_API()
        
    elif model_name.startswith("moss"):
        from serving.model_handler.moss.infer import Engine_API
        api_client = Engine_API()
        
    elif model_name.startswith("rwkv"):
        from serving.model_handler.rwkv.infer import Engine_API
        api_client = Engine_API()

    else:
        raise ValueError('not support yet')
    return api_client

class My_worker(ZMQ_process_worker):
    def __init__(self,model_name,config,*args,**kwargs):
        super(My_worker,self).__init__(*args,**kwargs)
        self._logger.info('Process id {}, group name {} , identity {}'.format(self._idx,self._group_name,self._identity))
        # config['model_config']['device_id'] = self._idx % device_num if device_num > 0 else 0
        self._logger.info(config)
        self.config = copy.deepcopy(config)
        self.model_name = model_name
        self.api_client = None

    #Process begin trigger this func
    def run_begin(self):
        self._logger.info('{} worker pid {}...'.format(self.model_name , os.getpid()))
        self.api_client = get_worker_instance(self.model_name,self.config)
        assert self.api_client is not None
        self.api_client.init(self.config)

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