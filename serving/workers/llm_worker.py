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
from serving.utils import logger
from serving.model_handler.base import CompletionResult

def get_worker_instance(model_name,config,group_name,worker_idx):
    model_name: str = model_name.lower()
    model_type = config["model_config"]["model_type"].lower()

    api_client = None
    if model_type == "baichuan":
        if model_name.startswith("baichuan2"):
            if model_name.find('13b') != -1:
                from serving.model_handler.baichuan2_13b.infer import EngineAPI
                api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
            else:
                from serving.model_handler.baichuan2_7b.infer import EngineAPI
                api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        elif model_name.startswith("baichuan"):
            if model_name.find('13b') != -1:
                from serving.model_handler.baichuan_13b.infer import EngineAPI
                api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
            else:
                from serving.model_handler.baichuan_7b.infer import EngineAPI
                api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
    elif model_type == "chatglm" or model_type == "chatglm2" or model_type == "chatglm3":
        if model_name.startswith("chatglm3"):
            from serving.model_handler.chatglm3.infer import EngineAPI
            api_client = EngineAPI(config, group_name=group_name, worker_idx=worker_idx)
        elif model_name.startswith("chatglm2"):
            from serving.model_handler.chatglm2.infer import EngineAPI
            api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
        elif model_name.startswith("chatglm"):
            from serving.model_handler.chatglm.infer import EngineAPI
            api_client = EngineAPI(config,group_name=group_name,worker_idx=worker_idx)
    elif model_type == "t5":
        from serving.model_handler.t5.infer import EngineAPI
        api_client = EngineAPI(config, group_name=group_name, worker_idx=worker_idx)

    elif model_type == "xverse":
        from serving.model_handler.xverse.infer import EngineAPI
        api_client = EngineAPI(config, group_name=group_name, worker_idx=worker_idx)
    elif model_name.startswith("tiger") or model_name.startswith("llama"):
        from serving.model_handler.llama.infer import EngineAPI
        api_client = EngineAPI(config, group_name=group_name, worker_idx=worker_idx)

    elif model_name.startswith("opt") or model_name.startswith("bloom"):
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

    elif model_name.startswith("qwen"):
        from serving.model_handler.qwen.infer import EngineAPI
        api_client = EngineAPI(config, group_name=group_name, worker_idx=worker_idx)

    if not api_client:
        raise ValueError(f'******* {model_name} {model_type} not support yet *********')
    return api_client

class My_worker(ZMQ_process_worker):
    def __init__(self,model_name,config,*args,**kwargs):
        super(My_worker,self).__init__(*args,**kwargs)
        logger.info('group name {} ,worker id {}'.format(self._group_name,self._idx))
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
            logger.info('{} worker pid {}...'.format(self.model_name, os.getpid()))
            self.api_client = get_worker_instance(self.model_name, self.config,self._group_name, self._idx)
            self.api_client.init()
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            self.api_client = None
            self.initial_error = str(e)

    # Process end trigger this func
    def run_end(self):
        print(self.model_name,' process ended....')
        if getattr(self,'api_client',None) is not None:
            try:
                del self.api_client
            except:
                pass

    #any data put will trigger this func
    def run_once(self,request_data):
        r = request_data
        start_time = time.time()
        try:
            if self.initial_error is None:
                method = r.get('method', "chat")
                if method == 'chat_stream':
                    gen = self.api_client.trigger_generator(r)
                    node: typing.Optional[CompletionResult]
                    for node in gen:
                        end_time = time.time()
                        ret = {
                            "code": node.code,
                            "runtime": (end_time - start_time) * 1000,
                            "msg": node.msg,
                            "complete": node.complete
                        }
                        if node.code == 0:
                            if isinstance(node.result,dict):
                                ret.update(node.result)
                            else:
                                ret["response"] = node.result
                        yield ret
                    return None
                else:
                    node = self.api_client.trigger(r)
            else:
                node = CompletionResult(code=-1,msg=self.initial_error)
        except Exception as e:
            traceback.print_exc()
            logger.info(e)
            node = CompletionResult(code=-1, msg=str(e))
        end_time = time.time()
        ret = {
            "code": node.code,
            "runtime": (end_time - start_time) * 1000,
            "msg": node.msg,
            "complete": True
        }
        if node.code == 0:
            if isinstance(node.result, dict):
                ret.update(node.result)
            else:
                ret["response"] = node.result
        yield ret

