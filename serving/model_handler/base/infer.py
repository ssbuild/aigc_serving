# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/21 10:53
import os
import time
import traceback
import typing
from abc import ABC
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
import threading
from multiprocessing import Queue
from serving.model_handler.base.data_define import WorkMode


class EngineAPI_Base(ABC):
    def __init__(self,model_config_dict,group_name="",worker_idx=0):
        self.model_config_dict = model_config_dict
        self.group_name = group_name
        self.worker_idx = worker_idx

        self.model_accelerate = None
        self.model_ds = None
        self.model = None

        self.work_mode_str = model_config_dict['work_mode'].lower()
        device_id = model_config_dict['workers'][worker_idx]['device_id']
        if device_id is None:
            device_id = [0]
        self.device_id = device_id
        self.world_size = len(self.device_id)
        self.work_mode = WorkMode.STANDORD_HF
        self._q_in = None
        self._q_out = None

    def __del__(self):
        self._release()

    def _release(self):
        if self.work_mode == WorkMode.STANDORD_HF:
            pass
        elif self.work_mode == WorkMode.DS:
            pass
        elif self.work_mode == WorkMode.ACCELERATE:
            pass
        if getattr(self, '_spawn_context',None) is not None:
            for process in self._spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            self._spawn_context = None

    def init(self):
        skip_init = False
        if self.world_size > 1:
            if self.work_mode_str == 'deepspeed':
                self.work_mode = WorkMode.DS
                skip_init = True
                # self.thread = threading.Thread(target=self._init_woker_ds(), args=())
                # self.thread.start()
                target = self._init_woker_ds()
            elif self.work_mode_str == 'accelerate':
                self.work_mode = WorkMode.ACCELERATE
                skip_init = True
                self._init_accelerate()
        else:
            self.work_mode_str = 'hf'

        if not skip_init:
            self.init_model()

    def get_model(self):
        return self.model_ds or self.model_accelerate or self.model

    def init_model(self,device_id=None):
        raise NotImplemented

    def chat_stream(self,input,**kwargs):
        raise NotImplemented

    def chat(self,input,**kwargs):
        raise NotImplemented

    def generate(self,input,**kwargs):
        raise NotImplemented

    def worker_ds(self,rank):
        try:
            import deepspeed
            from deepspeed.inference.config import DeepSpeedInferenceConfig
            from deepspeed.inference.engine import InferenceEngine

            torch.cuda.set_device(rank)
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size,group_name=self.group_name)
            self.init_model()
            old_current_device_function = deepspeed.get_accelerator().current_device_name
            def tmp_current_device_fn():
                deepspeed.get_accelerator().set_device(rank)
                deepspeed.get_accelerator().current_device_name = old_current_device_function
                return deepspeed.get_accelerator().current_device_name()
            deepspeed.get_accelerator().current_device_name = tmp_current_device_fn
            ds_config = DeepSpeedInferenceConfig(**dict(
                                                 mp_size=self.world_size,
                                                 # dtype=torch.half,
                                                 # checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                                 replace_with_kernel_inject=True
                                             ))
            self.model_ds = InferenceEngine(self.model,config=ds_config)
            if hasattr(self.model,'chat'):
                self.model_ds.chat = self.model.chat
            if hasattr(self.model,'chat_stream'):
                self.model_ds.chat_stream = self.model.chat_stream
            self.model_ds.device = self.model.device
            self.loop_forever(rank)
        except Exception as e:
            traceback.print_exc()
            print(e)
            self.model_ds = None



    def loop_forever(self,rank):
        while True:
            r = self._q_in.get()
            try:
                result, code, msg,complete_flag = self.trigger(r=r, is_first=False)
            except KeyboardInterrupt:
                break
            except Exception as e:
                result = None
                code = -1
                msg = str(e)
                complete_flag = True
            if rank == 0:
                self._q_out.put((result,code,msg,complete_flag))



    def _init_woker_ds(self):
        os_conf = self.model_config_dict['deepspeed']
        for k,v in os_conf.items():
            os.environ[k] = v
        self._q_in,self._q_out = multiprocessing.Manager().Queue(),multiprocessing.Manager().Queue()
        self._spawn_context = mp.spawn(self.worker_ds, nprocs=self.world_size, join=False)


    def _init_accelerate(self):
        from accelerate import dispatch_model
        self.init_model()
        self.device_map = self.infer_auto_device_map()
        self.model_accelerate = dispatch_model(self.model, device_map=self.device_map)
        if hasattr(self.model, 'chat'):
            self.model_accelerate.chat = self.model.chat
        if hasattr(self.model, 'chat_stream'):
            self.model_accelerate.chat_stream = self.model.chat_stream


    def infer_auto_device_map(self):
        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        # from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
        load_in_4bit = getattr(self.model, 'load_in_4bit', False)
        load_in_8bit = getattr(self.model, 'load_in_8bit', False)
        dtype = None
        if load_in_4bit:
            dtype = torch.int8
        elif load_in_8bit:
            dtype = torch.int8

        max_memory = get_balanced_memory(self.model,
                                         dtype=dtype,
                                         low_zero=False,
                                         no_split_module_classes=self.model._no_split_modules)

        device_map = infer_auto_device_map(self.model,
                                           dtype=dtype,
                                           max_memory=max_memory,
                                           no_split_module_classes=self.model._no_split_modules)
        return device_map


    def trigger_generator(self ,r: typing.Dict,is_first=True):
        if self.work_mode == WorkMode.DS:
            if is_first:
                for i in range(self.world_size):
                    self._q_in.put(r)
                result_tuple = self._q_out.get()
                while not result_tuple[-1]:
                    yield result_tuple
                yield [], 0, "ok", True

            if self.model_ds is None:
                yield [], -1, "ds_engine init failed",True

        result = []
        msg = "ok"
        code = 0

        # method = r.get('method', "generate")
        method = 'chat_stream'
        method_fn = getattr(self, method)
        if method_fn is not None:
            params = r.get('params', {})
            query = r.get('query', "")
            history = r.get('history', [])
            history = [(_["q"], _["a"]) for _ in history]
            for results in method_fn(query, history=history, **params):
                result = results[0]
                self._q_out.put((result, code, msg, False))
            result = ""
            code = 0
        else:
            code = -1
            msg = "{} not exist method {}".format(self.model_config_dict['model_config']['model_type'], method)
        yield result,code,msg,True


    def trigger(self ,r: typing.Dict,is_first=True):
        if self.work_mode == WorkMode.DS:
            if is_first:
                for i in range(self.world_size):
                    self._q_in.put(r)
                result_tuple = self._q_out.get()
                return result_tuple

            if self.model_ds is None:
                return [], -1, "ds_engine init failed",True

        result = []
        msg = "ok"
        code = 0

        method = r.get('method', "generate")
        method_fn = getattr(self, method)
        if method_fn is not None:
            params = r.get('params', {})
            if not isinstance(params, dict):
                code = -1
                msg = "params error"
                return result, code, msg,True

            if method == 'generate':
                texts = r.get('texts', [])
                for text in texts:
                    result.append(method_fn(text, **params))
            elif method == 'chat':
                query = r.get('query', "")
                history = r.get('history', [])
                history = [(_["q"], _["a"]) for _ in history]
                results = method_fn(query, history=history, **params)
                history = [{"q": _[0], "a": _[1]} for _ in results[1]]
                result = (results[0], history)
            else:
                code = -1
                msg = "{} not exist method {}".format(self.model_config_dict['model_config']['model_type'], method)
        else:
            code = -1
            msg = "{} not exist method {}".format(self.model_config_dict['model_config']['model_type'], method)
        return result,code,msg,True