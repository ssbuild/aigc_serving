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
from enum import Enum

class WorkMode(Enum):
    STANDORD_HF = 0
    DS = 1
    ACCELERATE = 2


class EngineAPI_Base(ABC):
    def __init__(self,model_config_dict,worker_idx=0):
        self.model_accelerate = None
        self.model_ds = None
        self.model = None
        self.model_config_dict = model_config_dict
        self.work_mode_str = model_config_dict['workers']['mode'].lower()
        device_id = model_config_dict['workers']['worker'][worker_idx]['device_id']
        if device_id is None:
            device_id = [0]
        self.device_id = device_id
        self.world_size = len(self.device_id)
        self.work_mode = WorkMode.STANDORD_HF
        self._q_in = None
        self._q_out = None

    # def __del__(self):
    #     self.release()

    # def release(self):
    #     if self.work_mode == WorkMode.STANDORD_HF:
    #         pass
    #     elif self.work_mode == WorkMode.DS:
    #         pass
    #     elif self.work_mode == WorkMode.ACCELERATE:
    #         pass
    #     if hasattr(self, '_spawn_context'):
    #         for process in self._spawn_context.processes:
    #             if process.is_alive():
    #                 process.terminate()
    #             process.join()

    def init(self):
        skip_init = False
        print('*' * 30,self.world_size)
        if self.world_size >= 1:
            if self.work_mode_str == 'deepspeed':
                self.work_mode = WorkMode.DS
                skip_init = True
                self.thread = threading.Thread(target=self._init_woker_ds(), args=())
                self.thread.start()
            elif self.work_mode_str == 'accelerate':
                self.work_mode = WorkMode.ACCELERATE
                skip_init = True
                self._init_accelerate()
        if not skip_init:
            self.init_model()

    def get_model(self):
        return self.model_ds or self.model_accelerate or self.model

    def init_model(self,device_id=0):
        raise NotImplemented

    def chat(self,input,**kwargs):
        raise NotImplemented

    def generate(self,input,**kwargs):
        raise NotImplemented

    def worker_ds(self,rank):
        import deepspeed
        from deepspeed.accelerator import get_accelerator
        dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
        torch.cuda.set_device(rank)
        self.init_model(rank)
        try:
            self.model_ds = deepspeed.init_inference(self.model,
                                                     mp_size=self.world_size,
                                                     # dtype=torch.half,
                                                     # checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                                     replace_with_kernel_inject=True)

            self.model_ds.device = self.model.device
            self.loop_forever()
        except Exception as e:
            traceback.print_exc()
            print(e)
            self.model_ds = None



    def loop_forever(self):
        while True:
            r = self._q_in.get()
            try:
                result, code, msg = self.trigger(r=r, is_first=False)
            except KeyboardInterrupt:
                break
            except Exception as e:
                result = None
                code = -1
                msg = str(e)
            self._q_out.put((result,code,msg))



    def _init_woker_ds(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        # os.environ[
        #     "TORCH_DISTRIBUTED_DEBUG"
        # ] = "DETAIL"  # set to DETAIL for runtime logging.

        self._q_in,self._q_out = multiprocessing.Manager().Queue(),multiprocessing.Manager().Queue()
        self._spawn_context = mp.spawn(self.worker_ds, nprocs=self.world_size, join=False)


    def _init_accelerate(self):
        from accelerate import dispatch_model
        self.init_model()
        self.device_map = self.infer_auto_device_map()
        self.model_accelerate = dispatch_model(self.model, device_map=self.device_map)


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




    def trigger(self ,r: typing.Dict , is_first=True):
        if self.work_mode == WorkMode.DS:
            if self.model_ds is None:
                return [], -1, "ds_engine init failed"
            if is_first:
                self._q_in.put(r)
                result,code,msg = self._q_out.get()
                return result,code,msg
        result = []
        msg = "ok"
        code = 0
        texts = r.get('texts', [])
        params = r.get('params', {})
        method = r.get('method', "generate")
        method_fn = getattr(self, method)
        if method_fn is not None:
            if isinstance(params, dict):
                for text in texts:
                    result.append(method_fn(text, **params))
            else:
                code = -1
                msg = "params error"
        else:
            code = -1
            msg = "{} not exist method {}".format(self.model_config_dict['model_config']['model_type'], method)
        return (result,code,msg)