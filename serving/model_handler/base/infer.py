# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/21 10:53
import logging
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
from deep_training.nlp.models.lora.v2 import LoraModel
from serving.model_handler.base.data_define import WorkMode
from serving.model_handler.base.data_process import preprocess_input_args,flat_input # noqa
from serving.model_handler.base.data_define import CompletionResult # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class EngineAPI_Base(ABC):
    def __init__(self,model_config_dict,group_name="",worker_idx=0):
        self.model_config_dict = model_config_dict
        self.group_name = group_name
        self.worker_idx = worker_idx

        self.model_accelerate = None
        self.model_ds = None
        self.model = None

        self.auto_quantize = model_config_dict.get('auto_quantize',True)
        self.lora_model: typing.Optional[LoraModel] = None
        self.current_adapter_name = ''
        self.lora_conf = model_config_dict['model_config']['lora']
        self.muti_lora_num = len(self.lora_conf.keys())
        self.work_mode_str = model_config_dict['work_mode'].lower()
        device_id = model_config_dict['workers'][worker_idx]['device_id']
        if device_id is None:
            device_id = [0]
        self.device_id = device_id
        self.world_size = len(self.device_id)
        self.work_mode = WorkMode.STANDORD_HF

        self._q_in = None
        self._q_out = None
        self.rank = 0

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

    def _init_data(self):
        if self._q_in is None:
            self._q_in = multiprocessing.Manager().Queue()
        if self._q_out is None:
            self._q_out = multiprocessing.Manager().Queue()

    def init(self):
        skip_init = False
        self._init_data()
        if self.world_size > 1 and self.muti_lora_num <= 1:
            if self.work_mode_str == 'deepspeed':
                self.work_mode = WorkMode.DS
                skip_init = True
                self._init_worker_ds()
            elif self.work_mode_str == 'accelerate':
                self.work_mode = WorkMode.ACCELERATE
                skip_init = True
                self._init_accelerate()
        else:
            self.work_mode_str = 'hf'

        if not skip_init:
            self.init_model()
        self._init_thead_generator()
        logger.info('serving ready')

    def init_model(self, device_id=None):
        self.model_config_dict['seed'] = None
        if self.muti_lora_num > 0:
            call_method = self._load_lora_model
        else:
            call_method = self._load_model

        self.model, self.config, self.tokenizer = call_method(device_id)

    def get_model(self):
        return self.model_ds or self.model_accelerate or self.model


    def chat_stream(self,query,nchar=1,gtype='total',**kwargs):
        raise NotImplemented

    def chat(self,input,**kwargs):
        raise NotImplemented

    def generate(self,input,**kwargs):
        raise NotImplemented

    def worker_ds(self,rank):
        try:
            self.rank = rank
            import deepspeed
            from deepspeed.inference.config import DeepSpeedInferenceConfig
            from deepspeed.inference.engine import InferenceEngine
            torch.cuda.set_device(rank)
            dist.init_process_group("nccl", rank=rank, world_size=self.world_size,group_name=self.group_name + str(self.worker_idx))
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
            logger.info(e)
            self.model_ds = None

    def push_request(self,data):
        return self._q_in.put(data)

    def pull_request(self):
        return self._q_in.get()

    def pull_response(self):
        return self._q_out.get()

    def push_response(self, data):
        if self.rank == 0:
            self._q_out.put(data)

    def loop_forever(self,rank):
        if self.rank == 0:
            logging.info('=' * 30)
            logging.info(self.group_name)
            logging.info('\nserving is loaded , wait for serve...\n')
            logging.info('=' * 30)

        while True:
            r = self.pull_request()
            try:
                if r.get('method', "chat") == 'chat_stream':
                    for item in self.trigger_generator(r=r, is_first=False):
                        self.push_response(item)
                    continue
                else:
                    ret = self.trigger(r=r, is_first=False)
            except KeyboardInterrupt:
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                ret = CompletionResult(code=-1,result=None,msg=str(e),complete=True)
            if rank == 0:
                self.push_response(ret)



    def _init_worker_ds(self):
        os_conf = self.model_config_dict['workers'][self.worker_idx]['deepspeed']
        for k,v in os_conf.items():
            os.environ[k] = v
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

    def _init_thead_generator(self):
        if self.work_mode != WorkMode.DS:
            self._thread_generator = threading.Thread(target=self._loop_thread)
            self._thread_generator.start()


    def clear_data(self):
        while self._q_in.qsize():
            self._q_in.get()
        while self._q_out.qsize():
            self._q_out.get()

    def _loop_thread(self):
        logging.info('=' * 30)
        logging.info(self.group_name)
        logging.info('\nserving is loaded , wait for serve...\n')
        logging.info('=' * 30)
        while True:
            r = self.pull_request()
            try:
                gen_out = self.trigger_generator(r=r, is_first=False)
                for out in gen_out:
                    self.push_response(out)
            except KeyboardInterrupt:
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(e)

                ret = CompletionResult(code=-1,result=None,msg=str(e),complete=True)
                self.push_response(ret)

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

    def switch_lora(self,adapter_name):
        if len(self.lora_conf) == 0:
            return 0,'ok'

        if adapter_name not in self.lora_conf:
            return -1,'{} not in {}'.format(adapter_name,','.join(list(self.lora_conf.keys())))

        if adapter_name == self.current_adapter_name:
            return 0, 'ok'
        
        self.current_adapter_name = adapter_name
        self.lora_model.set_adapter(adapter_name)
        return 0,'ok'

    def trigger_generator(self ,r: typing.Dict,is_first=True):
        ret = CompletionResult()
        if is_first:
            self.clear_data()
        if self.work_mode == WorkMode.DS:
            if is_first:
                for i in range(self.world_size):
                    self.push_request(r)
                while True:
                    result_tuple = self.pull_response()
                    yield result_tuple
                    if result_tuple.complete:
                        break
                return None

            if self.model_ds is None:
                yield ret._replace(code=-1,result=None,msg="ds_engine init failed",complete=True)
        else:
            if is_first:
                self.push_request(r)
                while True:
                    result_tuple = self.pull_response()
                    yield result_tuple
                    if result_tuple.complete:
                        break
                return None
        try:
            params = r.get('params', {})
            query = r.get('query', "")
            history = r.get('history', [])
            history = [(_["q"], _["a"]) for _ in history]

            adapter_name = params.pop('adapter_name', 'default')
            code, msg = self.switch_lora(adapter_name)
            if code != 0:
                yield ret._replace(code=code,result=None,msg=msg,complete=True)

            gen_results = self.chat_stream(query, history=history, **params)
            if gen_results is None:
                return None
            iter_: CompletionResult
            for iter_ in gen_results:
                if self.work_mode == WorkMode.DS:
                    self.push_response(ret._replace(code=code,result=iter_.result,msg=msg,complete=False))
                else:
                    yield ret._replace(code=code,result=iter_.result,msg=msg,complete=False)
            result = None
            code = 0
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            code = -1
            msg = str(e)
            result = None
        yield ret._replace(code=code,result=result,msg=msg,complete=True)


    def trigger(self ,r: typing.Dict,is_first=True):
        ret = CompletionResult(complete=True)
        if is_first:
            self.clear_data()
        if self.work_mode == WorkMode.DS:
            if is_first:
                for i in range(self.world_size):
                    self.push_request(r)
                result_tuple = self.pull_response()
                return result_tuple

            if self.model_ds is None:
                return ret._replace(code=-1, msg="ds_engine init failed")

        method_fn = getattr(self, "chat")
        if method_fn is not None:
            params = r.get('params', {})
            if not isinstance(params, dict):
                return ret._replace(code=-1,msg="params error")

            adapter_name = params.pop('adapter_name','default')
            code,msg = self.switch_lora(adapter_name)
            if code != 0:
                return ret._replace(code=-1,msg=msg)

            query = r.get('query', "")
            history = r.get('history', [])
            history = [(_["q"], _["a"]) for _ in history]
            node:CompletionResult = method_fn(query, history=history, **params)
            # history = [{"q": _[0], "a": _[1]} for _ in results["history"]]
            result = {
                "response": node.result["response"],
                # "history": history,
                "num_token": node.result.get('num_token',node.result["response"])
            }
        else:
            code = -1
            msg = "{} not exist method {}".format(self.model_config_dict['model_config']['model_type'], "chat")
            result = None
        return ret._replace(code=code,result=result, msg=msg)