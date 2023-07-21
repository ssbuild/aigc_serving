# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/21 10:53

from abc import ABC
import torch

class EngineAPI_Base(ABC):
    def __init__(self,model_config_dict):
        self.accelerate_model = None
        self.ds_model = None
        self.model = None
        self.model_config_dict = model_config_dict

    def init(self):
        self.init_model()
        if torch.cuda.device_count() > 1:
            if self.model_config_dict.get('deepspeed',{}).get("enable",False):
                self.wrapper_deepspeed()
            elif self.model_config_dict.get('accelerate',{}).get("enable",False):
                self.wrapper_accelerate()

    def get_model(self):
        return self.ds_model or self.accelerate_model or self.model

    def init_model(self):
        raise NotImplemented

    def chat(self,input,**kwargs):
        raise NotImplemented

    def generate(self,input,**kwargs):
        raise NotImplemented

    def with_deepspeed(self):
        raise NotImplemented


    def wrapper_deepspeed(self):
        raise NotImplemented
        # import deepspeed
        #
        # ds_engine = deepspeed.init_inference(
        #     self.model,
        #     config=ds_config,
        #     base_dir=model_path,
        #     checkpoint=checkpoint,
        # )

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

    def wrapper_accelerate(self,**kwargs):
        from accelerate import dispatch_model
        self.device_map = self.infer_auto_device_map()
        self.accelerate_model = dispatch_model(self.model,device_map = self.device_map,**kwargs)

