# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/21 10:53

from abc import ABC

class Engine_API_Base(ABC):
    def init(self, model_name):
        raise NotImplemented

    def chat(self,input,**kwargs):
        raise NotImplemented

    def generate(self,input,**kwargs):
        raise NotImplemented

    def with_deepspeed(self):
        raise NotImplemented


    def wrapper_deepspeed(self):
        import deepspeed

        ds_engine = deepspeed.init_inference(
            self.model,
            config=ds_config,
            base_dir=model_path,
            checkpoint=checkpoint,
        )