# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/10 17:24
import torch
from deep_training.data_helper import ModelArguments,DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.chatglm2.chatglm_model import MyTransformer, ChatGLMTokenizer, LoraArguments, \
    setup_model_profile, ChatGLMConfig
from serving.model_handler.base import EngineAPI_Base
from config.constant_map import models_info_args
class NN_DataHelper(DataHelper):pass


class EngineAPI(EngineAPI_Base):
    def init_model(self,device_id=None):
        models_info_args['seed'] = None
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)
        setup_model_profile()
        dataHelper = NN_DataHelper(model_args)
        tokenizer: ChatGLMTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
            tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16, )
        model = pl_model.get_llm_model()
        if not model.quantized:
            # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
            model.half().quantize(4)
        else:
            # 已经量化
            model.half()
        model = model.eval()
        if device_id is None:
            model.cuda()
        else:
            model.cuda(device_id)

        self.model = model
        self.tokenizer = tokenizer

    def chat(self, input, **kwargs):
        default_kwargs = dict(history=[], 
            eos_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        response, history = self.model.chat(self.tokenizer, query=input,  **default_kwargs)
        return response, history

    def generate(self,input,**kwargs):
        default_kwargs = dict(history=[], 
            eos_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        #response, history = self.model.chat(self.tokenizer, query=input,  **kwargs)
        output = self.model.chat(self.tokenizer, query=input, **default_kwargs)
        output_scores = default_kwargs.get('output_scores', False)
        if output_scores:
            return output
        response, history = output
        return response

if __name__ == '__main__':
    api_client = EngineAPI(models_info_args['chatglm2-6b-int4'])
    api_client.init()
    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    for input in text_list:
        response = api_client.generate(input)
        print("input", input)
        print("response", response)