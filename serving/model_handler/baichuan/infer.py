# -*- coding: utf-8 -*-
# @Time:  18:49
# @Author: tk
# @File：evaluate
import torch
from deep_training.data_helper import ModelArguments, DataArguments,DataHelper
from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig
from aigc_zoo.model_zoo.baichuan.llm_model import MyTransformer,BaiChuanConfig,BaiChuanTokenizer
from aigc_zoo.utils.llm_generate import Generate
from config.constant_map import models_info_args
from serving.model_handler.base import EngineAPI_Base
from serving.model_handler.base.data_define import ChunkData


class NN_DataHelper(DataHelper):pass


class EngineAPI(EngineAPI_Base):
    def init_model(self,device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=BaiChuanConfig,
                                                                       tokenizer_class_name=BaiChuanTokenizer)
        config.pad_token_id = config.eos_token_id

        pl_model = MyTransformer(config=config, model_args=model_args,
                                 torch_dtype=torch.float16, )

        model = pl_model.get_llm_model()
        model = model.eval()
        model.requires_grad_(False)

        model = model.half()
        if device_id is None:
            model.cuda()
        else:
            model.cuda(device_id)

        self.model = model
        self.tokenizer = tokenizer

    def chat_stream(self,  query, n,gtype='total', history=None,**kwargs):
        if history is None:
            history = []
        prompt = ""
        for q, a in history:
            prompt += q
            prompt += a
        prompt += query

        default_kwargs = dict(eos_token_id=self.model.config.eos_token_id,
                              pad_token_id=self.model.config.eos_token_id,
                              do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                              repetition_penalty=1.1,
                              )
        default_kwargs.update(kwargs)
        generation_config = GenerationConfig(**default_kwargs)

        prompt = query
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.get_model().device)

        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
        self.__class__.generate = NewGenerationMixin.generate
        self.__class__.sample_stream = NewGenerationMixin.sample_stream
        stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True)

        def stream_generator():
            outputs = []
            for token in self.get_model().generate(**inputs, generation_config=stream_config):
                outputs.append(token.item())
                yield self.tokenizer.decode(outputs, skip_special_tokens=True)

        chunk = ChunkData()
        chunk.idx = 0
        n_id = 0

        response = None
        for response in stream_generator():
            n_id += 1
            chunk.text = response
            if n_id % n == 0:
                if gtype == 'total':
                    yield (chunk.text, history)
                else:
                    yield (chunk.text[chunk.idx:], history)
                    chunk.idx = len(response)

        history = history + [(query, response)]
        if gtype != 'total' and chunk.idx != len(chunk.text):
            yield (chunk.text[chunk.idx:], history)

        return response, history


    def chat(self, query, history=None, **kwargs):
        if history is None:
            history = []
        prompt = ""
        for q, a in history:
            prompt += q
            prompt += a
        prompt += query

        default_kwargs = dict(
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        response = Generate.generate(self.get_model(),
                                     tokenizer=self.tokenizer,
                                     query=prompt, **kwargs)
        history = history + [(query, response)]
        return response, history

    def generate(self,input,**kwargs):
        default_kwargs = dict(
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        response = Generate.generate(self.get_model(),
                                     tokenizer=self.tokenizer,
                                     query=input,**default_kwargs)
        return response


if __name__ == '__main__':
    api_client = EngineAPI(models_info_args['baichuan-7B'])
    api_client.init()
    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 "登鹳雀楼->王之涣\n夜雨寄北->",
                 "Hamlet->Shakespeare\nOne Hundred Years of Solitude->",
                 ]
    for input in text_list:
        response = api_client.generate(input)
        print('input', input)
        print('output', response)
