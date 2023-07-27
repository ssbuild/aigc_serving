# -*- coding: utf-8 -*-
# @Time:  20:45
# @Author: tk
# @File：evaluate
import torch
from aigc_zoo.utils.streamgenerator import GenTextStreamer
from deep_training.data_helper import ModelArguments, DataArguments, DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.rwkv4.llm_model import MyTransformer, RwkvConfig, set_model_profile
from aigc_zoo.utils.rwkv4_generate import Generate
from serving.model_handler.base import EngineAPI_Base
from config.constant_map import models_info_args
from serving.model_handler.base.data_define import ChunkData


class NN_DataHelper(DataHelper):pass



class EngineAPI(EngineAPI_Base):
    def init_model(self,device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        # 可以自行修改 RWKV_T_MAX  推理最大长度
        set_model_profile(RWKV_T_MAX=2048, RWKV_FLOAT_MODE='16')

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16},
                                                                       config_class_name=RwkvConfig)

        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16)
        model = pl_model.get_llm_model()

        model.requires_grad_(False)
        model.eval().half()
        if device_id is None:
            model.cuda()
        else:
            model.cuda(device_id)

        self.model = model
        self.tokenizer = tokenizer

    def chat_stream(self, query, nchar=1,gtype='total', history=None, **kwargs):
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
        chunk = ChunkData()

        def process_token_fn(text,stream_end,chunk: ChunkData):
            chunk.text += text
            chunk.idx += 1
            if chunk.idx % nchar == 0 or stream_end or chunk.idx == 1:
                if gtype == 'total':
                    self.push_response(((chunk.text, history), 0, "ok", False))
                    chunk.idx = 0
                else:
                    self.push_response(((chunk.text, history), 0, "ok", False))
                    chunk.clear()

        streamer = GenTextStreamer(process_token_fn,chunk,tokenizer=self.tokenizer)
        _ = Generate.generate(self.get_model(),tokenizer=self.tokenizer,streamer=streamer, query=prompt, **default_kwargs)
        if gtype == 'total':
            self.push_response(((chunk.text, history), 0, "ok", False))
        self.push_response((('', history), 0, "ok", True))
        return None

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
                                     query=prompt, **default_kwargs)
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
    api_client = EngineAPI(models_info_args['rwkv-4-raven-3b-v12-Eng49%-Chn49%-Jpn1%-Other1%'])
    api_client.init()
    text_list = [
        "你是谁?",
        "你会干什么?",
    ]
    for input in text_list:
        response = api_client.generate(input)
        print('input', input)
        print('output', response)