# -*- coding: utf-8 -*-
# @Time:  20:45
# @Author: tk
# @File：evaluate
import os

import torch
from aigc_zoo.utils.streamgenerator import GenTextStreamer
from deep_training.data_helper import ModelArguments, DataArguments, DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.rwkv4.llm_model import MyTransformer, RwkvConfig, \
    set_model_profile,LoraArguments,LoraModel
from aigc_zoo.utils.rwkv4_generate import Generate
from serving.model_handler.base import EngineAPI_Base, preprocess_input_args,flat_input
from config.main import global_models_info_args
from serving.model_handler.base import CompletionResult,ChunkData,preprocess_input_args,postprocess_input_args


class NN_DataHelper(DataHelper):pass



class EngineAPI(EngineAPI_Base):
    def _load_model(self,device_id=None):
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
        return model,config,tokenizer

    def _load_lora_model(self, device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        # 可以自行修改 RWKV_T_MAX  推理最大长度
        set_model_profile(RWKV_T_MAX=2048, RWKV_FLOAT_MODE='16')

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16},
                                                                       config_class_name=RwkvConfig)

        if config.pad_token_id is None or config.pad_token_id >= config.vocab_size:
                config.pad_token_id = tokenizer.eos_token_id

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir, 'config.json')):
            config =  RwkvConfig.from_pretrained(ckpt_dir)
        config.initializer_weight = False
        lora_args = LoraArguments.from_pretrained(ckpt_dir)

        assert lora_args.inference_mode == True

        new_num_tokens = config.vocab_size
        if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
            config.vocab_size = config.task_specific_params['vocab_size']

        pl_model = MyTransformer(config=config, model_args=model_args,
                                 lora_args=lora_args,
                                 torch_dtype=torch.float16, new_num_tokens=new_num_tokens,
                                 # load_in_8bit=global_args["load_in_8bit"],
                                 # # device_map="auto",
                                 # device_map = {"":0} # 第一块卡
                                 )

        for adapter_name, ckpt_dir in self.lora_conf.items():
            pl_model.load_sft_weight(ckpt_dir, adapter_name=adapter_name)
        self.lora_model = pl_model.backbone
        if len(self.lora_conf) == 1:
            self.lora_model.merge_and_unload()
            self.lora_model.half().eval()
        else:
            self.lora_model = self.lora_model.half().eval()

        if device_id is None:
            self.lora_model.cuda()
        else:
            self.lora_model.cuda(device_id)
        return self.lora_model, config, tokenizer

    def chat_stream(self, query, nchar=1,gtype='total', history=None, **kwargs):
        preprocess_input_args(self.tokenizer, kwargs)

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
        postprocess_input_args(self.tokenizer,default_kwargs)
        chunk = ChunkData()
        chunk.n_id = 0
        def process_token_fn(text,stream_end,chunk: ChunkData):
            chunk.n_id += 1
            chunk.text += text
            chunk.idx += 1
            if chunk.idx % nchar == 0 or stream_end or chunk.idx == 1:
                ret = CompletionResult(result={
                    "response": chunk.text,
                    "history": history,
                    "num_token": chunk.n_id
                }, complete=False)
                if gtype == 'total':
                    self.push_response(ret)
                    chunk.idx = 0
                else:
                    self.push_response(ret)
                    chunk.clear()

        skip_word_list = default_kwargs.get('eos_token_id',None) or [self.tokenizer.eos_token_id]
        streamer = GenTextStreamer(process_token_fn,chunk,tokenizer=self.tokenizer,skip_word_list=flat_input(skip_word_list),skip_prompt=True)
        _ = Generate.generate(self.get_model(),tokenizer=self.tokenizer,streamer=streamer, query=prompt, **default_kwargs)
        if gtype == 'total':
            ret = CompletionResult(result={
                "response": chunk.text,
                "history": history,
                "num_token": chunk.n_id
            }, complete=False)
            self.push_response(ret)

        ret = CompletionResult(result={
            "response": "",
            "history": history,
            "num_token": chunk.n_id
        }, complete=True)
        self.push_response(ret)
        return None

    def chat(self, query, history=None, **kwargs):
        preprocess_input_args(self.tokenizer, kwargs)
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
        postprocess_input_args(self.tokenizer,default_kwargs)
        response = Generate.generate(self.get_model(),
                                     tokenizer=self.tokenizer,
                                     query=prompt, **default_kwargs)
        history = history + [(query, response)]
        return CompletionResult(result={
            "response": response,
            "history": history
        })


    def generate(self,input,**kwargs):
        default_kwargs = dict(
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,default_kwargs)
        response = Generate.generate(self.get_model(),
                                     tokenizer=self.tokenizer,
                                     query=input,**default_kwargs)
        return response


if __name__ == '__main__':
    api_client = EngineAPI(global_models_info_args['rwkv-4-raven-3b-v12-Eng49%-Chn49%-Jpn1%-Other1%'])
    api_client.init()
    text_list = [
        "你是谁?",
        "你会干什么?",
    ]
    for input in text_list:
        response = api_client.generate(input)
        print('input', input)
        print('output', response)