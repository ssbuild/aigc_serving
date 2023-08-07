# -*- coding: utf-8 -*-
# @Time:  19:26
# @Author: tk
# @File：evaluate
import os

import torch
from deep_training.data_helper import ModelArguments, DataArguments, DataHelper
from transformers import HfArgumentParser, TextStreamer
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer,LoraArguments,LoraModel,AutoConfig
from aigc_zoo.utils.llm_generate import Generate
from serving.model_handler.base import EngineAPI_Base
from config.main import global_models_info_args
from aigc_zoo.utils.streamgenerator import GenTextStreamer
from serving.model_handler.base.data_define import ChunkData
from deep_training.utils.hf import register_transformer_model
from deep_training.nlp.models.rellama.modeling_llama import ReLlamaForCausalLM
from transformers import AutoModelForCausalLM

class NN_DataHelper(DataHelper):pass



class EngineAPI(EngineAPI_Base):
    def _load_model(self,device_id=None):
        register_transformer_model(ReLlamaForCausalLM, AutoModelForCausalLM)
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()
        if config.pad_token_id is None or config.pad_token_id >= config.vocab_size:
            config.pad_token_id = tokenizer.eos_token_id

        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=config.torch_dtype, )
        model = pl_model.get_llm_model()
        model = model.eval()
        if hasattr(model,'quantize'):
            if not model.quantized:
                # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
                if self.auto_quantize:
                    model.half().quantize(4)
                else:
                    model.half()
            else:
                # 已经量化
                model.half()
        else:
            model.half()

        if device_id is None:
            model.cuda()
        else:
            model.cuda(device_id)
        return model, config, tokenizer

    def _load_lora_model(self, device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()
        if config.pad_token_id is None or config.pad_token_id >= config.vocab_size:
            config.pad_token_id = tokenizer.eos_token_id

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir, 'config.json')):
            config = AutoConfig.from_pretrained(ckpt_dir)
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
            self.lora_model.eval()

            model = self.lora_model
            if hasattr(model,'quantize'):
                if self.auto_quantize:
                    model.half().quantize(4)
                else:
                    model.half()
            else:
                model.half()
        else:
            self.lora_model = self.lora_model.half().eval()

        if device_id is None:
            self.lora_model.cuda()
        else:
            self.lora_model.cuda(device_id)
        return self.lora_model, config, tokenizer

    def chat_stream(self, query, nchar=1,gtype='total', history=None, **kwargs):
        if history is None:
            history = []
        prompt = ""
        for q, a in history:
            prompt += q
            prompt += a
        prompt += query

        default_kwargs = dict(
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.eos_token_id,
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
                else:
                    self.push_response(((chunk.text, history), 0, "ok", False))
                    chunk.clear()

        skip_word_list = [self.tokenizer.eos_token_id]
        streamer = GenTextStreamer(process_token_fn,chunk,tokenizer=self.tokenizer,skip_word_list=skip_word_list,skip_prompt=True)
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
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.eos_token_id,
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
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        response = Generate.generate(self.get_model(),
                                     tokenizer=self.tokenizer,
                                     query=input,**kwargs)
        return response

if __name__ == '__main__':
    api_client = EngineAPI(global_models_info_args['bloom-560m'])
    api_client.init()
    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]
    for input in text_list:
        response = api_client.generate(input)
        print('input', input)
        print('output', response)