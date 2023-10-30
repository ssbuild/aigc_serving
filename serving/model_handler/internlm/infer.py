# -*- coding: utf-8 -*-
# @Time:  18:49
# @Author: tk
# @File：infer
import json
import os
from typing import Dict, List

import torch
from torch.nn import functional as F
from deep_training.trainer.pl.modelweighter import default_peft_weight_preprocess
from deep_training.data_helper import ModelArguments, DataHelper
from deep_training.nlp.layers.rope_scale.patch import RotaryNtkScaledArguments
from transformers import HfArgumentParser, BitsAndBytesConfig
from aigc_zoo.model_zoo.internlm.llm_model import MyTransformer,InternLMConfig,InternLMTokenizer,\
    InternLMForCausalLM,PetlArguments,PetlModel
from serving.model_handler.base import EngineAPI_Base, CompletionResult,LoraModelState, load_lora_config, \
    GenerateProcess, WorkMode
from serving.prompt import *

class NN_DataHelper(DataHelper):pass


class EngineAPI(EngineAPI_Base):
    def _load_model(self,device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=InternLMConfig,
                                                                       tokenizer_class_name=InternLMTokenizer)
        config.pad_token_id = config.eos_token_id

        if self.ntk_scale > 1:
            rope_args = RotaryNtkScaledArguments(max_position_embeddings=config.max_position_embeddings,
                                                 alpha=self.ntk_scale)
        else:
            rope_args = None

        pl_model = MyTransformer(config=config, model_args=model_args,
                                 torch_dtype=torch.float16, rope_args=rope_args)

        model = pl_model.get_llm_model()
        model = model.eval()
        model.requires_grad_(False)

        if not self.is_config_quarted(config):
            if self.auto_quantize and hasattr(model,'quantize') and not model.quantized:
                # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
                model.half().quantize(4)
            else:
                # 已经量化
                model.half()

        if self.work_mode != WorkMode.ACCELERATE:
            if device_id is None:
                model.cuda()
            else:
                model.cuda(device_id)
        return model,config,tokenizer



    def _load_model_lora(self, device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=InternLMConfig,
                                                                       tokenizer_class_name=InternLMTokenizer)
        config.pad_token_id = config.eos_token_id

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir, 'config.json')):
            config = InternLMConfig.from_pretrained(ckpt_dir)
        config.initializer_weight = False
        lora_args,ls_peft = load_lora_config(ckpt_dir)

        assert lora_args.inference_mode == True

        new_num_tokens = config.vocab_size
        if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
            config.vocab_size = config.task_specific_params['vocab_size']

        if self.ntk_scale > 1:
            rope_args = RotaryNtkScaledArguments(max_position_embeddings=config.max_position_embeddings,
                                                 alpha=self.ntk_scale)
        else:
            rope_args = None
        pl_model = MyTransformer(config=config, model_args=model_args,
                                 lora_args=lora_args,
                                 torch_dtype=torch.float16, new_num_tokens=new_num_tokens,
                                 rope_args=rope_args,
                                 
                                 # # device_map="auto",
                                 # device_map = {"":0} # 第一块卡
                                 )

        for adapter_name, ckpt_dir in self.lora_conf.items():
            lora_args,ls_peft = load_lora_config(ckpt_dir)
            pl_model.load_sft_weight(ckpt_dir, adapter_name=adapter_name,
                                     lora_config=lora_args,
                                     map_preprocess=default_peft_weight_preprocess if ls_peft else None)
        self.lora_model = pl_model.backbone.eval()
        self.lora_state = LoraModelState.NONE
        if not self.is_config_quarted(config):
            if len(self.lora_conf) == 1:
                if self.auto_merge_lora_single:
                    self.lora_state = LoraModelState.MERGE_AND_LOCKED
                    self.lora_model.merge_and_unload()
                    model = self.lora_model
                    if hasattr(model,'quantize') and self.auto_quantize:
                        model.half().quantize(4)
                    else:
                        model.half()
                else:
                    self.lora_model = self.lora_model.half()
            else:
                self.lora_model = self.lora_model.half()

        if self.work_mode != WorkMode.ACCELERATE:
            if device_id is None:
                self.lora_model.cuda()
            else:
                self.lora_model.cuda(device_id)
        return self.lora_model, config, tokenizer

    def get_default_gen_args(self):
        default_kwargs = dict(eos_token_id=[2, 103028],
                              do_sample=True,
                              temperature=0.8,
                              top_p=0.8,
                              repetition_penalty=1.01, )
        return default_kwargs

    def chat_stream(self, messages: List[Dict], **kwargs):
        args_process = GenerateProcess(self,is_stream=True)
        args_process.preprocess(kwargs)
        default_kwargs = self.get_default_gen_args()
        default_kwargs.update(kwargs)
        args_process.postprocess(default_kwargs)
        query, history = args_process.get_chat_info(messages)
        skip_word_list = [self.tokenizer.eos_token_id,2, 103028]
        streamer = args_process.get_streamer(skip_word_list)
        self.get_model().chat( tokenizer=self.tokenizer, streamer=streamer, query=query,history=history, **default_kwargs)
        args_process.do_final_stream()
        return None


    def chat(self,messages: List[Dict], **kwargs):
        args_process = GenerateProcess(self)
        args_process.preprocess(kwargs)
        default_kwargs = self.get_default_gen_args()
        default_kwargs.update(kwargs)
        args_process.postprocess(default_kwargs)
        query, history = args_process.get_chat_info(messages)
        response, history = self.model.chat(self.tokenizer, query=query,history=history, **default_kwargs)
        response = args_process.postprocess_response(response, **kwargs)
        return CompletionResult(result={
            "response": response,
            #"history": history
        })

    def generate(self,messages: List[Dict],**kwargs):
        args_process = GenerateProcess(self)
        default_kwargs = self.get_default_gen_args()
        default_kwargs.update(kwargs)
        args_process.postprocess(default_kwargs)
        query = args_process.get_chat_info(messages,chat_format="generate")
        output,_ = self.model.chat(self.tokenizer, query=query **default_kwargs)
        output_scores = default_kwargs.get('output_scores', False)
        if output_scores:
            return output
        response, history = output
        return CompletionResult(result={
            "response": response,
            #"history": history
        })

    def embedding(self, query, **kwargs):
        model = self.get_model()
        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = inputs.to(model.device)
        model_output = model.forward(**inputs,return_dict=True, output_hidden_states=True, **kwargs)
        data = model_output.hidden_states[-1]
        data = F.normalize(torch.mean(data, dim=1), p=2, dim=1)
        embedding = data.detach().tolist()
        return CompletionResult(result={
            "response": embedding,
        })
