# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/10 17:24
import copy
import json
import os
from typing import Dict, List

import torch
from torch.nn import functional as F
from deep_training.trainer.pl.modelweighter import default_peft_weight_preprocess
from deep_training.data_helper import ModelArguments,DataHelper
from deep_training.nlp.layers.rope_scale.patch import RotaryNtkScaledArguments
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.chatglm3.llm_model import MyTransformer, ChatGLMTokenizer, PetlArguments, \
    setup_model_profile, ChatGLMConfig
from serving.model_handler.base import ModelEngine_Base,CompletionResult,LoraModelState, load_lora_config, GenArgs,WorkMode
from serving.prompt import *


class NN_DataHelper(DataHelper):pass


def _preprocess_messages_for_chatglm3(messages: List[Dict]):
    item = messages.pop(-1)
    role,query = item["role"],item["content"]
    history = []
    for message in messages:
        if message["role"] == "assistant":
            output = message["content"]
            for response in output.split("<|assistant|>"):
                item = response.split("\n", maxsplit=1)
                # 支持自定义历史记录 ， 例如 aigc_evals history
                if len(item) == 2:
                    metadata, content = item
                else:
                    metadata, content = "", item[0]

                if not metadata.strip():
                    content = content.strip()
                    history.append({"role": "assistant", "metadata": metadata, "content": content})
                else:
                    history.append({"role": "assistant", "metadata": metadata, "content": content})
        else:
            history.append(copy.copy(message))
    return role,query,history


class ModelEngine(ModelEngine_Base):

    def _load_model(self,device_id=None):
        
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)
        setup_model_profile()
        dataHelper = NN_DataHelper(model_args)
        tokenizer: ChatGLMTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
            tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

        if self.ntk_scale > 1:
            rope_args = RotaryNtkScaledArguments(model_type='chatglm2',name='rotary_pos_emb',
                                                 max_position_embeddings=2048,
                                                 alpha=self.ntk_scale)
        else:
            rope_args = None

        is_enbale_ptv2 = False
        if (getattr(config,"pre_seq_len",0) or 0) > 0:
            is_enbale_ptv2 = True
        if is_enbale_ptv2:
            model_name_or_path = model_args.model_name_or_path
            model_args.model_name_or_path = None

        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,rope_args=rope_args )

        if is_enbale_ptv2:
            model_args.model_name_or_path = model_name_or_path
            assert os.path.isdir(model_name_or_path)
            # 加载微调权重
            pl_model.load_sft_weight(os.path.join(model_name_or_path,"pytorch_model.bin"), strict=False)

        model = pl_model.get_llm_model()
        model = model.eval()

        if not self.is_config_bnb(config) and not self.is_config_awq(config):
            if not is_enbale_ptv2 and self.auto_quantize and hasattr(model,'quantize') and not model.quantized:
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



    def _load_model_lora(self,device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        setup_model_profile()

        dataHelper = NN_DataHelper(model_args)
        tokenizer: ChatGLMTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLMTokenizer,
                                                                       config_class_name=ChatGLMConfig)

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir,'config.json')):
            config = ChatGLMConfig.from_pretrained(ckpt_dir)
        config.initializer_weight = False
        lora_args,ls_peft = load_lora_config(ckpt_dir)

        assert lora_args.inference_mode == True and config.pre_seq_len is None

        new_num_tokens = config.vocab_size
        if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
            config.vocab_size = config.task_specific_params['vocab_size']

        if self.ntk_scale > 1:
            rope_args = RotaryNtkScaledArguments(model_type='chatglm2',name='rotary_pos_emb',
                                                 max_position_embeddings=2048,
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
        if not self.is_config_bnb(config) and not self.is_config_awq(config):
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
        default_kwargs = dict(do_sample=True, top_p=0.8, temperature=0.8,)
        return default_kwargs

    def chat_stream(self, messages: List[Dict], **kwargs):
        args_process = GenArgs(kwargs, self, is_stream=True)
        chunk = args_process.chunk
        default_kwargs = self.get_default_gen_args()
        default_kwargs.update(kwargs)
        args_process.build_args(default_kwargs)
        role,query,history = _preprocess_messages_for_chatglm3(messages)
        for response, history in self.get_model().stream_chat(self.tokenizer, query=query,role=role,history=history,with_postprocess=False,**default_kwargs):
            chunk.step(response)
            if chunk.can_output():
                text = chunk.step_text()
                yield CompletionResult(result={
                    "response": text,
                    #"history": history,
                    "num_token": args_process.get_num_tokens()
                }, complete=False)

        text = chunk.final_text()
        if text is not None:
            yield CompletionResult(result={
                "response": text,
                #"history": history + [(query, response)],
                "num_token": args_process.get_num_tokens()
            }, complete=False)

    def chat(self,messages: List[Dict], **kwargs):
        args_process = GenArgs(kwargs, self)
        default_kwargs = self.get_default_gen_args()
        default_kwargs.update(kwargs)
        args_process.build_args(default_kwargs)
        role, query, history = _preprocess_messages_for_chatglm3(messages)
        response, history = self.model.chat(self.tokenizer, query=query,role=role,history=messages,with_postprocess=False,**default_kwargs)
        if isinstance(response,str):
            response = args_process.postprocess_response(response)
        else:
            response = json.dumps(response,ensure_ascii=True)

        return CompletionResult(result={
            "response": response,
            #"history": history
        })

    def generate(self,messages: List[Dict],**kwargs):
        args_process = GenArgs(kwargs, self)
        default_kwargs = self.get_default_gen_args()
        default_kwargs.update(kwargs)
        args_process.build_args(default_kwargs)
        query = messages[0]["content"]
        output,_ = self.model.chat(self.tokenizer,query=query,with_postprocess=False, **default_kwargs)
        output_scores = default_kwargs.get('output_scores', False)
        if output_scores:
            return output
        response, history = output
        return CompletionResult(result={
            "response": response,
            #"history": history
        })

    def embedding(self, query,max_tokens=None, **kwargs):
        model = self.get_model()
        inputs = self.tokenizer(query, truncation=True,max_length=max_tokens, return_tensors="pt")
        inputs = inputs.to(model.device)
        model_output = model.forward(**inputs,return_dict=True, output_hidden_states=True, **kwargs)
        data = model_output.hidden_states[-1].transpose(0, 1)
        data = F.normalize(torch.mean(data, dim=1), p=2, dim=1)
        embedding = data.detach().tolist()
        return CompletionResult(result={
            "response": embedding,
        })

