# -*- coding: utf-8 -*-
# @Time:  18:49
# @Author: tk
# @File：infer
import json
import os
import torch
from torch.nn import functional as F
from deep_training.trainer.pl.modelweighter import default_peft_weight_preprocess
from aigc_zoo.utils.streamgenerator import GenTextStreamer
from deep_training.data_helper import ModelArguments,  DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.moss.llm_model import MyTransformer,MossConfig,MossTokenizer,PetlArguments,PetlModel
from aigc_zoo.generator_utils.generator_moss import Generate
from serving.model_handler.base import EngineAPI_Base, flat_input, preprocess_input_args, postprocess_input_args, \
    CompletionResult, LoraModelState, ChunkData, load_lora_config
from config.main import global_models_info_args
class NN_DataHelper(DataHelper):pass


class EngineAPI(EngineAPI_Base):
    def _load_model(self,device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer: MossTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer,
                                                                       config_class_name=MossConfig,
                                                                       config_kwargs={"torch_dtype": "float16"})

        if config.pad_token_id is None or config.pad_token_id >= config.vocab_size:
            config.pad_token_id = tokenizer.eos_token_id
        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16, )
        model = pl_model.get_llm_model()
        model.eval().half()
        if device_id is None:
            model.cuda()
        else:
            model.cuda(device_id)

        self.gen_core = Generate(model,tokenizer)
        return model,config,tokenizer

    def _load_lora_model(self, device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer: MossTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer,
                                                                       config_class_name=MossConfig,
                                                                       config_kwargs={"torch_dtype": "float16"})
        if config.pad_token_id is None or config.pad_token_id >= config.vocab_size:
            config.pad_token_id = tokenizer.eos_token_id

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir, 'config.json')):
            config = MossConfig.from_pretrained(ckpt_dir)
        config.initializer_weight = False
        lora_args,ls_peft = load_lora_config(ckpt_dir)

        assert lora_args.inference_mode == True

        new_num_tokens = config.vocab_size
        if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
            config.vocab_size = config.task_specific_params['vocab_size']

        pl_model = MyTransformer(config=config, model_args=model_args,
                                 lora_args=lora_args,
                                 torch_dtype=torch.float16, new_num_tokens=new_num_tokens,
                                 
                                 # # device_map="auto",
                                 # device_map = {"":0} # 第一块卡
                                 )

        for adapter_name, ckpt_dir in self.lora_conf.items():
            lora_args,ls_peft = load_lora_config(ckpt_dir)
            pl_model.load_sft_weight(ckpt_dir, adapter_name=adapter_name,
                                     lora_config=lora_args,
                                     map_preprocess=default_peft_weight_preprocess if ls_peft else None)
        self.lora_model = pl_model.backbone
        if len(self.lora_conf) == 1:
            if self.auto_merge_lora_single:
                self.lora_state = LoraModelState.MERGE_AND_LOCKED
                self.lora_model.merge_and_unload()
                self.lora_model.eval()
                model = self.lora_model
                if hasattr(model, 'quantize') and self.auto_quantize:
                    model.half().quantize(4)
                else:
                    model.half()
            else:
                self.lora_model = self.lora_model.half().eval()
        else:
            self.lora_model = self.lora_model.half().eval()

        if device_id is None:
            self.lora_model.cuda()
        else:
            self.lora_model.cuda(device_id)
        self.gen_core = Generate(self.lora_model, tokenizer)
        return self.lora_model, config, tokenizer


    def chat_stream(self, query, nchar=1,gtype='total', history=None, **kwargs):
        preprocess_input_args(self.tokenizer,self.config,kwargs)
        prompt = query
        default_kwargs = dict(
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.eos_token_id,)
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
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
                else:
                    self.push_response(ret)
                    chunk.clear()

        skip_word_list = default_kwargs.get('eos_token_id',None) or [self.tokenizer.eos_token_id]
        streamer = GenTextStreamer(process_token_fn,chunk,tokenizer=self.tokenizer,skip_word_list=flat_input(skip_word_list),skip_prompt=True)
        self.gen_core.chat(query=prompt,streamer=streamer,  **default_kwargs)

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
        preprocess_input_args(self.tokenizer,self.config,kwargs)
        prompt = query
        default_kwargs = dict(
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True,
        )
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
        response,history = self.gen_core.chat(prompt,history=history, **default_kwargs)
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
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
        response = self.model.generate(input, **kwargs)
        return response

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

if __name__ == '__main__':
    api_client = EngineAPI(global_models_info_args['moss-moon-003-sft-int4'])
    api_client.init()
    text_list = ["写一个诗歌，关于冬天",
                 "<|Human|>: 如果一个女性想要发展信息技术行业，她应该做些什么？<eoh>\n<|MOSS|>:",
                 ]
    for input in text_list:
        response = api_client.generate(input)
        print('input', input)
        print('output', response)
