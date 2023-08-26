# -*- coding: utf-8 -*-
# @Time:  18:49
# @Author: tk
# @File：infer
import json
import os
import torch
from torch.nn import functional as F
from deep_training.trainer.pl.modelweighter import default_peft_weight_preprocess
from deep_training.data_helper import ModelArguments, DataHelper
from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig
from aigc_zoo.model_zoo.baichuan2.llm_model import MyTransformer,BaichuanConfig,BaichuanTokenizer,\
    MyBaichuanForCausalLM,PetlArguments,PetlModel
from aigc_zoo.utils.llm_generate import Generate
from serving.model_handler.base import EngineAPI_Base, flat_input, LoraModelState, load_lora_config
from config.main import global_models_info_args
from serving.model_handler.base import CompletionResult,ChunkData,preprocess_input_args,postprocess_input_args


class NN_DataHelper(DataHelper):pass



def _build_message(query,history= None):
    if history is None:
        history = []
    messages = []
    for q, a in history:
        messages.append({
            "role": "user",
            "content": q
        })
        messages.append({
            "role": "assistant",
            "content": a
        })

    messages.append({
        "role": "user",
        "content": query
    })
    return messages

class EngineAPI(EngineAPI_Base):
    def _load_model(self,device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=BaichuanConfig,
                                                                       tokenizer_class_name=BaichuanTokenizer)
        config.pad_token_id = config.eos_token_id

        pl_model = MyTransformer(config=config, model_args=model_args,
                                 torch_dtype=torch.float16, )

        model: MyBaichuanForCausalLM = pl_model.get_llm_model()
        model = model.eval()
        model.requires_grad_(False)

        if not model.quantized:
            # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
            if self.auto_quantize:
                model.half().quantize(4)
            else:
                model.half()
        else:
            # 已经量化
            model.half()

        if device_id is None:
            model.cuda()
        else:
            model.cuda(device_id)

        return model,config,tokenizer



    def _load_lora_model(self, device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=BaichuanConfig,
                                                                       tokenizer_class_name=BaichuanTokenizer)
        config.pad_token_id = config.eos_token_id

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir, 'config.json')):
            config = BaichuanConfig.from_pretrained(ckpt_dir)
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
        self.lora_state = LoraModelState.NONE
        if len(self.lora_conf) == 1:
            if self.auto_merge_lora_single:
                self.lora_state = LoraModelState.MERGE_AND_LOCKED
                self.lora_model.merge_and_unload()
                self.lora_model.eval()
                model = self.lora_model
                if hasattr(model,'quantize') and self.auto_quantize:
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
        return self.lora_model, config, tokenizer

    @torch.no_grad()
    def _generate(self,  query: str,do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        gen_kwargs = {"do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        output_scores = gen_kwargs.get('output_scores', False)
        if output_scores:
            gen_kwargs['return_dict_in_generate'] = True
        # prompt = "Human：" + query + "\nAssistant："
        # 自行加模板
        prompt = query
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        if output_scores:
            score = outputs.scores[0]
            return score
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)
        return response

    def chat_stream(self,  query, nchar=1,gtype='total', history=None,**kwargs):
        preprocess_input_args(self.tokenizer,self.config,kwargs)

        messages = _build_message(query,history=history)
        default_kwargs = dict(eos_token_id=self.model.config.eos_token_id,
                              pad_token_id=self.model.config.eos_token_id,
                              do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                              repetition_penalty=1.1,
                              )
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
        stopping_criteria = default_kwargs.pop('stopping_criteria', None)
        generation_config = GenerationConfig(**default_kwargs)


        chunk = ChunkData()
        chunk.idx = 0
        n_id = 0

        response = None
        for response in self.get_model().chat(tokenizer=self.tokenizer,
                                              messages=messages,
                                              stream=True,
                                              generation_config=generation_config,
                                              stopping_criteria=stopping_criteria):
            n_id += 1
            chunk.text = response
            if n_id % nchar == 0:
                if gtype == 'total':
                    yield CompletionResult(result={
                        "response": chunk.text,
                        "history": history,
                        "num_token": n_id
                    },complete=False)
                else:
                    yield CompletionResult(result={
                        "response": chunk.text[chunk.idx:],
                        "history": history,
                        "num_token": n_id
                    },complete=False)
                    chunk.idx = len(response)

        history = history + [(query, response)]

        if gtype != 'total' and chunk.idx != len(chunk.text):
            yield CompletionResult(result={
                "response": chunk.text[chunk.idx:],
                "history": history,
                "num_token": n_id
            },complete=False)


    def chat(self, query, history=None, **kwargs):
        preprocess_input_args(self.tokenizer,self.config,kwargs)

        messages = _build_message(query, history=history)

        default_kwargs = dict(eos_token_id=self.model.config.eos_token_id,
                              pad_token_id=self.model.config.eos_token_id,
                              do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                              repetition_penalty=1.1,
                              )
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
        stopping_criteria = default_kwargs.pop('stopping_criteria', None)
        generation_config = GenerationConfig(**default_kwargs)
        response = self.get_model().chat(tokenizer=self.tokenizer,
                                         messages=messages,
                                         generation_config=generation_config,
                                         stopping_criteria=stopping_criteria)
        history = history + [(query, response)]
        return CompletionResult(result={
            "response": response,
            "history": history
        })



    def generate(self,input,**kwargs):
        default_kwargs = dict(eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True, top_k=5,top_p=0.85, temperature=0.3,
            repetition_penalty=1.1,
        )
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
        response = Generate.generate(self.get_model(),
                                     tokenizer=self.tokenizer,
                                     query=input,**default_kwargs)
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
    api_client = EngineAPI(global_models_info_args['Baichuan-13B-Chat'])
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
