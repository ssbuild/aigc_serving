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
from aigc_zoo.model_zoo.baichuan.baichuan_13b.llm_model import MyTransformer,BaichuanConfig,BaichuanTokenizer,\
    MyBaichuanForCausalLM,PetlArguments,PetlModel
from serving.model_handler.base import EngineAPI_Base,CompletionResult,flat_input, LoraModelState, load_lora_config, GenerateProcess,WorkMode



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

        if self.work_mode != WorkMode.ACCELERATE:
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

    def chat_stream(self, query, history=None, **kwargs):
        args_process = GenerateProcess(self.tokenizer, self.config,is_stream=True)
        args_process.preprocess(kwargs)
        chunk = args_process.chunk

        messages = _build_message(query,history=history)
        default_kwargs = dict(eos_token_id=self.model.config.eos_token_id,
                              pad_token_id=self.model.config.eos_token_id,
                              do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                              repetition_penalty=1.1,
                              )
        default_kwargs.update(kwargs)
        args_process.postprocess(default_kwargs)
        stopping_criteria = default_kwargs.pop('stopping_criteria', None)
        generation_config = GenerationConfig(**default_kwargs)

        response = None
        for response in self.get_model().chat(tokenizer=self.tokenizer,
                                              messages=messages,
                                              stream=True,
                                              generation_config=generation_config,
                                              stopping_criteria=stopping_criteria):
            chunk.step(response)
            if chunk.can_output():
                text = chunk.step_text()
                yield CompletionResult(result={
                    "response": text,
                    #"history": history,
                    "num_token": chunk.n_id
                }, complete=False)

        # history = history + [(query, response)]
        text = chunk.final_text()
        if text is not None:
            yield CompletionResult(result={
                "response": text,
                #"history": history,
                "num_token": chunk.n_id
            }, complete=False)


    def chat(self, query, history=None, **kwargs):
        args_process = GenerateProcess(self.tokenizer, self.config)
        args_process.preprocess(kwargs)
        messages = _build_message(query, history=history)
        default_kwargs = dict(eos_token_id=self.model.config.eos_token_id,
                              pad_token_id=self.model.config.eos_token_id,
                              do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                              repetition_penalty=1.1,
                              )
        default_kwargs.update(kwargs)
        args_process.postprocess(default_kwargs)
        stopping_criteria = default_kwargs.pop('stopping_criteria', None)
        generation_config = GenerationConfig(**default_kwargs)
        response = self.get_model().chat(tokenizer=self.tokenizer,
                                         messages=messages,
                                         generation_config=generation_config,
                                         stopping_criteria=stopping_criteria)
        response = args_process.postprocess_response(response, **kwargs)
        # history = history + [(query, response)]
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
