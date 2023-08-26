# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/10 17:24
import json
import os
import torch
from torch.nn import functional as F
from deep_training.trainer.pl.modelweighter import default_peft_weight_preprocess
from aigc_zoo.utils.streamgenerator import GenTextStreamer
from deep_training.data_helper import ModelArguments,DataHelper
from transformers import HfArgumentParser, BitsAndBytesConfig
from aigc_zoo.model_zoo.qwen.llm_model import MyTransformer, QWenTokenizer, PetlArguments, \
    setup_model_profile, QWenConfig
from serving.model_handler.base import EngineAPI_Base, flat_input, LoraModelState, load_lora_config
from config.main import global_models_info_args
from serving.model_handler.base import CompletionResult,ChunkData,preprocess_input_args,postprocess_input_args


class NN_DataHelper(DataHelper):pass


class EngineAPI(EngineAPI_Base):

    def _load_model(self,device_id=None):
        
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)
        setup_model_profile()
        dataHelper = NN_DataHelper(model_args)
        tokenizer: QWenTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
            tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # # quantization configuration for Int8 (8 bits)
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        pl_model = MyTransformer(config=config, model_args=model_args,
                                 torch_dtype=torch.float16,
                                 # device_map="cuda:{}".format(device_id if device_id is None else 0),
                                 # quantization_config=quantization_config,
                                 )
        model = pl_model.get_llm_model()
        model = model.eval()
        # if hasattr(model, 'is_loaded_in_4bit') or hasattr(model, 'is_loaded_in_8bit'):
        #     model.eval().cuda()
        # else:
        #     model.half().eval().cuda()

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


    def _load_lora_model(self,device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        setup_model_profile()

        dataHelper = NN_DataHelper(model_args)
        tokenizer: QWenTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
            tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir,'config.json')):
            config = QWenConfig.from_pretrained(ckpt_dir)
        config.initializer_weight = False
        lora_args = load_lora_config(ckpt_dir)

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
            lora_args = load_lora_config(ckpt_dir)
            pl_model.load_sft_weight(ckpt_dir, adapter_name=adapter_name, lora_config=lora_args,map_preprocess=default_peft_weight_preprocess)
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
        return self.lora_model, config, tokenizer


    def chat_stream(self, query, nchar=1,gtype='total', history=None, **kwargs):
        preprocess_input_args(self.tokenizer,self.config,kwargs)
        if history is None:
            history = []


        default_kwargs = {
            "history": [],
            "chat_format": "chatml",
            "eos_token_id": 151643,
            "max_new_tokens": 512,
            "pad_token_id": 151643,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.5,
        }
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)

        chunk = ChunkData()
        chunk.n_id = 0
        def process_token_fn(text, stream_end, chunk: ChunkData):
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

        skip_word_list = [self.tokenizer.im_end_id, self.tokenizer.im_start_id, self.tokenizer.eos_token_id or 151643]
        skip_word_list += default_kwargs.get('stop_words_ids',[])
        streamer = GenTextStreamer(process_token_fn, chunk, tokenizer=self.tokenizer,skip_word_list=flat_input(skip_word_list),skip_prompt=True)
        _ = self.get_model().chat(tokenizer=self.tokenizer, streamer=streamer, query=query, **default_kwargs)
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



    def chat(self, query, **kwargs):
        preprocess_input_args(self.tokenizer,self.config,kwargs)
        default_kwargs = {
            "history": [],
            "chat_format": "chatml",
            "eos_token_id": 151643,
            "max_new_tokens": 512,
            "pad_token_id": 151643,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.5,
        }
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
        response, history = self.model.chat(self.tokenizer, query=query,  **default_kwargs)
        return CompletionResult(result={
            "response": response,
            "history": history
        })

    def generate(self,input,**kwargs):
        default_kwargs = dict(history=[], 
            eos_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        postprocess_input_args(self.tokenizer,self.config,default_kwargs)
        #response, history = self.model.chat(self.tokenizer, query=input,  **kwargs)
        output = self.model.chat(self.tokenizer, query=input, **default_kwargs)
        output_scores = default_kwargs.get('output_scores', False)
        if output_scores:
            return output
        response, history = output
        return response

    def embedding(self, query, **kwargs):
        from deep_training.nlp.models.qwen.modeling_qwen import QWenLMHeadModel
        model: QWenLMHeadModel = self.get_model()
        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = inputs.to(model.device)
        model_output = model.forward(**inputs,return_dict=True, output_hidden_states=True, **kwargs)
        data = model_output.hidden_states[-1]
        data = F.normalize(torch.mean(data, dim=1), p=2, dim=1)
        embedding = data.detach().tolist()
        return CompletionResult(result={
            "response": embedding,
        })

# if __name__ == '__main__':
#     api_client = EngineAPI(global_models_info_args['chatglm2-6b-int4'])
#     api_client.init()
#     text_list = [
#         "写一个诗歌，关于冬天",
#         "晚上睡不着应该怎么办",
#     ]
#     for input in text_list:
#         response = api_client.generate(input)
#         print("input", input)
#         print("response", response)