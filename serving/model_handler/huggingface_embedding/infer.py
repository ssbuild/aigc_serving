# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/21 14:02

import functools
import importlib
import json
import os
import types
from collections import OrderedDict
from typing import List, Dict, Union, Optional, Callable
import numpy as np
import torch
from numpy import ndarray
from sentence_transformers.models import Pooling
from torch import Tensor
from torch.nn import functional as F
from deep_training.trainer.pl.modelweighter import default_peft_weight_preprocess
from deep_training.data_helper import ModelArguments, DataHelper
from deep_training.nlp.layers.rope_scale.patch import RotaryNtkScaledArguments
from transformers import HfArgumentParser, AutoModel, PreTrainedTokenizer
from aigc_zoo.model_zoo.auto.llm_model import MyTransformer,AutoConfig, PetlArguments, PetlModel
from serving.model_handler.base import ModelEngine_Base, CompletionResult, LoraModelState, load_lora_config, GenArgs, \
    WorkMode
from serving.prompt import *


class NN_DataHelper(DataHelper): pass


def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)



global_forward_fn: Optional[Callable] = None
def forward_new_fn(self,features):
    """Returns token_embeddings, cls_token"""
    trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
    if 'token_type_ids' in features:
        trans_features['token_type_ids'] = features['token_type_ids']

    output_states = global_forward_fn(**trans_features, return_dict=False)
    output_tokens = output_states[0]

    features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

    # if self.config.output_hidden_states:
    #     all_layer_idx = 2
    #     if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
    #         all_layer_idx = 1
    #
    #     hidden_states = output_states[all_layer_idx]
    #     features.update({'all_layer_embeddings': hidden_states})

    return features


class ModelEngine(ModelEngine_Base):

    def _load_sub_module(self,model):
        global global_forward_fn
        global_forward_fn = model.forward
        model.forward = types.MethodType(forward_new_fn,model)

        model_dir = self.model_config_dict["model_config"]["model_name_or_path"]
        modules_path = os.path.join(model_dir, 'modules.json')




        is_sub_module_empty = False
        if not os.path.exists(modules_path):
            is_sub_module_empty = True
        else:

            with open(modules_path) as fIn:
                modules_config = json.load(fIn)

            if len(modules_config) == 1:
                is_sub_module_empty = True

        device = model.device
        config = model.config
        modules = (model,)
        if is_sub_module_empty:
            pooling_model = Pooling(model.config.hidden_size, 'mean')
            modules += (pooling_model,)
        else:
            for module_config in modules_config:
                mtype = module_config['type'].strip()
                if '.Transformer' in mtype:
                    continue
                module_class = import_from_string(module_config['type'])
                module = module_class.load(os.path.join(model_dir, module_config['path']))
                modules += (module,)

        model = torch.nn.Sequential(*modules)
        model.device = device
        model.config = config
        return model
    def _load_model(self, device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()

        if self.ntk_scale > 1:
            rope_args = RotaryNtkScaledArguments(max_position_embeddings=config.max_sequence_length,
                                                 alpha=self.ntk_scale)
        else:
            rope_args = None


        pl_model = MyTransformer(model_class=AutoModel,config=config, model_args=model_args, torch_dtype=torch.float16, rope_args=rope_args)
        model = pl_model.get_llm_model()
        model = self._load_sub_module(model)
        model = model.eval()

        if not self.is_config_quarted(config):
            if self.auto_quantize and hasattr(model, 'quantize') and not model.quantized:
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

        try:
            model.device = torch.device(device_id or torch.cuda.current_device())
        except AttributeError:
            pass

        return model, config, tokenizer

    def _load_model_lora(self, device_id=None):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()

        ckpt_dir = list(self.lora_conf.values())[0]
        if os.path.exists(os.path.join(ckpt_dir, 'config.json')):
            config = AutoConfig.from_pretrained(ckpt_dir)
        config.initializer_weight = False
        lora_args, ls_peft = load_lora_config(ckpt_dir)

        assert lora_args.inference_mode == True

        new_num_tokens = config.vocab_size
        if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
            config.vocab_size = config.task_specific_params['vocab_size']

        if self.ntk_scale > 1:
            rope_args = RotaryNtkScaledArguments(max_position_embeddings=config.max_sequence_length,
                                                 alpha=self.ntk_scale)
        else:
            rope_args = None
        pl_model = MyTransformer(model_class=AutoModel,config=config, model_args=model_args,
                                 lora_args=lora_args,
                                 torch_dtype=torch.float16, new_num_tokens=new_num_tokens,
                                 rope_args=rope_args,
                                 # # device_map="auto",
                                 # device_map = {"":0} # 第一块卡
                                 )

        for adapter_name, ckpt_dir in self.lora_conf.items():
            lora_args, ls_peft = load_lora_config(ckpt_dir)
            pl_model.load_sft_weight(ckpt_dir, adapter_name=adapter_name, lora_config=lora_args,
                                     map_preprocess=default_peft_weight_preprocess if ls_peft else None)
        self.lora_model = pl_model.backbone.eval()
        self.lora_state = LoraModelState.NONE
        if not self.is_config_quarted(config):
            if len(self.lora_conf) == 1:
                if self.auto_merge_lora_single:
                    self.lora_state = LoraModelState.MERGE_AND_LOCKED
                    self.lora_model.merge_and_unload()
                    model = self.lora_model
                    if hasattr(model, 'quantize') and self.auto_quantize:
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

        self.lora_model = self._load_sub_module(self.lora_model)

        try:
            self.lora_model.device = torch.device(device_id or torch.cuda.current_device())
        except AttributeError:
            pass

        return self.lora_model, config, tokenizer


    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings

    def _encode(self,
                model,
                sentences: Union[str, List[str]],
                batch_size: int = 32,
                output_value: str = 'sentence_embedding',
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False,
                max_tokens= None,
                device: str = None,
                normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True


        if device is None:
            device = model.device

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            self.tokenizer: PreTrainedTokenizer
            features = self.tokenizer(sentences_batch,truncation=True,max_length=max_tokens,return_tensors="pt")
            features = features.to(device=device)
            with torch.no_grad():
                out_features = model(features)
                embeddings = out_features[output_value]
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
    def embedding(self, query,max_tokens=None, **kwargs):
        model = self.get_model()
        emb = self._encode(model,sentences=query,max_tokens=max_tokens,normalize_embeddings=True)
        emb = emb.tolist()
        return CompletionResult(result={
            "response": emb,
        })
