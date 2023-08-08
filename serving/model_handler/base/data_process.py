# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/8 16:55
import typing

from transformers import PreTrainedTokenizer

def preprocess_input_args(tokenizer: PreTrainedTokenizer,args_dict: dict):
    stop = args_dict.pop('stop',None)
    if stop is None:
        return args_dict
    eos_token_id = None
    if isinstance(stop,list):
        eos_token_id = []
        for s in stop:
            if s is not None:
                eos_token_id.append(tokenizer.encode(s,add_special_tokens=False))
    elif isinstance(stop,str):
        eos_token_id = tokenizer.encode(stop,add_special_tokens=False)

    if eos_token_id:
        args_dict['eos_token_id'] = eos_token_id
    return args_dict


def flat_input(ids: typing.Union[typing.List,int]):
    if isinstance(ids,int):
        return [ids]
    ids_ = []
    for i in ids:
        if isinstance(i,list):
            ids_.extend(i)
        else:
            ids_.append(i)
    return ids_
