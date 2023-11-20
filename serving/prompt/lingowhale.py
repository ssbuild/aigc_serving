# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/20 15:52

def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or ''
    if history is not None:
        for q, a in history:
            prompt += "User: {}\nAssistant:{}".format(q, a)
    prompt += "User: {}\nAssistant:".format(query)
    return prompt