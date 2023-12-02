# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/30 10:20




def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or ''
    if history is not None:
        for q, a in history:
            prompt += "User: {}\nAssistant: {}<｜end▁of▁sentence｜>".format(q, a)
    prompt += "User: {}\nAssistant:".format(query)
    return prompt