# -*- coding: utf-8 -*-
# @Time:  12:24
# @Author: tk
# @File：default

def get_chat(tokenizer,query,history = None):
    prompt = ''
    if history is not None:
        for q, a in history:
            prompt += "User: {}\nAssistant:{}".format(q, a)
    prompt += "User: {}\nAssistant:".format(query)
    return prompt