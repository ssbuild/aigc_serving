# -*- coding: utf-8 -*-
# @Time:  12:24
# @Author: tk
# @File：default

def get_chat(tokenizer,query,history = None):
    if history is None:
        history = []

    tok_ins = ""
    tok_res = ""

    # eos_token =  tokenizer.eos_token or ''
    eos_token = ''
    prompt_text = ''
    for q,a in history:
        prompt_text += tok_ins + q + tok_res + a + eos_token

    prompt_text = prompt_text + tok_ins + query + tok_res
    return prompt_text