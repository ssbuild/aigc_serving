# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/11 15:15


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")


def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or ''
    if history is not None:
        for q, a in history:
            prompt += "用户：{}\n小元：{}".format(q, a)
    prompt += "用户：{}\n小元：".format(query)

    prompt = preprocess(prompt)
    return prompt