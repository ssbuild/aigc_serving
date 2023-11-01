# coding=utf8
# @Time    : 2023/11/1 22:37
# @Author  : tk
# @FileName: bluelm


def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or ''
    if history is not None:
        for q, a in history:
            prompt += "[|Human|]:{}[|AI|]:{}".format(q, a)
    prompt += "[|Human|]:{}[|AI|]:".format(query)
    return prompt