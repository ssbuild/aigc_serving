# -*- coding: utf-8 -*-
# @Time:  14:03
# @Author: tk
# @File：chatglm

def get_chat(query, history=None):
    prompt = ''
    sid = 0
    if history is not None:
        for q, a in history:
            prompt += "[Round {}]\n问：{}\n答：{}".format(sid,q, a)
            sid += 1
    prompt += query if sid == 0 else "[Round {}]\n问：{}\n答：".format(sid, query)
    return prompt