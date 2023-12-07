# coding=utf8
# @Time    : 2023/12/7 8:34
# @Author  : tk
# @FileName: sus




def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or ''
    if history is not None:
        for q, a in history:
            prompt += "### Human: {}\n\n### Assistant: {}".format(q, a)
    prompt += "### Human: {}\n\n### Assistant: ".format(query)
    return prompt