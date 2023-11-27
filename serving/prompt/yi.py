# coding=utf8
# @Time    : 2023/10/31 2:34
# @Author  : tk
# @FileName: skywork

def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or ''
    if prompt:
        prompt = f"<|im_start|>system\n{prompt}<|im_end|>\n"
    if history is not None:
        for q, a in history:
            prompt += "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>".format(q, a)
    prompt += "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
    return prompt