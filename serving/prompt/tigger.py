# coding=utf8
# @Time    : 2023/9/3 1:45
# @Author  : tk
# @FileName: tigger


def get_chat(tokenizer,query,history = None):
    if history is None:
        history = []

    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"

    eos_token =  tokenizer.eos_token or ''
    prompt_text = ''
    for q,a in history:
        prompt_text += tok_ins + q  + tok_res + a + eos_token

    prompt_text = prompt_text + tok_ins + query + tok_res
    return prompt_text

