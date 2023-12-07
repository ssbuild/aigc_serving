# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/30 10:20




def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or 'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n'
    if history is not None:
        for q, a in history:
            prompt += '### Instruction:\n{}\n### Response:\n{}\n<|EOT|>\n'.format(q, a)
    prompt += "### Instruction:\n{}\n### Response:\n".format(query)
    return prompt