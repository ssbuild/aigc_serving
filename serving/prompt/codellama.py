# coding=utf8
# @Time    : 2023/11/30
# @Author  : tk
# @FileName: codellama

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_chat(tokenizer,query,history = None,prefix=None):
    prompt = prefix or ''
    if prompt:
        prompt = B_SYS + prompt + E_SYS

    if history is not None:
        for q, a in history:
            prompt += f"{B_INST} {(q).strip()} {E_INST} {(a).strip()} "
    prompt +=  f"{B_INST} {(query).strip()} {E_INST}"
    return prompt