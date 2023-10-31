# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/10/30 10:07
import copy
import json
from typing import Tuple, Union, List
from serving.openai_api.openai_api_protocol import ChatMessage

__all__ = [
    'get_react_prompt_for_chatglm3'
]

_TOOL_INFO = "Answer the following questions as best as you can. You have access to the following tools:"




def get_react_prompt_for_chatglm3(messages: List[ChatMessage], functions=None, function_call="auto"):
    if functions is not None:
        if "name" in functions[0]:
            new_function = []
            for info in functions:
                new_info = {}
                new_info["name"] = info.get("name","")
                new_info["description"] = info.get("description", "")
                new_info["parameters"] = {}
                new_info["parameters"]["type"] = info["parameters"].get("type", "")
                new_info["parameters"]["required"] = info["parameters"].get("required", [])
                properties = info["parameters"]["properties"]

                for name, p in properties.items():
                    new_info["parameters"][name] = p

                new_function.append(new_info)
            functions = new_function

    else:
        for message in messages:
            if message.functions:
                functions = message.functions
                break

    if function_call != "auto" and isinstance(function_call, dict):
        functions = [info for info in functions if info["name"] in [function_call["name"]]]

    tool_descs = []
    for info in functions:
        tool_descs.append({
            "name": info["name"],
            "description": info["description"],
            "parameters": info["parameters"]
        })
    sys_info = copy.deepcopy(_TOOL_INFO)
    sys_info += "\n" + json.dumps(tool_descs,ensure_ascii=False,indent=4)
    messages_ = [ChatMessage(role="system",content=sys_info)]
    for message in messages:
        role, content = message.role, message.content
        # if role == "assistant":
        #     for response in content.split("<|assistant|>"):
        #         messages_.append(ChatMessage(role=role, content=response))
        # else:
        #     messages_.append(ChatMessage(role=role,content=content))
        messages_.append(ChatMessage(role=role, content=content))
    return messages_, functions