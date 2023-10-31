# -*- coding: utf-8 -*-
# @Time:  21:11
# @Author: tk
# @File：api_react
import json
from typing import Union, Optional
from serving.openai_api.openai_api_protocol import CompletionRequest, ChatCompletionRequest, ChatFunctionCallResponse
from serving.react.qwen.react_prompt import get_react_prompt_for_qwen, parse_qwen_plugin_call
from serving.react.chatglm3.react_prompt import get_react_prompt_for_chatglm3

__all__ = [
    'build_react_functions',
    "parse_tools_functions",
]

def build_react_functions(request: Union[CompletionRequest,ChatCompletionRequest]):
    if request.model_type not in ["qwen","chatglm","chatglm3"]:
        return None

    function_call = request.function_call
    functions = request.functions
    messages = request.messages
    use_function = False
    if isinstance(messages, list) and isinstance(messages[0], dict):
        for message in messages:
            if message.functions:
                use_function = True
                break
    model_type = request.model_type
    if functions is not None or use_function:
        if model_type == "qwen":
            request.messages, functions = get_react_prompt_for_qwen(messages, functions, function_call)
        elif model_type in ["chatglm","chatglm3"]:
            request.messages, functions = get_react_prompt_for_chatglm3(messages, functions, function_call)

    return functions




def parse_tools_functions(request: Union[CompletionRequest,ChatCompletionRequest],
                           functions,
                           context_text)-> Optional[ChatFunctionCallResponse]:
    model_type = request.model_type
    if model_type == "qwen":
        return _process_qwen_function(request,functions,context_text)
    elif model_type in ["chatglm", "chatglm3"]:
        return _process_chatglm3_function(request,functions,context_text)
    return None
def _process_qwen_function(request: Union[CompletionRequest,ChatCompletionRequest],
                           functions,
                           context_text)-> Optional[ChatFunctionCallResponse]:
    function_call = None
    if "Thought:" in context_text:
        react_res = parse_qwen_plugin_call(context_text)
        if react_res is not None:
            # if plugin_name contains other str
            available_functions = [ f.get("name", None) or f.get("name_for_model", None) for f in functions ]
            plugin_name = react_res[ 1 ]
            if plugin_name not in available_functions:
                for fct in available_functions:
                    if fct in plugin_name:
                        plugin_name = fct
                        break

            function_call = ChatFunctionCallResponse(
                thought=react_res[ 0 ],
                name=plugin_name,
                arguments=react_res[ 2 ],
            )
    return function_call

def _process_chatglm3_function(request: Union[CompletionRequest,ChatCompletionRequest],
                               functions,
                               context_text)-> Optional[ChatFunctionCallResponse]:
    thought= ""
    for response in context_text.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            content = content.replace("[[训练时间]]", "2023年")
            thought += content
        else:
            if request.messages[0].role == "system":
                content = "\n".join(content.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs
                parameters = eval(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                content = {"name": metadata.strip(), "content": content}

    jd = content
    if not isinstance(jd,dict) or ("parameters" not in jd and not isinstance(jd["parameters"], dict)):
        function_call = None
    else:
        parameters = jd["parameters"]
        # if plugin_name contains other str
        available_functions = [f.get("name", None) for f in functions]
        plugin_name = jd.get("name", "")
        if plugin_name not in available_functions:
            for fct in available_functions:
                if fct in plugin_name:
                    plugin_name = fct
                    break
        function_call = ChatFunctionCallResponse(
            name=plugin_name,
            thought=thought,
            arguments=parameters if isinstance(parameters,str) else json.dumps(parameters, ensure_ascii=False),
        )
    return function_call