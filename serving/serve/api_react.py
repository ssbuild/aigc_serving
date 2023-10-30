# -*- coding: utf-8 -*-
# @Time:  21:11
# @Author: tk
# @File：api_react
from typing import Union
from serving.openai_api.openai_api_protocol import CompletionRequest, ChatCompletionRequest
from serving.react.qwen.react_prompt import get_react_prompt_for_qwen
from serving.react.chaglm3.react_prompt import get_react_prompt_for_chatglm3


def build_request_functions(request: Union[CompletionRequest,ChatCompletionRequest],model_type):
    function_call = request.function_call
    functions = request.functions
    messages = request.messages
    use_function = False
    if isinstance(messages, list) and isinstance(messages[0], dict):
        for message in messages:
            if message.functions:
                use_function = True
                break

    if functions is not None or use_function:
        if model_type == "qwen":
            request.messages, functions = get_react_prompt_for_qwen(messages, functions, function_call)
        elif model_type == "chatglm3":
            request.messages, functions = get_react_prompt_for_chatglm3(messages, functions, function_call)

    return functions