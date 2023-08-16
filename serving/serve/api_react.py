# -*- coding: utf-8 -*-
# @Time:  21:11
# @Author: tk
# @File：api_react
import typing
from serving.openai_api.openai_api_protocol import CompletionRequest, ChatCompletionRequest
from serving.react.qwen.react_prompt import get_qwen_react_prompt


def build_request_functions(request: typing.Union[CompletionRequest,ChatCompletionRequest]):
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
        request.messages, functions = get_qwen_react_prompt(messages, functions, function_call)

    return functions