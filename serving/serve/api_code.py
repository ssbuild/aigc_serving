# coding=utf8
# @Time    : 2023/10/31 22:32
# @Author  : tk
# @FileName: api_code
import json
from typing import Optional, Union
from serving.openai_api.openai_api_protocol import CompletionRequest, ChatCompletionRequest, ChatCodeCallResponse


def parse_tools_code(request: Union[CompletionRequest,ChatCompletionRequest],
                    context_text)-> Optional[ChatCodeCallResponse]:
    model_type = request.model_type
    if model_type in ["chatglm", "chatglm3"]:
        return _process_chatglm3_code(request,context_text)
    return None


def _process_chatglm3_code(request: Union[CompletionRequest,ChatCompletionRequest],
                            context_text)-> Optional[ChatCodeCallResponse]:
    thought,metadata = "",""
    blocks = context_text.split("<|assistant|>")
    if len(blocks) == 1:
        return None
    for response in blocks:
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            content = content.replace("[[训练时间]]", "2023年")
            thought += content
        else:
            content = {"name": metadata.strip(), "content": content}

    jd = content
    if not isinstance(jd,dict) or ("content" not in jd and not isinstance(jd["content"], str)):
        code_call = None
    else:
        code = jd["content"]
        code_call = ChatCodeCallResponse(
            metadata=metadata,
            thought=thought,
            code=code if isinstance(code,str) else json.dumps(code, ensure_ascii=False),
        )
    return code_call