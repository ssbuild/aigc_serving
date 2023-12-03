# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/11 14:25
from typing import Optional
from starlette.responses import JSONResponse
from serving.openai_api.openai_api_protocol import ErrorResponse, CompletionRequest
from serving.serve.api_constants import ErrorCode


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=500
    )


def check_requests_gen(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    elif request.max_tokens is None:
        request.max_tokens = 2048 # 默认4k

    if request.n is not None and (request.n <= 0 or request.n >16) :
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 16",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
            not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )

    if isinstance(request,CompletionRequest) and request.stream:
        if isinstance(request.prompt,list) and len(request.prompt) > 1:
            return create_error_response(
                ErrorCode.RATE_LIMIT,
                f"request.prompt is only 1 for stream=True",
            )

    return None


def check_requests_embedding(request) -> Optional[JSONResponse]:
    if isinstance(request.input,list)  and len(request.input) > 100:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"input is overflow , the minimum of 100",
        )
    return None