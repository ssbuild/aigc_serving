# -*- coding: utf-8 -*-
# @Time:  15:59
# @Author: tk
# @File：api
import json
import traceback
import typing
import uuid

from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from serving.config_parser.main import global_models_info_args
from serving.openai_api.openai_api_protocol import ModelCard, ModelPermission, ModelList, ChatCompletionRequest, Role, \
    ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionStreamResponse, Finish, \
    ChatCompletionResponseChoice, ChatMessage, UsageInfo, ChatCompletionResponse, CompletionRequest, \
    CompletionResponseChoice, CompletionResponseStreamChoice, CompletionStreamResponse, CompletionResponse, \
    ChatFunctionCallResponse, EmbeddingsRequest, EmbeddingsResponse
from serving.react.qwen.react_prompt import parse_qwen_plugin_call
from serving.serve.api_keys import auth_api_key
from serving.serve.api_react import build_request_functions
from serving.utils import logger
from serving.serve.api_check import check_requests, create_error_response, ErrorCode, check_requests_embedding


class Resource:
    def __init__(self):
       self.valid_model_map = set([k for k, v in global_models_info_args.items() if v["enable"]])
       self.lifespan = None

    def set_mapper(self,queue_mapper):
        self.queue_mapper = queue_mapper

_g_instance = Resource()

def global_instance() -> Resource:
    global _g_instance
    return _g_instance



app = FastAPI()
app.add_middleware(  # 添加中间件
    CORSMiddleware,  # CORS中间件类
    allow_origins=["*"],  # 允许起源
    allow_credentials=True,  # 允许凭据
    allow_methods=["*"],  # 允许方法
    allow_headers=["*"],  # 允许头部
)



@app.get("/")
def read_root():
    return {"aigc_serving": "hello world"}

@app.get("/v1/models", dependencies=[Depends(auth_api_key)])
def list_models():
    models = [(k,v['model_config'].get('lora',{}),v.get('auto_merge_lora_single',False)) for k, v in global_models_info_args.items() if v["enable"]]
    # models.sort()
    # TODO: return real model permission details
    model_cards = []
    for m,lora_conf,auto_merge_lora_single in models:
        adapters = None
        if len(lora_conf) > 1 or (len(lora_conf) == 1 and not auto_merge_lora_single):
            adapters = list(lora_conf.keys())

        model_cards.append(ModelCard(id=m, root=m,
                                     permission=[ModelPermission()],
                                     adapters=adapters))
    return ModelList(data=model_cards)




@app.post("/v1/completions", dependencies=[Depends(auth_api_key)])
@app.post("/v1/chat/completions", dependencies=[Depends(auth_api_key)])
def create_chat_completion(request: typing.Union[CompletionRequest,ChatCompletionRequest]):
    self = global_instance()
    try:
        logger.info(request.json(indent=2,ensure_ascii=False))
        error_check_ret = check_requests(request)
        if error_check_ret is not None:
            return error_check_ret

        if request.model not in self.valid_model_map:
            msg = "{} Invalid model: model not in [".format(request.model) + ','.join(self.valid_model_map) + "]"
            return create_error_response(ErrorCode.INVALID_MODEL,msg)

        if request.stream:
            _openai_chat_stream_generate = _openai_chat_stream(request) if isinstance(request,ChatCompletionRequest) else _openai_legend_stream(request)
            return StreamingResponse(_openai_chat_stream_generate, media_type="text/event-stream")
        else:
            return _openai_chat(request) if isinstance(request,ChatCompletionRequest) else _openai_legend(request)
    except Exception as e:
        traceback.print_exc()
        logger.info(e)
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))


def _openai_chat(request: typing.Union[CompletionRequest,ChatCompletionRequest]):
    functions = None
    if request.model.lower().find('qwen') != -1:
        functions = build_request_functions(request)
    self = global_instance()
    rs = request.build_request_chat()
    choices = []
    prompt_length, response_length = 0, 0
    for r in rs:
        for i in range(max(1,request.n)):
            instance = self.queue_mapper[request.model]
            request_id = instance.put(r)
            result = instance.get(request_id)
            if result["code"] != 0:
                logger.error(result["msg"])
                return create_error_response(ErrorCode.INTERNAL_ERROR, result["msg"])

            for x in r["history"]:
                prompt_length += len(x['q'])
                prompt_length += len(x['a'])
            prompt_length += len(r['query'])
            response_length += len(result["response"])
            context_text = result["response"]
            if functions is not None:
                functions = request.functions
                if "Thought:" in context_text:
                    react_res = parse_qwen_plugin_call(context_text)
                    if react_res is not None:
                        # if plugin_name contains other str
                        available_functions = [f.get("name",None) or f.get("name_for_model",None) for f in functions]
                        plugin_name = react_res[1]
                        if plugin_name not in available_functions:
                            for fct in available_functions:
                                if fct in plugin_name:
                                    plugin_name = fct
                                    break

                        function_call = ChatFunctionCallResponse(
                            thought=react_res[0],
                            name=plugin_name,
                            arguments=react_res[2],
                        )
                    else:
                        function_call = None
                    choices.append(
                        ChatCompletionResponseChoice(
                            index=len(choices),
                            message=ChatMessage(role=Role.ASSISTANT,content="", function_call=function_call, functions=functions),
                            finish_reason="function_call",
                        )
                    )
            else:
                choices.append(ChatCompletionResponseChoice(
                    index=len(choices),
                    message=ChatMessage(role=Role.ASSISTANT, content=context_text,function_call=None),
                    finish_reason=Finish.STOP
                ))
    usage = UsageInfo(
        prompt_tokens=prompt_length,
        completion_tokens=response_length,
        total_tokens=prompt_length + response_length
    )
    return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

def _openai_chat_stream(request: typing.Union[CompletionRequest,ChatCompletionRequest]):
    self = global_instance()

    for i in range(max(1, request.n)):
        idx = i
        choice_data = ChatCompletionResponseStreamChoice(
            index=idx,
            delta=DeltaMessage(role=Role.ASSISTANT,content=''),
            finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        r = request.build_request_streaming()
        instance = self.queue_mapper[request.model]
        request_id = instance.put(r)

        while True:
            result = instance.get(request_id)
            if result["code"] != 0:
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            elif not result["complete"] and len(result["response"]) > 0:
                choice_data = ChatCompletionResponseStreamChoice(
                    index=idx,
                    delta=DeltaMessage(role=Role.ASSISTANT,content=result["response"]),
                    finish_reason=None
                )
                chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

            if result["complete"]:
                break


        choice_data = ChatCompletionResponseStreamChoice(
            index=idx,
            delta=DeltaMessage(),
            finish_reason=Finish.STOP
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"









def _openai_legend(request: CompletionRequest):
    self = global_instance()
    rs = request.build_request_chat()
    choices = []
    prompt_length, response_length = 0, 0
    for r in rs:
        for i in range(max(1,request.n)):
            instance = self.queue_mapper[request.model]
            request_id = instance.put(r)
            result = instance.get(request_id)
            if result["code"] != 0:
                logger.error(result["msg"])
                return create_error_response(ErrorCode.INTERNAL_ERROR, result["msg"])

            for x in r["history"]:
                prompt_length += len(x['q'])
                prompt_length += len(x['a'])
            prompt_length += len(r['query'])

            response_length += len(result["response"])
            choice_data = CompletionResponseChoice(
                index=len(choices),
                text=result["response"],
                finish_reason=Finish.STOP
            )
            choices.append(choice_data)
    usage = UsageInfo(
        prompt_tokens=prompt_length,
        completion_tokens=response_length,
        total_tokens=prompt_length + response_length
    )
    return CompletionResponse(model=request.model, choices=choices, usage=usage)

def _openai_legend_stream(request: CompletionRequest):
    self = global_instance()

    for i in range(max(1, request.n)):
        idx = i
        choice_data = CompletionResponseStreamChoice(
            index=idx,
            text="",
            finish_reason=None
        )
        chunk = CompletionStreamResponse(model=request.model, choices=[choice_data])
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        r = request.build_request_streaming()
        instance = self.queue_mapper[request.model]
        request_id = instance.put(r)

        while True:
            result = instance.get(request_id)
            if result["code"] != 0:
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            elif not result["complete"] and len(result["response"]) > 0:
                choice_data = CompletionResponseStreamChoice(
                    index=idx,
                    text=result["response"],
                    finish_reason=None
                )
                chunk = CompletionStreamResponse(model=request.model, choices=[choice_data])
                yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

            if result["complete"]:
                break


        choice_data = CompletionResponseStreamChoice(
            index=idx,
            text="",
            finish_reason=Finish.STOP
        )
        chunk = CompletionStreamResponse(model=request.model, choices=[choice_data])
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"










@app.post("/v1/embeddings", dependencies=[Depends(auth_api_key)])
@app.post("/v1/engines/{model_name}/embeddings", dependencies=[Depends(auth_api_key)])
def create_embeddings(request: EmbeddingsRequest, model_name: str = None):
    """Creates embeddings for the text"""
    if request.model is None:
        request.model = model_name
    self = global_instance()
    try:
        logger.info(request.json(indent=2, ensure_ascii=False))
        error_check_ret = check_requests_embedding(request)
        if error_check_ret is not None:
            return error_check_ret

        if request.model not in self.valid_model_map:
            msg = "{} Invalid model: model not in [".format(request.model) + ','.join(self.valid_model_map) + "]"
            return create_error_response(ErrorCode.INVALID_MODEL, msg)

        conf = global_models_info_args[request.model]
        max_batch_size = getattr(conf,"max_batch_size",1)

        batches = request.build_data(max_batch_size)

        data,prompt_length = [],0
        for bs_id,bs in enumerate(batches):
            r = request.build_request(bs)
            instance = self.queue_mapper[request.model]
            request_id = instance.put(r)
            result = instance.get(request_id)
            if result["code"] != 0:
                logger.error(result["msg"])
                return create_error_response(ErrorCode.INTERNAL_ERROR, result["msg"])
            vecs = result["response"]
            data += [
                {
                    "object": "embedding",
                    "embedding": emb,
                    "index": bs_id * max_batch_size + i,
                }
                for i, emb in enumerate(vecs)
            ]
            prompt_length += sum([len(i) for i in bs])

        return EmbeddingsResponse(
            data=data,
            model=request.model,
            usage=UsageInfo(
                prompt_tokens=prompt_length,
                total_tokens=prompt_length,
                completion_tokens=None,
            ),
        ).dict(exclude_none=True)
    except Exception as e:
        traceback.print_exc()
        logger.info(e)
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

