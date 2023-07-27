# -*- coding: utf-8 -*-
# @Time:  23:32
# @Author: tk
# @File：http_serving_openai

import json
import logging
import traceback

from starlette.responses import StreamingResponse
import typing
from multiprocessing import Process
import uvicorn
from fastapi.openapi.models import HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseSettings
from fastapi import FastAPI, Depends, HTTPException
from starlette.middleware.cors import CORSMiddleware

from config.constant_map import models_info_args as model_config_map
from serving.openai_api.openai_api_protocol import ModelCard, ModelPermission, ModelList, ChatCompletionResponse, \
    ChatCompletionRequest, Role, ChatCompletionResponseChoice, UsageInfo, ChatMessage, Finish, \
    ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionStreamResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)



class AppSettings(BaseSettings):
    # The address of the model controller.
    api_keys: typing.List[str] = None

app_settings = AppSettings()
headers = {"User-Agent": "FastChat API Server"}
get_bearer_token = HTTPBearer(auto_error=False)

async def check_api_key(
    auth: typing.Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None











class HTTP_Serving(Process):
    def __init__(self,
                 queue_mapper: dict,
                 http_ip='0.0.0.0',
                 http_port=8088,
                 http_num_workers=1,
                 ):
        super().__init__(daemon=True)
        self.http_num_workers = http_num_workers
        self.http_ip = http_ip
        self.http_port = http_port
        self.queue_mapper = queue_mapper
        self.app = None

    def create_app(self):
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
            return {"Hello": "World"}

        #@app.get("/v1/models", dependencies=[Depends(check_api_key)])
        @app.get("/v1/models")
        async def show_available_models():
            models = [k for k, v in model_config_map.items() if v["enable"]]
            models.sort()
            # TODO: return real model permission details
            model_cards = []
            for m in models:
                model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
            return ModelList(data=model_cards)

        @app.post("/v1/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            try:
                if len(request.messages) == 0:
                    raise HTTPException(status_code=400, detail="Invalid request")

                if request.messages[-1].role != Role.USER:
                    raise HTTPException(status_code=400, detail="Invalid request")

                if request.model not in model_config_map:
                    msg = "model not in " + ','.join([k for k, v in model_config_map.items() if v["enable"]])
                    raise HTTPException(status_code=400, detail=msg)
                if request.stream:
                    _openai_chat_stream_generate = _openai_chat_stream(request)
                    return StreamingResponse(_openai_chat_stream_generate, media_type="text/event-stream")
                else:
                    return await _openai_chat(request)
            except Exception as e:
                traceback.print_exc()
                traceback.print_last()
                print(e)
                return HTTPException(status_code=501, detail=str(e))

        async def _openai_chat(request: ChatCompletionRequest):
            r = request.build_request_chat()
            choices = []

            prompt_length, response_length = 0, 0
            for i in range(max(1,r.n)):
                instance = self.queue_mapper[request.model]
                request_id = instance.put(r)
                result = instance.get(request_id)
                if result["code"] != 0:
                    raise HTTPException(status_code=400, detail=result["msg"])


                for x in r["history"]:
                    prompt_length += len(x['q'])
                    prompt_length += len(x['a'])
                prompt_length += len(r['query'])

                response_length = len(result["result"])

                choice_data = ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role=Role.ASSISTANT, content=result["result"]),
                    finish_reason=Finish.STOP
                )
                choices.append(choice_data)
            usage = UsageInfo(
                prompt_tokens=prompt_length,
                completion_tokens=response_length,
                total_tokens=prompt_length + response_length
            )
            return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)


        async def _openai_chat_stream(request: ChatCompletionRequest):
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role=Role.ASSISTANT),
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
                elif len(result["result"]) > 0:
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=result["result"]),
                        finish_reason=None
                    )
                    chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
                    yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

                if result["complete"]:
                    break


            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=Finish.STOP
            )
            chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        @app.post("/generate")
        async def generate(r: typing.Dict):
            try:
                logger.info(r)
                r["method"] = "generate"
                model_name = r.get('model', None)
                texts = r.get('texts', [])
                if len(texts) == 0 or texts is None:
                    return {'code': -1, "msg": "invalid data"}
                if model_name not in model_config_map:
                    msg = "model not in " + ','.join([k for k, v in model_config_map.items() if v["enable"]])
                    print(msg)
                    return {'code': -1, "msg": msg}

                instance = self.queue_mapper[model_name]
                request_id = instance.put(r)
                result = instance.get(request_id)

                return result
            except Exception as e:
                traceback.print_exc()
                print(e)
                return {'code': -1, "msg": str(e)}

        @app.post("/chat")
        async def chat(r: typing.Dict):
            try:
                logger.info(r)
                r["method"] = "chat"
                model_name = r.get('model', None)
                history = r.get('history', [])
                query = r.get('query', "")
                if len(query) == 0 or query is None:
                    return {'code': -1, "msg": "invalid data"}
                if len(history) != 0:
                    assert isinstance(history[0], dict), ValueError('history require dict data')
                    if 'q' not in history[0] or 'a' not in history[0]:
                        raise ValueError('q,a is required in list item')
                if model_name not in model_config_map:
                    msg = "model not in " + ','.join([k for k, v in model_config_map.items() if v["enable"]])
                    print(msg)
                    return {'code': -1, "msg": msg}

                instance = self.queue_mapper[model_name]
                request_id = instance.put(r)
                result = instance.get(request_id)

                return result
            except Exception as e:
                traceback.print_exc()
                print(e)
                return {'code': -1, "msg": str(e)}

        @app.post("/chat_stream")
        def chat_stream(r: typing.Dict):
            try:
                logger.info(r)
                r["method"] = "chat_stream"
                model_name = r.get('model', None)
                history = r.get('history', [])
                query = r.get('query', "")
                param = r.get('param',{})
                nchar = param.get('nchar', 4)
                gtype = param.get('gtype', 'total')
                do_sample = param.get('do_sample', True)
                assert do_sample, ValueError("stream not support do_sample=False")
                param['do_sample'] = True
                assert isinstance(nchar, int) and nchar > 0, ValueError("require nchar > 0")
                assert gtype in ['total', 'increace'], ValueError("gtype one of increace , total")

                if len(query) == 0 or query is None:
                    return {'code': -1, "msg": "invalid data"}
                if len(history) != 0:
                    assert isinstance(history[0], dict), ValueError('history require dict data')
                    if 'q' not in history[0] or 'a' not in history[0]:
                        raise ValueError('q,a is required in list item')
                if model_name not in model_config_map:
                    msg = "model not in " + ','.join([k for k, v in model_config_map.items() if v["enable"]])
                    print(msg)
                    return {'code': -1, "msg": msg}

                instance = self.queue_mapper[model_name]
                request_id = instance.put(r)

                def iterdata():
                    while True:
                        result = instance.get(request_id)
                        yield json.dumps(result, ensure_ascii=False)
                        if result["complete"]:
                            break
            except Exception as e:
                traceback.print_exc()
                print(e)

                def iterdata():
                    yield json.dumps({'code': -1, "msg": str(e)}, ensure_ascii=False)

            return StreamingResponse(iterdata(), media_type="application/json")

        return app

    def close_server(self):
        if self.app is not None:
            self.app.stop()

    def run(self):
        self.app = self.create_app()
        uvicorn.run(self.app, host=self.http_ip, port=self.http_port, workers=self.http_num_workers)

