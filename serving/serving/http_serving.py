# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/21 8:55
import json
import logging
import traceback
from starlette.responses import StreamingResponse
import typing
from multiprocessing import Process
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from config.constant_map import models_info_args as model_config_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class HTTP_Serving(Process):
    def __init__(self,
                 queue_mapper : dict,
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
                    msg = "model not in " + ','.join( [k for k,v in model_config_map.items() if v["enable"]])
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
                    assert isinstance(history[0],dict),ValueError('history require dict data')
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
                n = r.get('n', 4)
                gtype = r.get('gtype', 'total')

                do_sample = r.get('do_sample',True)
                assert do_sample, ValueError("stream not support do_sample=False")
                r['do_sample'] = True
                assert isinstance(n,int) and n > 0, ValueError("require n > 0")
                assert gtype in ['total','increace'], ValueError("gtype one of increace , total")

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
                    yield json.dumps({'code': -1, "msg": str(e)},ensure_ascii=False)
            

            return StreamingResponse(iterdata(), media_type="application/json")
        return app

    def close_server(self):
        if self.app is not None:
            self.app.stop()

            
    def run(self):
        self.app = self.create_app()
        uvicorn.run(self.app, host=self.http_ip, port=self.http_port, workers=self.http_num_workers)




