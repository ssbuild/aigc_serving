# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/21 8:55
import logging
import sys
import traceback

sys.path.append('..')
from starlette.responses import StreamingResponse
import os
import typing
import multiprocessing
from multiprocessing import Process
from typing import Union
import numpy as np
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from ipc_worker.ipc_zmq_loader import IPC_zmq, ZMQ_process_worker # noqa
from config.constant_map import models_info_args as model_config_map
from serving.workers import llm_worker

logger = logging.Logger("http_serving")

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
                    msg = "mode not in " + ','.join( [k for k,v in model_config_map.items() if v["enable"]])
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
                    msg = "mode not in " + ','.join([k for k, v in model_config_map.items() if v["enable"]])
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
                if len(query) == 0 or query is None:
                    return {'code': -1, "msg": "invalid data"}
                if len(history) != 0:
                    assert isinstance(history[0], dict), ValueError('history require dict data')
                    if 'q' not in history[0] or 'a' not in history[0]:
                        raise ValueError('q,a is required in list item')
                if model_name not in model_config_map:
                    msg = "mode not in " + ','.join([k for k, v in model_config_map.items() if v["enable"]])
                    print(msg)
                    return {'code': -1, "msg": msg}

                instance = self.queue_mapper[model_name]
                request_id = instance.put(r)

                def iterdata():
                    result = instance.get(request_id)
                    while not result["complete"]:
                        result = instance.get(request_id)
                        if result["code"] == 0:
                            yield result["result"]
                        yield result


            except Exception as e:
                traceback.print_exc()
                print(e)
                raise {'code': -1, "msg": str(e)}
            

            return StreamingResponse(iterdata(), media_type="text/plain")
        return app

    def close_server(self):
        if self.app is not None:
            self.app.stop()

            
    def run(self):
        self.app = self.create_app()
        uvicorn.run(self.app, host=self.http_ip, port=self.http_port, workers=self.http_num_workers)



def runner():
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    os.environ['ZEROMQ_SOCK_TMP_DIR'] = tmp_dir

    if __name__ == '__main__':
        evt_quit = multiprocessing.Manager().Event()
        queue_mapper = {}
        process_list = []

        for model_name, config in model_config_map.items():
            if not config["enable"]:
                continue
            group_name = 'serving_group_{}_1'.format(model_name)
            # group_name
            # manager is an agent  and act as a load balancing
            # worker is real doing your work
            instance = IPC_zmq(
                CLS_worker=llm_worker.My_worker,
                worker_args=(model_name, config,),  # must be tuple
                worker_num=len(config['workers']),  # number of worker Process  大模型 建议使用1个 worker
                group_name=group_name,  # share memory name
                evt_quit=evt_quit,
                queue_size=20,  # recv queue size
                is_log_time=True,  # whether log compute time
            )
            process_list.append(instance)
            queue_mapper[model_name] = instance
            instance.start()

        http_ = HTTP_Serving(queue_mapper,
                             http_ip='0.0.0.0',
                             http_port=8081, )
        http_.start()
        process_list.append(http_)
        try:
            for p in process_list:
                p.join()
        except Exception as e: # noqa
            evt_quit.set()
            for p in process_list:
                p.terminate()
        del evt_quit


if __name__ == '__main__':
  runner()

