# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/24 17:26
import json

import requests

# url = 'http://192.168.16.157:8081/chat_stream'
# model = "bloom-560m"

url = 'http://192.168.2.180:8081/chat_stream'
model = "chatglm2-6b-int4"

data = {
    "query": "你是谁",
    "model": model,
    "params": {
        "gtype": "total", # one of total,increace
        "max_new_tokens": 512,"do_sample": True,"temperature": 0.95,"top_p": 0.8,"repetition_penalty": 1.01}
}

r:requests.Response = requests.post(url,json=data,stream=True)

idx = 0
for chunk in r.iter_content(chunk_size=8172):
    if chunk:
        idx += 1
        d = json.loads(chunk.decode())
        print(d)