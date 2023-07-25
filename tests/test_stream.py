# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/24 17:26

import requests

url = 'http://192.168.2.180:8081/chat_stream'

data = {
    "texts": ["2023年是什么年"],
    "model": "bloom-560m",
    "params": {"max_new_tokens": 512,"do_sample": True,"temperature": 0.95,"top_p": 0.8,"repetition_penalty": 1.01}
}

r:requests.Response = requests.post(url,json=data,stream=True)

idx = 0
for chunk in r.iter_content(chunk_size=1):

    if chunk:
        idx += 1
        print(idx,chunk.decode())