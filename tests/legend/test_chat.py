# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/24 15:56
import requests

url = 'http://192.168.16.157:8081/chat'
model = "bloom-560m"

url = 'http://192.168.2.180:8081/chat'
model = "chatglm2-6b-int4"


data = {
    "query": "你会干什么",
    "model": model,
    "params": {"adapter_name": "default","max_new_tokens": 512,"do_sample": True,"temperature": 1.0,"top_p": 0.8,"repetition_penalty": 1.01}
}

r = requests.post(url,json=data).json()
print(r)