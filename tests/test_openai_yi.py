# -*- coding: utf-8 -*-
import openai

# 新版本
openai.api_key = "112233"
openai.api_base = "http://192.168.2.180:8081/v1"
openai.api_base = "http://106.12.147.243:8081/v1"

openai.proxy = None
# model = "chatglm2-6b-int4"
# model = "qwen-7b-chat-int4"
# model = "ChatYuan-large-v2"
model = "qwen-7b-chat"
model = "qwen-chat-7b-int4"
model = "Yi-34B-Chat"
model = "Yi-34B-Chat-4bits"

# # Test list models API
# models = openai.Model.list()
# print("Models:", models)
# Test completion API
stream = True

input_text ="你是谁"

data = {
    "model": model,
    "adapter_name": None, # lora头
    "messages": [{"role": "user", "content": input_text}],
    "frequency_penalty": 1.01,
    "stream": stream,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    # "stop": ["Observation:"]
    "temperature": 0.6,
    "top_p": 0.8,
}

for i in range(1):
    completion = openai.ChatCompletion.create(**data)
    if stream:
        text = ''
        for choices in completion:
            c = choices.choices[0]
            delta = c.delta
            if hasattr(delta,'content'):
                text += delta.content
                print(delta.content)
        print(text)
    else:
        for choice in completion.choices:
            print("result:", choice.message.content)
