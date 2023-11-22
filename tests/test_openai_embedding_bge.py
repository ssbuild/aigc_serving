import openai

# 新版本
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.16.157:8081/v1"

model = "chatglm2-6b-int4"
model = "qwen-7b-chat-int4"
model = "bge-base-zh-v1.5"
model = "m3e-base"

# # Test list models API
# models = openai.Model.list()
# print("Models:", models)

# Test completion API
stream = False

data = {
    "model": model,
    "adapter_name": None, # lora头
    "input": ["你是谁",],
}


completion = openai.Embedding.create(**data)

for d in completion.data:
    print(d)