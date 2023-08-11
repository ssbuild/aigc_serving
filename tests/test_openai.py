import openai

# 旧版本
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.2.180:8081/v1"
# openai.api_base = "http://101.42.176.124:8081/v1"
model = "chatglm2-6b-int4"
model = "qwen-7b-chat-int4"

# # Test list models API
# models = openai.Model.list()
# print("Models:", models)

# Test completion API
stream = True

data = {
    "model": model,
    "adapter_name": "default",
    "prompt": "你是谁?",
    "top_p": 0.8,
    "temperature": 1.0,
    "frequency_penalty": 1.01,
    "stream": stream,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    # "stop": ["Observation:","Observation:\n"]
}


completion = openai.Completion.create(**data)
if stream:
    text = ''
    for choices in completion:
        c = choices.choices[0]
        text += c.text
        print(c.text)
    print(text)
else:
    for choice in completion.choices:
        print("Completion result:", choice.text)
