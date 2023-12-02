import openai

# 旧版本
openai.api_key = "112233"
openai.api_base = "http://106.12.147.243:9090/v1"
model = "chatglm2-6b-int4"
model = "ChatYuan-large-v2"
model = "Qwen-14B-Chat"

# Test completion API
stream = True
# 非 聊天接口 ， 不带 prompt 和历史会话
data = {
    "model": model,
    "adapter_name": None, # lora头
    "prompt": ["你是",],
    "top_p": 0.8,
    "temperature": 1.0,
    "frequency_penalty": 1.01,
    "stream": stream,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    "max_tokens": 512,
    # "stop": ["Observation:","Observation:\n"]
    "seed": None,
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
        print("result:", choice.text)
