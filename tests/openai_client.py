import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.2.180:8081/v1"
model = "chatglm2-6b-int4"



# # Test list models API
# models = openai.Model.list()
# print("Models:", models)

# Test completion API
stream = False

data = {
    "model": model,
    "messages": [{"role": "user", "content": "你是谁"}],
    "top_p": 0.8,
    "temperature": 0.95,
    "frequency_penalty": 1.01,
    "stream": stream,
    "nchar": 1,# stream 字符
    "n": 1 # 返回 n 个choices
}


completion = openai.Completion.create(**data)
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
        print("Completion result:", choice.message.content)
