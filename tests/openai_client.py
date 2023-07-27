import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.16.157:8081/v1"
model = "bloom-560m"



# # Test list models API
# models = openai.Model.list()
# print("Models:", models)

# Test completion API
stream = True

data = {
    "model": model,
    "messages": [{"role": "user", "content": "你是谁"}],
    "top_p": 0.8,
    "temperature": 0.95,
    "frequency_penalty": 1.01,
    "stream": stream,
}


completion = openai.Completion.create(**data)

# print the completion
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
    print("Completion result:", completion.choices[0].message.content)
