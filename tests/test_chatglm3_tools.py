# coding=utf8
# @Time    : 2023/10/31 20:35
# @Author  : tk
# @FileName: test_chatglm3_tools
import copy
import json
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://192.168.2.180:8081/v1"
model = "chatglm3-6b"


data = {
    "model": model,
    "adapter_name": None, # lora头
    "top_p": 0.8,
    "temperature": 1.0,
    "frequency_penalty": 1.01,
    "stream": False,
    "nchar": 1,# stream 字符
    "n": 1, # 返回 n 个choices
    "max_tokens": 2048,
    "stop": ["Observation:"]
}




def chat_stage_1():
    functions  = [
        {
            "name": "track",
            "description": "追踪指定股票的实时价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "description": "需要追踪的股票代码"
                    }
                },
                "required": ['symbol']
            }
        },
        {
            "name": "text-to-speech",
            "description": "将文本转换为语音",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "description": "需要转换成语音的文本"
                    },
                    "voice": {
                        "description": "要使用的语音类型（男声、女声等）"
                    },
                    "speed": {
                        "description": "语音的速度（快、中等、慢等）"
                    }
                },
                "required": ['text']
            }
        }
    ]

    data["messages"] =  [{"role": "user", "content": "帮我查询股票10111的价格"}]

    completion = openai.ChatCompletion.create(**data,functions=functions)
    # response = completion.choices[0].message.content
    # print("result:", completion.choices[0].message.content)
    # print(completion.choices[0].message)
    print(completion.choices[0].message.function_call.arguments)
    # 添加历史聊天信息
    data["messages"].append({"role": "assistant","content": completion.choices[0].message.content})

    arguments = copy.deepcopy(completion.choices[0].message.function_call.arguments)

    return arguments

#结构化抽取函数参数
arguments = chat_stage_1()
symbol = eval(arguments)

print('结构化参数',symbol)

#调用自定义函数，例如各种查询操作，返回字符串结果
def track(symbol):
    return json.dumps({"price": 12412}, ensure_ascii=False)

result = track(**symbol)



# 获取并观察最终结果
data["messages"].append({"role": "observation", "content": result})
completion = openai.ChatCompletion.create(**data)
print(completion.choices[0].message.content)