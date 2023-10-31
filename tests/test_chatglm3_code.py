# coding=utf8
# @Time    : 2023/10/31 20:35
# @Author  : tk
# @FileName: test_chatglm3_tools
import copy
import json
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://106.12.147.243:8082/v1"
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


    data["messages"] =  [{"role": "system", "content": "你是一位智能AI助手，你叫ChatGLM3，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。"}]

    data["messages"].append({
        "role": "user",
        "content": """#File: /mnt/data/metadata.jsonl
#Size: 35380
#File uploaded
文件中是否存在缺失值或异常值？
"""
    })

    completion = openai.ChatCompletion.create(**data)

    # print("result:", completion.choices[0].message.content)
    print("***解析提示字符串:", completion.choices[0].message.code_call.thought)
    print("***解析名字:", completion.choices[0].message.code_call.metadata)
    print("***解析代码:", completion.choices[0].message.code_call.code)


    # 添加历史聊天信息
    data["messages"].append({"role": "assistant","content": completion.choices[0].message.content})



def chat_stage_2():
    data["messages"].append({
        "role": "observation",
        "content": """```result
[{'file_name': 'aceinthehole.png',
  'name': 'Ace in the Hole',
  'type': 'survivor',
  'description': 'Lady Luck always seems to be throwing something good your way.'},
 {'file_name': 'adrenaline.png',
  'name': 'Adrenaline',
  'type': 'survivor',
  'description': 'You are fuelled by unexpected energy when on the verge of escape.'},
 {'file_name': 'aftercare.png',
  'name': 'Aftercare',
  'type': 'survivor',
  'description': "Unlocks potential in one's aura reading ability."},
 {'file_name': 'agitation.png',
  'name': 'Agitation',
  'type': 'killer',
  'description': 'You get excited in anticipation of hooking your prey.'},
 {'file_name': 'alert.png',
  'name': 'Alert',
  'type': 'survivor',
  'description': 'Your acute senses are on high alert.'}]
```
    """
    })

    completion = openai.ChatCompletion.create(**data)

    print("result:", completion.choices[0].message.content)
    # print("***解析提示字符串:", completion.choices[0].message.code_call.thought)
    # print("***解析名字:", completion.choices[0].message.code_call.metadata)
    # print("***解析代码:", completion.choices[0].message.code_call.code)

    # 添加历史聊天信息
    data["messages"].append({"role": "assistant", "content": completion.choices[0].message.content})

def chat_stage_3():
    data["messages"].append({
        "role": "observation",
        "content":  """```result
{'file_name': 0, 'name': 0, 'type': 0, 'description': 0}
```"""
    })

    completion = openai.ChatCompletion.create(**data)

    print("result:", completion.choices[0].message.content)
    # print("***解析提示字符串:", completion.choices[0].message.code_call.thought)
    # print("***解析名字:", completion.choices[0].message.code_call.metadata)
    # print("***解析代码:", completion.choices[0].message.code_call.code)

    # 添加历史聊天信息
    data["messages"].append({"role": "assistant", "content": completion.choices[0].message.content})

def chat_stage_4():
    data["messages"].append({
        "role": "observation",
        "content":  """```result
{'survivor': 116, 'killer': 103}
````"""
    })

    completion = openai.ChatCompletion.create(**data)

    print("result:", completion.choices[0].message.content)
    # print("***解析提示字符串:", completion.choices[0].message.code_call.thought)
    # print("***解析名字:", completion.choices[0].message.code_call.metadata)
    # print("***解析代码:", completion.choices[0].message.code_call.code)

    # 添加历史聊天信息
    data["messages"].append({"role": "assistant", "content": completion.choices[0].message.content})


def chat_stage_5():
    data["messages"].append({
        "role": "observation",
        "content":  """帮我画一个爱心"""
    })

    completion = openai.ChatCompletion.create(**data)

    print("result:", completion.choices[0].message.content)
    # print("***解析提示字符串:", completion.choices[0].message.code_call.thought)
    # print("***解析名字:", completion.choices[0].message.code_call.metadata)
    # print("***解析代码:", completion.choices[0].message.code_call.code)

    # 添加历史聊天信息
    data["messages"].append({"role": "assistant", "content": completion.choices[0].message.content})


chat_stage_1()

chat_stage_2()

chat_stage_3()

chat_stage_4()

chat_stage_5()