
## 简介

   aigc_serving light and efficient 大语言模型高性能纯python推理服务


## update information
```text
    08-07 support llama llama2 量化推理 , 可选版本要求 deep_training 0.1.15.rc0
    08-05 aigc_zoo 最低版本0.1.14 
    08-03 support qwen (千问）
    08-02 support muti lora infer , 手动升级 aigc_zoo , pip install -U git+https://github.com/ssbuild/aigc_zoo.git --force-reinstall --no-deps
    07-27 support openai client
    07-26 support streaming
    07-24 support chat
    07-23 support deepspeed , accelerate
```
   注意:
   linux python >=3.8
    

## install
pip install -r requirements.txt



## 启动
修改 config/constant_map.py 配置文件

```commandline
cd script
bash start.sh
```

## openai 接口  demo tests/openai_client.py

```text
Models: {
  "data": [
    {
      "created": 1690434104,
      "id": "bloom-560m",
      "object": "model",
      "owned_by": "aigc_serving",
      "parent": null,
      "permission": [
        {
          "allow_create_engine": false,
          "allow_fine_tuning": false,
          "allow_logprobs": true,
          "allow_sampling": true,
          "allow_search_indices": true,
          "allow_view": true,
          "created": 1690434104,
          "group": null,
          "id": "modelperm-fd413444-e293-4500-a4f6-653c773f17f8",
          "is_blocking": false,
          "object": "model_permission",
          "organization": "*"
        }
      ],
      "root": "bloom-560m"
    }
  ],
  "object": "list"
}
Completion result: ？我的父亲。我曾以为我是一个很坚强的女孩,但是当我遇到这个男孩的时候我才发现原来我没有那么的坚强！曾经我们相恋了三个月的时间,他却让我失去了理智并且对我很冷淡.一次又一次地伤害了我！我开始怀疑自己的感情是不是真的能够长久的存在下去！于是我在网上看到了这样一篇文章:其实爱情是很容易变质的！也许是你喜欢上了别人才变的爱你；也可能是你因为爱而爱上另一个人才会把你放弃；而或许是你不懂珍惜一个人就已经不在乎了！</s>

```

## generate 接口  demo tests/test_client.py

params参数参考 [https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)

```text
http://127.0.0.1:8081/generate
请求
{
    "texts": ["你会干什么？"],
    "model": "bloom-560m",
    "params": {"adapter_name": "default","max_new_tokens": 512,"do_sample": true,"temperature": 0.9,"top_p": 0.7}
}

例子 
curl http://127.0.0.1:8081/generate -H "Content-Type: application/json" -X POST -d '{"texts":["你会干什么？"],"model":"bloom-560m","params":{"max_new_tokens":512,"do_sample":true,"temperature":0.9,"top_p":0.7}}'


返回
{
    "code": 0,
    "runtime": 520.856618881226,
    "result": [
        " 学了那么多年英语,总是会忘记你问的单词和句子。 这不是一件好事情。 有时候我甚至还会把自己关在房间里、躲在被子里哭闹着寻找一个能够安慰自己的人- 我不想听你说话,我不想再看见你笑； 我不愿意去理你发脾气-我怕听到你的抱怨声-你只是不愿面对现实罢了... 我想说很多废话来表达我对你的想念-- 你是我生命中不能承受之轻的部分,因为你是一个让我感觉很温暖的女人！ 尽管现在我们已然分开十几年多,可是我们都依然爱着你: 因为在你身边有我最最爱的爸爸妈妈和最亲爱的朋友；在我的心中只有他们两个。 也许有一天我真的要离开这里了(虽然是心痛)。 可是我想对你说\"对不起\"的字眼。</s>"
    ],
    "msg": "ok"
}
```

## chat  接口  demo tests/test_chat.py

```text
http://127.0.0.1:8081/chat
请求
{
    "history": [{
        "q": "你是谁",
        "a": "学了那么多年英语,总是会忘记你问的单词和句子。 这不是一件好事情。 有时候我甚至还会把自己关在房间里、躲在被子里哭闹着寻找一个能够安慰自己的人- 我不想听你说话,我不想再看见你笑； 我不愿意去理你发脾气-我怕听到你的抱怨声-你只是不愿面对现实罢了... 我想说很多废话来表达我对你的想念-- 你是我生命中不能承受之轻的部分,因为你是一个让我感觉很温暖的女人！ 尽管现在我们已然分开十几年多,可是我们都依然爱着你: 因为在你身边有我最最爱的爸爸妈妈和最亲爱的朋友；在我的心中只有他们两个。 也许有一天我真的要离开这里了(虽然是心痛)。 可是我想对你说\"对不起\"的字眼。</s>"
   
    }],
    "query": "你会干什么？",
    "model": "bloom-560m",
    "params": {"adapter_name": "default","max_new_tokens": 512,"do_sample": true,"temperature": 0.9,"top_p": 0.7}
}


返回
{
    "code": 0,
    "runtime": 520.856618881226,
    "result": " 每个人都是一个自恋狂,但真正自恋狂的并不多。因为自恋狂是无法理智地思考问题的,自恋狂总是幻想自己能成为别人眼中完美无瑕的自己。 现实是残酷的,自恋狂往往自以为是,自以为是的人往往会变成自恋狂。 事实上,自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 首先,自恋狂往往会给自己找很多借口,把自己推向极端。自恋狂往往会给自己找很多借口,把自己推向极端。 自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 其次,自恋狂常常会把自己当成一个疯子,把自己当成一个疯子,把自己当成一个疯子。 自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 最后,自恋狂往往会给自己找很多理由,给自己找很多理由,给自己找很多理由,给自己找很多理由,给自己找很多理由。 自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 那么,自恋狂的背后到底是什么？ 自恋狂的背后到底是什么？ 实际上,自恋狂是自我满足的。自恋狂是自我满足的。自恋狂是自我满足的。 自恋狂是自我满足的。自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足",
    "history": [
          {
            "q": "你是谁",
            "a": "学了那么多年英语,总是会忘记你问的单词和句子。 这不是一件好事情。 有时候我甚至还会把自己关在房间里、躲在被子里哭闹着寻找一个能够安慰自己的人- 我不想听你说话,我不想再看见你笑； 我不愿意去理你发脾气-我怕听到你的抱怨声-你只是不愿面对现实罢了... 我想说很多废话来表达我对你的想念-- 你是我生命中不能承受之轻的部分,因为你是一个让我感觉很温暖的女人！ 尽管现在我们已然分开十几年多,可是我们都依然爱着你: 因为在你身边有我最最爱的爸爸妈妈和最亲爱的朋友；在我的心中只有他们两个。 也许有一天我真的要离开这里了(虽然是心痛)。 可是我想对你说\"对不起\"的字眼。</s>"
        },
        {
            "q": "你会干什么？",
            "a": " 每个人都是一个自恋狂,但真正自恋狂的并不多。因为自恋狂是无法理智地思考问题的,自恋狂总是幻想自己能成为别人眼中完美无瑕的自己。 现实是残酷的,自恋狂往往自以为是,自以为是的人往往会变成自恋狂。 事实上,自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 首先,自恋狂往往会给自己找很多借口,把自己推向极端。自恋狂往往会给自己找很多借口,把自己推向极端。 自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 其次,自恋狂常常会把自己当成一个疯子,把自己当成一个疯子,把自己当成一个疯子。 自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 最后,自恋狂往往会给自己找很多理由,给自己找很多理由,给自己找很多理由,给自己找很多理由,给自己找很多理由。 自恋狂的背后是很多现实问题,自恋狂的背后是很多现实问题。 那么,自恋狂的背后到底是什么？ 自恋狂的背后到底是什么？ 实际上,自恋狂是自我满足的。自恋狂是自我满足的。自恋狂是自我满足的。 自恋狂是自我满足的。自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足的。 自恋狂是自我满足"
        }
    ],
    "msg": "ok"
}
```



## test_stream  接口  demo tests/test_stream.py
```text

请求
{
    "query": "你是谁",
    "model": model,
    "params": {
        "adapter_name": "default",
        "gtype": "total", # one of total,increace
        "max_new_tokens": 512,"do_sample": True,"temperature": 0.95,"top_p": 0.8,"repetition_penalty": 1.01}
}
返回

{'code': 0, 'runtime': 784.7540378570557, 'result': '我是一个名为 Chat', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 1467.4859046936035, 'result': '我是一个名为 ChatGLM2-', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 2151.3359546661377, 'result': '我是一个名为 ChatGLM2-6B 的人工', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 2835.114002227783, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 3520.101308822632, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 4205.242872238159, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 4891.727447509766, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 5578.71413230896, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 6267.045497894287, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 6956.869840621948, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型开发的。我的', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 7646.951675415039, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型开发的。我的任务是针对用户', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 8337.846517562866, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型开发的。我的任务是针对用户的问题和要求提供适当的', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 9029.44540977478, 'result': '我是一个名为 ChatGLM2-6B 的人工智能助手，是基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型开发的。我的任务是针对用户的问题和要求提供适当的答复和支持。', 'msg': 'ok', 'complete': False}
{'code': 0, 'runtime': 9029.971837997437, 'result': '', 'msg': 'ok', 'complete': True}
```

## 推荐界面 ChatGPT-Next-Web

![界面](asserts/1.png)

## 
    纯粹而干净的代码


## 注意事项
```text
1、 如果deepspeed ， 确保 num_attention_heads % len(device_id) == 0

```



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/aigc_serving&type=Date)](https://star-history.com/#ssbuild/aigc_serving&Date)

