
## 简介

   aigc_serving lightweight and efficient 大语言模型高性能纯python推理服务


## update information
```text
   07-23 support deepspeed , accelerate
```
   注意:
   linux python >=3.8
   因对进程的管理和共享内存机制，暂时不支持windows
    

## install
pip install -r requirements.txt



## 启动
```commandline
cd script
bash start.sh
```

## 请求
```text
http://127.0.0.1:8081/generate
请求
{
    "texts": ["你会干什么？"],
    "model": "bloom-560m",
    "params": {"max_new_tokens": 512,"do_sample": true,"temperature": 0.9,"top_p": 0.7}
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



```text
http://127.0.0.1:8081/chat
请求
{
    "texts": ["你会干什么？"],
    "model": "bloom-560m",
    "params": {"max_new_tokens": 512,"do_sample": true,"temperature": 0.9,"top_p": 0.7}
}


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



## 
    纯粹而干净的代码




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/aigc_serving&type=Date)](https://star-history.com/#ssbuild/aigc_serving&Date)

