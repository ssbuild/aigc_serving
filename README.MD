# 简介

   aigc_serving Lightweight and efficient 大语言模型推理服务
   

# 推荐环境
   linux python >=3.8
   因对进程的管理和共享内存机制，暂时不支持windows
    

# 安装模块
pip install -r nn-serving/requirements.txt




## 回显


```commandline
cd script
bash start.sh

客户端执行
curl http://192.168.16.157:8081/predict -H "Content-Type: application/json" -X POST -d '{"param":{"mode":"cls"},"texts":["111"]}'
返回
{"param":{"mode":"cls"},"texts":["111"]}

```

