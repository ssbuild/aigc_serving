# -*- coding: utf-8 -*-
# @Time:  21:36
# @Author: tk
# @File：qwen_conf

qwen_conf = {

    "Qwen-7B": {
        "enable": False,
        "work_mode": "deepspeed",  # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],

        "auto_quantize": True, # 是否自动量化模型
        "model_config": {
            "model_type": "qwen",
            "model_name_or_path": "/data/nlp/pre_models/torch/qwen/Qwen-7B",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
            "lora": {
                # 多个 lora , adapter_name: lora weight dir
                # "default": "/data/nlp/pre_models/torch/lora_path_dir",
                # "your_adapter_name": "/data/nlp/pre_models/torch/your_adapter_dir",
            }
        }

    },
    "Qwen-7B-Chat": {
        "enable": False,
        "work_mode": "deepspeed",  # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],

        "auto_quantize": True, # 是否自动量化模型
        "model_config": {
            "model_type": "qwen",
            "model_name_or_path": "/data/nlp/pre_models/torch/qwen/Qwen-7B-Chat",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
            "lora": {
                # 多个 lora , adapter_name: lora weight dir
                # "default": "/data/nlp/pre_models/torch/lora_path_dir",
                # "your_adapter_name": "/data/nlp/pre_models/torch/your_adapter_dir",
            }
        }

    },

}