# -*- coding: utf-8 -*-
# @Time:  21:36
# @Author: tk
# @File：qwen_conf

qwen_conf = {

    "Qwen-7B": {
        "alias": None,  # 别用
        "enable": False,
        "work_mode": "deepspeed",  # one of deepspeed,accelerate,hf    单卡建议使用 hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],

        "auto_quantize": False, # 是否自动量化模型
        "auto_merge_lora_single": True, # 当仅有一个lora, 是否自动合并成一个模型，此方法将无法调用基础模型，只能使用合并后的权重
        "max_batch_size": 1, # embdding max batch size
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
        "work_mode": "deepspeed",  # one of deepspeed,accelerate,hf    单卡建议使用 hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],

        "auto_quantize": False, # 是否自动量化模型
        "auto_merge_lora_single": True, # 当仅有一个lora, 是否自动合并成一个模型，此方法将无法调用基础模型，只能使用合并后的权重
        "max_batch_size": 1, # embdding max batch size
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

    "qwen-7b-chat-int4": {
        "enable": False,
        "work_mode": "deepspeed",  # one of deepspeed,accelerate,hf    单卡建议使用 hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],

        "auto_quantize": False, # 是否自动量化模型
        "auto_merge_lora_single": True, # 当仅有一个lora, 是否自动合并成一个模型，此方法将无法调用基础模型，只能使用合并后的权重
        "max_batch_size": 1, # embdding max batch size
        "model_config": {
            "model_type": "qwen",
            "model_name_or_path": "/data/nlp/pre_models/torch/qwen/qwen-7b-chat-int4",
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