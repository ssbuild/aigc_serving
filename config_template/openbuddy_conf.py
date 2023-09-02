# coding=utf8
# @Time    : 2023/9/3 3:23
# @Author  : tk
# @FileName: openbuddy_conf

openbuddy_conf = {
    "openbuddy-llama2-70b": {
        "enable": False,
        "work_mode": "accelerate",  # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],

        "auto_quantize": False,  # 是否自动量化模型
        "auto_merge_lora_single": True,  # 当仅有一个lora, 是否自动合并成一个模型，此方法将无法调用基础模型，只能使用合并后的权重
        "ntk_scale": 1,  # 扩展位置长度 1 即为 1 * 2048 不扩充 , 4 即为 4 * 2048 以此类推
        "max_batch_size": 1,  # embdding max batch size
        "model_config": {
            "model_type": "llama",
            "model_name_or_path": "/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-v10.1-bf16",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
            "lora": {
                # 多个 lora , adapter_name: lora weight dir
                # "default": "/data/nlp/pre_models/torch/lora_path_dir",
                # "your_adapter_name": "/data/nlp/pre_models/torch/your_adapter_dir",
            }
        }

    },

    "openbuddy-llama2-70b-int4": {
        "enable": False,
        "work_mode": "deepspeed",  # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],

        "auto_quantize": False,  # 是否自动量化模型
        "auto_merge_lora_single": True,  # 当仅有一个lora, 是否自动合并成一个模型，此方法将无法调用基础模型，只能使用合并后的权重
        "ntk_scale": 1,  # 扩展位置长度 1 即为 1 * 2048 不扩充 , 4 即为 4 * 2048 以此类推
        "max_batch_size": 1,  # embdding max batch size
        "model_config": {
            "model_type": "llama",
            "model_name_or_path": "/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-int4",
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