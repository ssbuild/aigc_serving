# -*- coding: utf-8 -*-
# @Time:  18:46
# @Author: tk
# @File：constant_map
import socket

models_info_args = {
    "baichuan-7B":{
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "model_config" : {
            "model_type": "baichuan",
            "model_name_or_path": "/data/nlp/pre_models/torch/baichuan/baichuan-7B",
            "config_name": "/data/nlp/pre_models/torch/baichuan/baichuan-7B/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/baichuan/baichuan-7B",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }
    },

    "Baichuan-13B-Chat":{
        "enable": False,
        "work_mode": "deepspeed",  # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "baichuan",
            "model_name_or_path": "/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat",
            "config_name": "/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },

    "chatglm-6b":{
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "chatglm",
            "model_name_or_path": "/data/nlp/pre_models/torch/chatglm/chatglm-6b",
            "config_name": "/data/nlp/pre_models/torch/chatglm/chatglm-6b/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/chatglm/chatglm-6b",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }
    },
    "chatglm-6b-int4":{
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "chatglm",
            "model_name_or_path": "/data/nlp/pre_models/torch/chatglm/chatglm-6b-int4",
            "config_name": "/data/nlp/pre_models/torch/chatglm/chatglm-6b-int4/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/chatglm/chatglm-6b-int4",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },
    "chatglm-6b-int8":{
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "chatglm",
            "model_name_or_path": "/data/nlp/pre_models/torch/chatglm/chatglm-6b-int8",
            "config_name": "/data/nlp/pre_models/torch/chatglm/chatglm-6b-int8/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/chatglm/chatglm-6b-int8",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },
    "chatglm2-6b-int4" : {
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config" : {
            "model_type": "chatglm2",
            "model_name_or_path": "/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-int4",
            "config_name": "/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-int4/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/chatglm2/chatglm2-6b-int4",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }
    },
    "chatglm2-6b": {
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "chatglm2",
            "model_name_or_path": "/data/nlp/pre_models/torch/chatglm2/chatglm2-6b",
            "config_name": "/data/nlp/pre_models/torch/chatglm2/chatglm2-6b/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/chatglm2/chatglm2-6b",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },
    "bloom-560m": {
        "enable": True,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "bloom",
            "model_name_or_path": "/data/nlp/pre_models/torch/bloom/bloom-560m",
            "config_name": "/data/nlp/pre_models/torch/bloom/bloom-560m/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/bloom/bloom-560m",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },
    "bloom-1b7": {
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "bloom",
            "model_name_or_path": "/data/nlp/pre_models/torch/bloom/bloom-1b7",
            "config_name": "/data/nlp/pre_models/torch/bloom/bloom-1b7/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/bloom/bloom-1b7",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },
    "opt-350m": {
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "opt",
            "model_name_or_path": "/data/nlp/pre_models/torch/opt/opt-350m",
            "config_name": "/data/nlp/pre_models/torch/opt/opt-350m/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/opt/opt-350m",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },

    "llama-7b-hf": {
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "llama",
            "model_name_or_path": "/data/nlp/pre_models/torch/llama/llama-7b-hf",
            "config_name": "/data/nlp/pre_models/torch/llama/llama-7b-hf/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/llama/llama-7b-hf",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },

    "moss-moon-003-sft-int4":{
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "moss",
            "model_name_or_path": "/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4",
            "config_name": "/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },

    # 中英日语
    "rwkv-4-raven-3b-v12-Eng49%-Chn49%-Jpn1%-Other1%": {
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "rwkv",
            "model_name_or_path": "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12",
            "config_name": "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/rwkv_gf/rwkv-4-raven-3b-v12",
            "use_fast_tokenizer": True,
            "do_lower_case": None,
        }

    },

    "internlm-chat-7b": {
        "enable": False,
        "work_mode": "deepspeed", # one of deepspeed,accelerate,hf
        "workers": [
            {
                "device_id": [0]  # 默认启动一个worker , 使用第一块显卡
            }
        ],
        "vllm": {
            "enable": False # 暂时未支持
        },
        "model_config": {
            "model_type": "internlm",
            "model_name_or_path": "/data/nlp/pre_models/torch/internlm/internlm-chat-7b",
            "config_name": "/data/nlp/pre_models/torch/internlm/internlm-chat-7b/config.json",
            "tokenizer_name": "/data/nlp/pre_models/torch/internlm/internlm-chat-7b",
            "use_fast_tokenizer": False,
            "do_lower_case": None,
        }

    },
}


def get_free_tcp_port():
    """获取可用的端口"""
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port




for model_name,model_config in models_info_args.items():
    if not model_config['enable']:
        continue

    if model_config['work_mode'] != 'deepspeed':
        continue
    workers = model_config['workers']
    flag = False
    for worker in workers:
        if len(worker['device_id']) > 1:
            flag = True
            break
    model_config['deepspeed'] = {}
    conf = model_config['deepspeed']
    if flag:
        port = get_free_tcp_port()
        conf["MASTER_ADDR"] = "127.0.0.1"
        conf["MASTER_PORT"] = str(port)
        conf["TORCH_CPP_LOG_LEVEL"] = "INFO"





