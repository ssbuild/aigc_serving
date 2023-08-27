# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/21 9:34
from config_template.baichuan_conf import baichuan_config
from config_template.baichuan2_conf import baichuan2_config
from config_template.bloom_conf import bloom_conf
from config_template.chatglm_conf import chatglm_conf
from config_template.chatglm2_conf import chatglm2_conf
from config_template.internlm_conf import internlm_conf
from config_template.llama_conf import llama_conf
from config_template.moss_conf import moss_conf
from config_template.opt_conf import opt_conf
from config_template.rwkv_conf import rwkv_conf
from config_template.qwen_conf import qwen_conf
from config_template.tiger_conf import tiger_conf
from config_template.xverse_conf import xverse_conf

import yaml

with open("../yamlconfig/baichuan.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(baichuan_config,f)

with open("../yamlconfig/baichuan2.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(baichuan2_config,f)


with open("../yamlconfig/bloom.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(bloom_conf,f)

with open("../yamlconfig/chatglm.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(chatglm_conf,f)

with open("../yamlconfig/chatglm2.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(chatglm2_conf,f)

with open("../yamlconfig/internlm.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(internlm_conf,f)


with open("../yamlconfig/llama.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(llama_conf,f)

with open("../yamlconfig/moss.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(moss_conf,f)


with open("../yamlconfig/opt.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(opt_conf,f)


with open("../yamlconfig/rwkvf.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(rwkv_conf,f)


with open("../yamlconfig/qwen.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(qwen_conf,f)


with open("../yamlconfig/tiger.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(tiger_conf,f)
    
with open("../yamlconfig/xverse.yaml", mode='w',encoding='utf-8',newline='\n') as f:
    yaml.dump(xverse_conf,f)