# -*- coding: utf-8 -*-
# @Time:  10:34
# @Author: tk
# @File：loaders
import json
import os


def load_lora_config(ckpt_dir):
    from deep_training.nlp.models.petl import PetlArguments,LoraConfig, AdaLoraConfig, IA3Config
    with open(os.path.join(ckpt_dir, 'adapter_config.json'), mode='r', encoding='utf-8') as f:
        jd = json.loads(f.read())
    peft_type = jd.pop('peft_type', None)
    is_peft =False
    if peft_type is not None:
        jd.pop('auto_mapping', None)
        jd.pop('task_type',None)
        jd.pop('revision',None)
        is_peft = True
        peft_type: str
        peft_type = peft_type.lower()
        assert peft_type in ['lora', 'adalora', 'ia3']
        jd["with_lora"] = True
        if peft_type == 'lora':
            lora_config = LoraConfig(**jd)
        elif peft_type == 'adalora':
            lora_config = AdaLoraConfig(**jd)
        else:
            lora_config = IA3Config(**jd)
    else:
        lora_config = PetlArguments.from_pretrained(ckpt_dir)
    return lora_config,is_peft
