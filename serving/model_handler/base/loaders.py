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
    peft_type = jd.get('peft_type', None)
    if peft_type is not None:
        peft_type: str
        peft_type = peft_type.lower()
        assert peft_type in ['lora', 'adalora', 'ia3']
        jd["with_lora"] = True
        if peft_type == 'lora':
            config = LoraConfig(**jd)
        elif peft_type == 'adalora':
            config = AdaLoraConfig(**jd)
        else:
            config = IA3Config(**jd)
    else:
        config = PetlArguments.from_pretrained(ckpt_dir)
    return config


if __name__ == '__main__':
    lora_args = load_lora_config('lora')
    print(lora_args)
