# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/21 10:03

def load_config():
    from config.pyconfig.baichuan_conf import baichuan_config
    from config.pyconfig.baichuan2_conf import baichuan2_config
    from config.pyconfig.bloom_conf import bloom_conf
    from config.pyconfig.chatglm_conf import chatglm_conf
    from config.pyconfig.chatglm2_conf import chatglm2_conf
    from config.pyconfig.internlm_conf import internlm_conf
    from config.pyconfig.llama_conf import llama_conf
    from config.pyconfig.moss_conf import moss_conf
    from config.pyconfig.opt_conf import opt_conf
    from config.pyconfig.rwkv_conf import rwkv_conf
    from config.pyconfig.qwen_conf import qwen_conf
    from config.pyconfig.tiger_conf import tiger_conf
    from config.pyconfig.xverse_conf import xverse_conf

    def _update_config(group_conig, dict_objs):
        for config in group_conig.values():
            if not isinstance(config, dict):
                continue
            if not "model_config" in config:
                continue
            if not config['enable']:
                continue
            dict_objs.update(config)

    all_config = {}

    _update_config(baichuan_config, all_config)
    _update_config(baichuan2_config, all_config)
    _update_config(bloom_conf, all_config)
    _update_config(chatglm_conf, all_config)
    _update_config(chatglm2_conf, all_config)
    _update_config(internlm_conf, all_config)
    _update_config(llama_conf, all_config)
    _update_config(moss_conf, all_config)
    _update_config(opt_conf, all_config)
    _update_config(rwkv_conf, all_config)
    _update_config(qwen_conf, all_config)
    _update_config(tiger_conf, all_config)
    _update_config(xverse_conf, all_config)
    return all_config