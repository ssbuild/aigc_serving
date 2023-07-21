# -*- coding: utf-8 -*-
# @Time:  18:49
# @Author: tk
# @File：evaluate
import torch
from deep_training.data_helper import ModelArguments, DataArguments, DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.moss.llm_model import MyTransformer,MossConfig,MossTokenizer
from aigc_zoo.utils.moss_generate import Generate
from serving.model_handler.base import Engine_API_Base
from serving.config.constant_map import models_info_args
class NN_DataHelper(DataHelper):pass


class Engine_API(Engine_API_Base):
    def init(self,model_config_dict):
        models_info_args['seed'] = None
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer: MossTokenizer
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer,
                                                                       config_class_name=MossConfig,
                                                                       config_kwargs={"torch_dtype": "float16"})

        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16, )
        model = pl_model.get_llm_model()
        model.eval().half().cuda()

        self.model = model
        self.tokenizer: MossTokenizer = tokenizer
        self.gen_core = Generate(model,tokenizer)

    def infer(self,input,**kwargs):
        default_kwargs = dict(
            
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        response = self.gen_core.chat(input, **kwargs)
        return response

    def generate(self,input,**kwargs):
        default_kwargs = dict(
            
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        response = self.model.generate(input, **kwargs)
        return response


if __name__ == '__main__':
    api_client = Engine_API()
    api_client.init("moss-moon-003-sft-int4")
    text_list = ["写一个诗歌，关于冬天",
                 "<|Human|>: 如果一个女性想要发展信息技术行业，她应该做些什么？<eoh>\n<|MOSS|>:",
                 ]
    for input in text_list:
        response = api_client.generate(input)
        print('input', input)
        print('output', response)
