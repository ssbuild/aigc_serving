# -*- coding: utf-8 -*-
# @Time:  19:26
# @Author: tk
# @File：evaluate
import torch
from deep_training.data_helper import ModelArguments, DataArguments, DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer
from aigc_zoo.utils.llm_generate import Generate
from serving.model_handler.base import EngineAPI_Base
from serving.config.constant_map import models_info_args
class NN_DataHelper(DataHelper):pass



class EngineAPI(EngineAPI_Base):
    def init_model(self):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(self.model_config_dict["model_config"], allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()

        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=config.torch_dtype, )
        model = pl_model.get_llm_model()

        model.eval().half().cuda()

        self.model = model
        self.tokenizer = tokenizer

    def generate(self,input,**kwargs):
        default_kwargs = dict(
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True, top_p=0.7, temperature=0.95,
        )
        default_kwargs.update(kwargs)
        response = Generate.generate(self.get_model(),
                                     tokenizer=self.tokenizer,
                                     query=input,**kwargs)
        return response

if __name__ == '__main__':
    api_client = EngineAPI(models_info_args['bloom-560m'])
    api_client.init()
    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]
    for input in text_list:
        response = api_client.generate(input)
        print('input', input)
        print('output', response)