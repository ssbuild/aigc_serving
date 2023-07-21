# @Time    : 2022/3/13 11:29
# @Author  : tk
# @FileName: config.py

import os
base_path =  os.path.join(os.path.dirname(__file__),'../..')

llm_model_config = {
    "model_dir": os.path.join(base_path,'train/model.ckpt'),

}

config = dict(
    cls=dict(
        type="cls",
        max_len=400,
        model_config=llm_model_config,
    )
)