# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/11 15:31
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel

__all__ = [
    'CustomChatParams'
]


class CustomChatParams(BaseModel):
    model: str
    messages: List
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

    adapter_name: Optional[str] = "default"
    gtype: Optional[str] = "increace"  # one of total,increace
    do_sample: Optional[bool] = True
    nchar: Optional[int] = None
    min_length: Optional[int] = None
    min_new_tokens: Optional[int] = None
    early_stopping: Optional[Union[bool, str]] = None
    max_time: Optional[float] = None
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    penalty_alpha: Optional[float] = None
    top_k: Optional[int] = None
    epsilon_cutoff: Optional[float] = None
    eta_cutoff: Optional[float] = None
    diversity_penalty: Optional[float] = None
    encoder_repetition_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    guidance_scale: Optional[float] = None
    low_memory: Optional[bool] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Union[str, Dict[str, str]] = "auto"

#   私有成员
    _model_type: Optional[str] = None



    @property
    def model_type(self):
        return self._model_type
    def _update_params(self, r):
        params = {
            "adapter_name": self.adapter_name,
            "max_new_tokens": self.max_tokens,
            "top_p": self.top_p,
            "min_length": self.min_length,
            "min_new_tokens": self.min_new_tokens,
            "early_stopping": self.early_stopping,
            "max_time": self.max_time,
            "num_beams": self.num_beams,
            "num_beam_groups": self.num_beam_groups,
            "penalty_alpha": self.penalty_alpha,
            "top_k": self.top_k,
            "epsilon_cutoff": self.epsilon_cutoff,
            "eta_cutoff": self.eta_cutoff,
            "diversity_penalty": self.diversity_penalty,
            "encoder_repetition_penalty": self.encoder_repetition_penalty,

            "forced_bos_token_id": self.forced_bos_token_id,
            "forced_eos_token_id": self.forced_eos_token_id,
            "guidance_scale": self.guidance_scale,
            "low_memory": self.low_memory,
        }
        if self.frequency_penalty is not None and self.frequency_penalty > 0:
            params["repetition_penalty"] = self.frequency_penalty

        if self.presence_penalty is not None and self.presence_penalty > 0:
            params["presence_penalty"] = self.presence_penalty

        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty

        if self.temperature <= 0:
            self.temperature = 1

        params["temperature"] = self.temperature

        if self.stop is not None:
            params["stop"] = self.stop

        if self.stream:
            params["gtype"] = self.gtype
            params["nchar"] = self.nchar
        else:
            params["do_sample"] = self.do_sample

        keep_keys = [k for k, v in params.items() if v is not None]
        r["params"] = {k: params[k] for k in keep_keys}
        return r

    def build_messages(self):
        raise NotImplemented
    def build_request(self):
        messages_list = self.build_messages()
        if self.stream:
            messages = messages_list[0]
            r = self._update_params({
                "method": "chat_stream",
                "model": self.model,
                "messages": messages,
            })
        else:
            r = []
            for messages in messages_list:
                r.append(self._update_params({
                    "method": "chat",
                    "model": self.model,
                    "messages": messages,
                }))
        return r



class CustomEmbeddingParams(BaseModel):
    def build_data(self,max_batch_size):
        max_batch_size = max(1,max_batch_size)
        inputs = [self.input] if isinstance(self.input, str) else self.input
        batches = [
            inputs[i: min(i + max_batch_size, len(inputs))]
            for i in range(0, len(inputs), max_batch_size)
        ]
        return batches
    def build_request(self,data):
        r = {
            "method": "embedding",
            "model": self.model,
            "adapter_name": self.adapter_name,
            "query": data,
        }
        return r