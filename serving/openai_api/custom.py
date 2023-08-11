# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/11 15:31
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel

__all__ = [
    'CustomParams'
]

class CustomParams(BaseModel):

    adapter_name: Optional[str] = "default"
    gtype: Optional[str] = "increace",  # one of total,increace
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


    def build_query_history(self):
        raise NotImplemented

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

    def build_request_chat(self):
        r = []
        items = self.build_query_history()
        if isinstance(items,tuple):
            query, history = items
            r_ = {
                "method": "chat",
                "model": self.model,
                "history": history,
                "query": query,
            }
            r_ = self._update_params(r_)
            r.append(r_)
        else:
            for item in items:
                query, history = item
                r_ = {
                    "method": "chat",
                    "model": self.model,
                    "history": history,
                    "query": query,
                }
                r_ = self._update_params(r_)
                r.append(r_)
        return r

    def build_request_streaming(self):
        items = self.build_query_history()
        if isinstance(items, list):
            items = items[0]
        query, history = items
        r = {
            "method": "chat_stream",
            "model": self.model,
            "history": history,
            "query": query,
        }
        r = self._update_params(r)
        return r
