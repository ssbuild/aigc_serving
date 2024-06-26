# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/8 16:55
from typing import Iterable, List, Optional, Callable, Tuple, Union, Dict

import numpy as np
import torch
from deep_training.zoo.utils.streamgenerator import GenTextStreamer
from transformers import PreTrainedTokenizer, LogitsProcessorList, LogitsProcessor, PretrainedConfig, StoppingCriteria, \
    StoppingCriteriaList

from serving.model_handler.base.data_define import ChunkData, CompletionResult
from serving.model_handler.base.kmp import KMP


class StopWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.
    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
            )
        if any(
            any(
                (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                for token_id in stop_word_ids
            )
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )

        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert (
                len(stop_token_seq) > 0
            ), "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens) :].tolist() == tokens:
            # if tokens match
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        for prev_input_ids_slice in prev_input_ids:
            match = False
            for stop_token_seq in self.stop_words_ids:
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    # if tokens do not match continue
                    match = True
                    break
            stopped_samples.append(match)
        return stopped_samples


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int,chunk=None):

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
            )
        if any(
                any(
                    (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                    for token_id in stop_word_ids
                )
                for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )

        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert (
                    len(stop_token_seq) > 0
            ), "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )
        self.init_pos = None
        self.chunk = chunk



    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.chunk is not None:
            finished = getattr(self.chunk,'finished',False)
            if finished:
                return True

        if self.init_pos is None:
            self.init_pos = input_ids.size(1)
        return self._calc_stopped_samples(input_ids)


    def _calc_stopped_samples(self, input_ids:  torch.LongTensor) -> bool:
        if len(self.stop_words_ids) == 0:
            return False
        kmp = KMP()
        for ids_ in input_ids[self.init_pos:]:
            ids = ids_.tolist()
            for stop_token_seq in self.stop_words_ids:
                if kmp.indexOf(ids, stop_token_seq) != -1:
                    return True
            del ids
        return False


global_multinomial_fn = torch.multinomial
class GenArgs:
    def __init__(self,args_dict:Dict, this_obj,is_stream=False):
        if args_dict is None:
            args_dict = {}

        self.tokenizer: Optional[PreTrainedTokenizer] = this_obj.tokenizer
        self.config: Optional[PretrainedConfig] = this_obj.config
        self.is_stream = is_stream
        self.chunk: Optional[ChunkData] = None
        self.this_obj = this_obj

        global global_multinomial_fn
        if global_multinomial_fn is None:
            global_multinomial_fn = torch.multinomial

        if global_multinomial_fn != torch.multinomial:
            torch.multinomial = global_multinomial_fn
        # support seed
        self.multinomial_fn = torch.multinomial

        self.stop_args = args_dict.pop('stop', None)

        self.__preprocess(args_dict)




    def __del__(self):
        # restore
        if torch.multinomial != self.multinomial_fn:
            torch.multinomial = self.multinomial_fn

    def __preprocess(self, args_dict):
        if self.is_stream:
            nchar = args_dict.pop('nchar',1)
            gtype = args_dict.pop('gtype',"total")
            self.chunk = ChunkData(nchar=nchar, stop=self.stop_args, mode=gtype)

        seed = args_dict.pop('seed',None)

        #进程隔离，互不影响
        if isinstance(seed,int):
            device = self.this_obj.get_model().device
            torch.multinomial = lambda *args, **kwargs: self.multinomial_fn(*args,
                                                                     generator=torch.Generator(device=device).manual_seed(seed),
                                                                     **kwargs)
        return args_dict

    def build_args(self, args_dict):
        stop = self.stop_args
        if stop is None:
            return args_dict

        stop_words_ids = None
        if isinstance(stop,list):
            stop_words_ids = []
            for s in stop:
                if s is not None:
                    token_ids = self.tokenizer.encode(s,add_special_tokens=False)
                    stop_words_ids.append(token_ids)

        elif isinstance(stop,str):
            stop_words_ids = [self.tokenizer.encode(stop,add_special_tokens=False)]

        if stop_words_ids:
            # if config.model_type.lower() == 'qwen':
            #     args_dict["stop_words_ids"] = stop_words_ids
            # else:
            stopping_criteria = args_dict.pop("stopping_criteria", None)
            if stopping_criteria is None:
                stopping_criteria = StoppingCriteriaList()
            stop_words_criteria= StopWordsCriteria(
                stop_words_ids=stop_words_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                chunk=self.chunk,
            )
            stopping_criteria.append(stop_words_criteria)

            if len(stopping_criteria):
                args_dict["stopping_criteria"] = stopping_criteria
        return args_dict

    def get_chat_info(self,messages: List[Dict],chat_format="chat"):
        query = messages.pop(-1)["content"]
        if chat_format == "chat":
            prefix = ""
            if len(messages) > 0:
                if messages[0]["role"] == "system":
                    prefix = messages.pop(0)["content"]
            history = [(q["content"], a["content"]) for q, a in zip(messages[::2], messages[1::2])]
            if prefix:
                if history:
                    history[0] = (prefix + history[0][0],history[0][1])
                else:
                    query = prefix + query
            return (query, history)

        query = messages[0]["content"]
        return query

    def get_chat_info_with_system(self,messages: List[Dict],chat_format="chat"):
        query = messages.pop(-1)["content"]
        if chat_format == "chat":
            prefix = None
            if len(messages) > 0:
                if messages[0]["role"] == "system":
                    prefix = messages.pop(0)["content"]
            history = [(q["content"], a["content"]) for q, a in zip(messages[::2], messages[1::2])]
            return (prefix, query, history)

        query = messages[0]["content"]
        return query

    def postprocess_response(self, response):
        stops = self.stop_args
        if stops is None:
            if isinstance(self.tokenizer.eos_token,str):
                stops = [self.tokenizer.eos_token]

        if stops is not None:
            pos = [response.find(stop) for stop in stops if isinstance(stop,str)]
            pos = [_ for _ in pos if _ != -1]
            if pos:
                pos = min(pos)
                response = response[:pos]
        return response

    def get_streamer(self,skip_word_list,text_filter_fn:Optional[Callable] = None):
        def process_token_fn(text, stream_end,user_data: Tuple):
            chunk: ChunkData
            chunk,this_obj = user_data
            chunk.step(text, is_append=True)
            if chunk.can_output() or stream_end:
                text = chunk.step_text()
                if text_filter_fn:
                    text = text_filter_fn(text)
                ret = CompletionResult(result={
                    "response": text,
                    # "history": history,
                    "num_token": self.get_num_tokens()
                }, complete=False)
                this_obj.push_response(ret)

        streamer = GenTextStreamer(process_token_fn, (self.chunk,self.this_obj),
                                   tokenizer=self.tokenizer, # noqa
                                   skip_word_list=flat_input(skip_word_list),
                                   skip_prompt=True)
        return streamer

    def get_num_tokens(self):
        if self.chunk:
            return self.chunk.n_id
        return 0

    def do_final_stream(self):
        if self.chunk:
            self.this_obj.push_response(CompletionResult(result={
                "response": "",
                # "history": history,
                "num_token": self.get_num_tokens()
            }, complete=True))

def flat_input(ids: Union[List,int]):
    if isinstance(ids,int):
        return [ids]
    ids_ = []
    for i in ids:
        if isinstance(i,list):
            ids_.extend(i)
        else:
            ids_.append(i)
    return ids_
