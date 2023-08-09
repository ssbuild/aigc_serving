# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/8 16:55
import typing

import numpy as np
import torch
from transformers import PreTrainedTokenizer, LogitsProcessorList, LogitsProcessor


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

    def __init__(self, stop_words_ids: typing.Iterable[typing.Iterable[int]], eos_token_id: int):

        if not isinstance(stop_words_ids, typing.List) or len(stop_words_ids) == 0:
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

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: typing.List[int]) -> bool:
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

    def _calc_stopped_samples(self, prev_input_ids: typing.Iterable[int]) -> typing.Iterable[int]:
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

def preprocess_input_args(tokenizer: PreTrainedTokenizer,args_dict: dict):
    stop = args_dict.pop('stop',None)
    if stop is None:
        return args_dict

    logits_processor = args_dict.pop("logits_processor",None)
    if logits_processor is None:
        logits_processor = LogitsProcessorList()

    stop_words_ids = None
    if isinstance(stop,list):
        stop_words_ids = []
        for s in stop:
            if s is not None:
                token_ids = tokenizer.encode(s,add_special_tokens=False)
                stop_words_ids.append(token_ids)

    elif isinstance(stop,str):
        stop_words_ids = [tokenizer.encode(stop,add_special_tokens=False)]

    if stop_words_ids:
        stop_words_logits_processor = StopWordsLogitsProcessor(
            stop_words_ids=stop_words_ids,
            eos_token_id=tokenizer.eos_token_id,
        )
        logits_processor.append(stop_words_logits_processor)

    if len(logits_processor):
        args_dict["logits_processor"] = logits_processor
    return args_dict


def flat_input(ids: typing.Union[typing.List,int]):
    if isinstance(ids,int):
        return [ids]
    ids_ = []
    for i in ids:
        if isinstance(i,list):
            ids_.extend(i)
        else:
            ids_.append(i)
    return ids_
