# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/25 11:16
from enum import Enum
from collections import namedtuple
from typing import Union, Any, Optional, List


class WorkMode(Enum):
    STANDORD_HF = 0
    DS = 1
    ACCELERATE = 2

class ChunkMode(Enum):
    CHUNK_INCREACE = 0
    CHUNK_TOTAL = 1
    @staticmethod
    def from_string(mode):
        if mode is None:
            mode = 'total'
        mode = mode.lower()
        if mode == 'total':
            return ChunkMode.CHUNK_TOTAL
        return ChunkMode.CHUNK_INCREACE

class ChunkData:
    n_id = 0
    n_pos = 0
    n_last_pos = 0
    text = ''
    def __init__(self, nchar=1, stop=Optional[List[Union[str,int]]], mode: Union[str,ChunkMode,Any] = ChunkMode.CHUNK_INCREACE):
        self.nchar = nchar
        self.mode = ChunkMode.from_string(mode) if isinstance(mode,str) else mode
        self._is_finished = False
        self.stop = None
        if stop is not None:
            if isinstance(stop, List):
                if len(stop) > 0 and isinstance(stop[0], str):
                    self.stop = stop
            elif isinstance(stop, str):
                self.stop = [stop]

    @property
    def finished(self):
        return self._is_finished

    def step(self,words,is_append = False):
        if self._is_finished:
            return

        self.n_id += 1
        if is_append:
            self.text += words
        else:
            self.text = words

        if self.stop is not None:
            for stop in self.stop:
                if isinstance(stop, str) and stop in self.text:
                    self.text = self.text.split(stop)[0] + stop
                    self._is_finished = True
                    break

        self.n_pos = len(self.text)




    def step_text(self):
        if self.mode == ChunkMode.CHUNK_INCREACE:
            text = self.text[self.n_last_pos:]
            self.n_last_pos = self.n_pos
            return text
        return self.text

    def final_text(self):
        if self.n_last_pos == self.n_pos:
            return None
        return self.step_text()

    def can_output(self):
        if self.nchar < 0:
            return True
        return self.n_id % self.nchar == 0

    def get_text(self):
        return self.text




class LoraModelState(Enum):
    NONE = 0
    MERGED = 1
    MERGE_AND_LOCKED = 2
    UNMERGED = 3
    DISABLED = 4



CompletionResult = namedtuple('CompletionResult', ['code', 'result','msg','complete'],defaults=(0,{},"ok",True))
