# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/25 11:16
from enum import Enum
from collections import namedtuple

class WorkMode(Enum):
    STANDORD_HF = 0
    DS = 1
    ACCELERATE = 2

class ChunkData:
    idx = 0
    text = ''

    def clear(self):
        self.text = ''


class LoraModelState(Enum):
    NONE = 0
    MERGED = 1
    MERGE_AND_LOCKED = 2
    UNMERGED = 3
    DISABLED = 4



CompletionResult = namedtuple('CompletionResult', ['code', 'result','msg','complete'],defaults=(0,{},"ok",True))

#
# from threading import RLock
# class QueueData:
#     def __init__(self,maxsize=0):
#         self.queue: queues.Queue = multiprocessing.Manager().Queue(maxsize=maxsize)
#         self.lock = RLock()
#         self.token_id = 0
#
#     def put(self,obj, block=True, timeout=None):
#         self.lock.acquire()
#         self.token_id += 1
#         self.lock.release()
#         return self.queue.put(obj,block=block,timeout=timeout)
#
#     def get(self,block=True, timeout=None):
#         return self.queue.get(block=block,timeout=timeout)
#
#     def qsize(self):
#         # Raises NotImplementedError on Mac OSX because of broken sem_getvalue()
#         return self.queue.qsize()
#
#     def empty(self):
#         return self.queue.empty()
#
#     def full(self):
#         return self.queue.full()
#
#     def get_nowait(self):
#         return self.get_nowait()
#
#     def put_nowait(self, obj):
#         return self.put_nowait(obj)
#
#     def close(self):
#        self.queue.close()
#
#     def join_thread(self):
#        self.queue.join_thread()
#
#     def cancel_join_thread(self):
#        self.queue.cancel_join_thread()
