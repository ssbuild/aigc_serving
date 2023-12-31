# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/11 15:31

from enum import Enum
from typing import Literal, Optional, List, Dict, Any, Union
import time
import uuid
from pydantic import BaseModel, Field
from pydantic.v1 import PrivateAttr
from transformers import GenerationConfig

from serving.openai_api.custom import CustomChatParams, CustomEmbeddingParams


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int

class Role:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class Finish:
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"






class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{str(uuid.uuid4())}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "aigc_serving"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []
    adapters: Optional[List[str]] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatFunctionCallResponse(BaseModel):
    name: str
    arguments: str
    thought: Optional[str] = None

class ChatCodeCallResponse(BaseModel):
    metadata: Optional[str]
    thought: Optional[str]
    code: Optional[str]

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str],ChatFunctionCallResponse]] = "auto"
    code_call: Optional[Union[str, Dict[str, str],ChatCodeCallResponse]] = None

class ChatCompletionRequest(CustomChatParams):
    messages: List[ChatMessage]

    @property
    def is_chat(self):
        return True

    def _build_messages(self):
        messages = [message.model_dump() for message in self.messages]
        assert self.messages[-1].role in [Role.USER,Role.OBSERVATION,Role.FUNCTION]
        return [messages]






class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "code_call"]]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    
    def json(self,*args,exclude_unset=True, ensure_ascii=False,**kwargs):
        return super().json(*args,exclude_unset=exclude_unset, ensure_ascii=ensure_ascii,**kwargs)


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class TokenCheckRequestItem(BaseModel):
    model: str
    prompt: str
    max_tokens: int


class TokenCheckRequest(BaseModel):
    prompts: List[TokenCheckRequestItem]


class TokenCheckResponseItem(BaseModel):
    fits: bool
    tokenCount: int
    contextLength: int


class TokenCheckResponse(BaseModel):
    prompts: List[TokenCheckResponseItem]


class EmbeddingsRequest(CustomEmbeddingParams):
    model: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None
    adapter_name: Optional[str] = None
    max_tokens: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo




class CompletionRequest(CustomChatParams):
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


    @property
    def is_chat(self):
        return False

    def _build_messages(self):
        if isinstance(self.prompt,str):
            self.prompt = [self.prompt]
        messages_list = [[{"role": Role.USER,"content": prompt}] for prompt in self.prompt]
        return messages_list

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[float] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]

