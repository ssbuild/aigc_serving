from enum import Enum
from typing import Literal, Optional, List, Dict, Any, Union

import time
import uuid
from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int



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
    owned_by: str = "fastchat"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 2
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    gtype: Optional[str] = "increace",  # one of total,increace
    do_sample: Optional[bool] = True



    def build_query_history(self):
        prev_messages = self.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == Role.SYSTEM:
            prefix = prev_messages.pop(0).content
        else:
            prefix = ""
        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == Role.USER and prev_messages[i + 1].role == Role.ASSISTANT:
                    history.append({
                        "q": prefix + prev_messages[i].content,
                        "a": prefix + prev_messages[i + 1].content
                    })
        query = self.messages[-1].content
        return (query,history)

    def _update_params(self,r):
        params = {
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": self.frequency_penalty,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
        if self.stream:
            params["gtype"] = self.gtype
            params["n"] = self.n
        else:
            params["do_sample"] = self.do_sample

        keep_keys = [k for k, v in params.items() if v is not None]
        r["params"] = {k: params[k] for k in keep_keys}
        return r

    def build_request_chat(self):
        query,history = self.build_query_history()
        r = {
            "method": "chat",
            "model": self.model,
            "history": history,
            "query": query,
        }
        r = self._update_params(r)
        return r
    def build_request_streaming(self):
        query,history = self.build_query_history()
        r = {
            "method": "chat_stream",
            "model": self.model,
            "history": history,
            "query": query,
        }
        r = self._update_params(r)
        return r

    def build_request_generate(self):
        query,history = self.build_query_history()
        r = {
            "method": "generate",
            "model": self.model,
            "history": history,
            "texts": [query],
        }
        r = self._update_params(r)
        return r

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]]


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
    finish_reason: Optional[Literal["stop", "length"]]


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


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    engine: Optional[str] = None
    input: Union[str, List[Any]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal["stop", "length"]]


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4())}")
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
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4())}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
