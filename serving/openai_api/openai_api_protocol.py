import typing
from enum import Enum
from typing import Literal, Optional, List, Dict, Any, Union

import time
import uuid
from pydantic import BaseModel, Field
from transformers import GenerationConfig

from serving.openai_api.custom import CustomParams


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class Finish(str, Enum):
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

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = "auto"


class ChatCompletionRequest(CustomParams):
    model: str
    messages: List[ChatMessage]
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
    gtype: Optional[str] = "increace",  # one of total,increace
    do_sample: Optional[bool] = True
    nchar: Optional[int] = None
    min_length: Optional[int] =None
    min_new_tokens: Optional[int] =None
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
        prev_messages = self.messages[:-1]
        if len(prev_messages) > 0 and prev_messages[0].role == Role.SYSTEM:
            prefix = prev_messages.pop(0).content
        else:
            prefix = ""

        flag = False
        history = []
        if len(prev_messages) % 2 == 0:
            for i in range(0, len(prev_messages), 2):
                if prev_messages[i].role == Role.USER and prev_messages[i + 1].role == Role.ASSISTANT:
                    history.append({
                        "q": prefix + prev_messages[i].content if not flag else prev_messages[i].content ,
                        "a": prev_messages[i + 1].content
                    })
                    flag = True
        query = prefix + self.messages[-1].content if not flag else self.messages[-1].content
        return (query,history)




class ChatFunctionCallResponse(BaseModel):
    name: str
    arguments: str
    thought: str = None



class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]


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




class CompletionRequest(CustomParams):
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

    def build_query_history(self):
        if isinstance(self.prompt,list):
            return [(_,[]) for _ in self.prompt]
        return [(self.prompt,[])]

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

