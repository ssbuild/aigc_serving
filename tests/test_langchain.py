from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage, ChatMessage


def _convert_dict_to_langchain_message(_dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict["content"] or ""  # OpenAI returns None for tool invocations
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


messages =  [{"role": "user", "content": "你是谁"}]
messages = [_convert_dict_to_langchain_message(_) for _ in messages]



stream = False
llm = ChatOpenAI(
    # model_name="gpt-3.5-turbo",
    # model_name="chatglm2-6b-int4",
    model_name="qwen-7b-chat-int4",
    temperature=0,
    max_tokens=2000,
    frequency_penalty=1.01,
    presence_penalty=0,
    streaming=stream,
    model_kwargs={
        "adapter_name": None,# lora头
        "stop":  ["Observation:"],
        "top_p": 1.0,
    },

    openai_api_key = "EMPTY",
    openai_api_base = "http://192.168.2.180:8081/v1"
)

completion = llm.generate([messages],stop=None)
print(completion)
