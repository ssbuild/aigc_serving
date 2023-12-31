import json
from typing import Tuple, Union, List
from serving.openai_api.openai_api_protocol import ChatMessage

__all__ = [
    'get_react_prompt_for_qwen'
]

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""




def get_react_prompt_for_qwen(messages: List[ChatMessage], functions=None, function_call="auto"):
    if functions is not None:
        if "name" in functions[0]:
            new_function = []
            for info in functions:
                new_info = {}
                if isinstance(info["name"], dict):
                    new_info.update(info["name"])
                else:
                    new_info["name_for_model"] = info["name"]
                    new_info["name_for_human"] = info["name"]

                required = info["parameters"]["required"]
                new_info["description_for_model"] = info["description"]
                new_info["parameters"] = []
                for name, p in info["parameters"]["properties"].items():
                    new_info["parameters"].append(
                        {
                            "name": name,
                            "description": p.get("description", ""),
                            "required": name in required,
                            "schema": {
                                "type": p.get("type", "string"),
                            }
                        }
                    )

                new_function.append(new_info)
            functions = new_function
        elif "name_for_model" in functions[0]:
            new_function = []
            for info in functions:
                new_info = {}
                new_info["name_for_model"] = info["name_for_model"]
                new_info["name_for_human"] = info["name_for_human"]
                new_info["description_for_model"] = info["description_for_model"]
                new_info["parameters"] = []
                params = new_info["parameters"]
                for param_ in info["parameters"]:
                    params.append({
                        "name": param_["name"],
                        "description": param_["description"],
                        "required": param_["required"],
                        "schema": param_.get("schema",{}),

                    })
                new_function.append(new_info)
            functions = new_function

    else:
        for message in messages:
            if message.functions:
                functions = message.functions
                break

    if function_call != "auto" and isinstance(function_call, dict):
        functions = [info for info in functions if info["name_for_model"] in [function_call["name_for_model"]]]


    tool_descs, tool_names = [], []
    for info in functions:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info["name_for_model"],
                name_for_human=info["name_for_human"],
                description_for_model=info["description_for_model"],
                parameters=json.dumps(info["parameters"], ensure_ascii=False))
        )
        tool_names.append(info["name_for_model"])

    tool_descs = "\n\n".join(tool_descs)
    tool_names = ",".join(tool_names)

    messages_new = []
    for message in messages:
        ret = ""
        role, content = message.role, message.content
        if role == "user":
            ret = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=content)
            messages_new.append(ChatMessage(**{
                "role": role,
                "content": ret
            }))
        elif role == "assistant":
            if message.function_call and isinstance(message.function_call,dict):
                thought = message.function_call["thought"]
                function_name = message.function_call["name"]
                arguments = message.function_call["arguments"]

                if thought is not None:
                    ret += f"\nThought: {thought.strip()}"

                ret += f"\nAction: {function_name.strip()}"
                ret += f"\nAction Input: {arguments.strip()}"
            messages_new.append(ChatMessage(**{
                "role": role,
                "content": ret
            }))
        elif role == "function" or role == "observation":
            ret += f"\nObservation: output of {message.name} is {str(content).strip()}"
            messages_new.append(ChatMessage(**{
                "role": role,
                "content": ret
            }))


    return messages_new, functions


def parse_qwen_plugin_call(text: str) -> Union[Tuple[str, str , str], None]:
    t = text.rfind('Thought:')
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        thought = text[t + len("Thought:"): i].strip() if t >= 0 else None
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return thought,plugin_name, plugin_args
    return None
