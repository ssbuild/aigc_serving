import json
import math

import openai
from scipy import integrate


# 新版本
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.2.180:8081/v1"
openai.api_base = "http://101.42.176.124:8081/v1"
model = "chatglm2-6b-int4"
model = "qwen-7b-chat-int4"

def calculate_quad(formula_str: str, a: float, b: float) -> float:
    """ 计算数值积分 """
    return integrate.quad(eval('lambda x: ' + formula_str), a, b)[0]


def calculate_sqrt(y: float) -> float:
    """ 计算平方根 """
    return math.sqrt(y)


class QuadCalculator:
    def __init__(self):
        self.functions = [
            {
                "name": "calculate_quad",
                "description": "calculate_quad是一个可以计算给定区间内函数定积分数值的工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "formula_str": {
                            "type": "string",
                            "description": "一个数学函数的表达式，例如x**2",
                        },
                        "a": {
                            "type": "string",
                            "description": "积分区间的左端点，例如1.0",
                        },
                        "b": {
                            "type": "string",
                            "description": "积分区间的右端点，例如5.0",
                        },
                    },
                    "required": ["formula_str", "a", "b"],
                },
            },
            {
                "name": {
                    "name_for_human":
                        "平方根计算器",
                    "name_for_model":
                        "calculate_sqrt"
                },
                "description": "计算一个数值的平方根。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "y": {
                            "type": "string",
                            "description": "被开方数",
                        },
                    },
                    "required": ["y"],
                },
            },
        ]

    def run(self, query: str) -> str:
        # Step 1: send the conversation and available functions to model
        messages = [{"role": "user", "content": query}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            functions=self.functions,
            stop=["Observation:"]
        )

        while True:
            if response["choices"][0]["finish_reason"] == "stop":
                answer = response["choices"][0]["message"]["content"]
                print(f"Model output: {answer}")
                j = answer.rfind("Final Answer:")
                return answer[j + 14:] if answer else answer

            elif response["choices"][0]["finish_reason"] == "function_call":
                response_message = response["choices"][0]["message"]
                # Step 2: check if model wanted to call a function
                if response_message.get("function_call"):
                    print(f"Function call: {response_message['function_call']}")
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors
                    available_functions = {
                        "calculate_quad": calculate_quad,
                        "calculate_sqrt": calculate_sqrt,
                    }

                    function_name = response_message["function_call"]["name"]
                    fuction_to_call = available_functions[function_name]
                    function_args = json.loads(response_message["function_call"]["arguments"])
                    print(f"Function args: {function_args}")

                    for k in ["a", "b", "y"]:
                        if k in function_args:
                            function_args[k] = float(function_args[k])
                    function_response = fuction_to_call(**function_args)
                    print(f"Function response: {function_response}")

                    # Step 4: send the info on the function call and function response to model
                    messages.append(response_message)  # extend conversation with assistant's reply
                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response,
                        }
                    )  # extend conversation with function response

                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=0,
                        stop=["Observation:"],
                    )  # get a new response from model where it can see the function response
            else:
                break


if __name__ == '__main__':


    query = "函数f(x)=x**2在区间[0,5]上的定积分是多少？其平方根又是多少？"

    calculator = QuadCalculator()
    answer = calculator.run(query)
    print(answer)
