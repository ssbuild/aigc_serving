# coding=utf8
# @Time    : 2023/9/3 1:45
# @Author  : tk
# @FileName: __init__.py

from serving.prompt.openbuddy import get_chat as get_chat_openbuddy
from serving.prompt.tiger import get_chat as get_chat_tiger
from serving.prompt.default import get_chat as get_chat_default
from serving.prompt.chatglm import get_chat as get_chat_chatglm
from serving.prompt.chatglm2 import get_chat as get_chat_chatglm2
from serving.prompt.chatyuan import get_chat as get_chat_chatyaun,postprocess as postprocess_chatyuan
from serving.prompt.causallm import get_chat as get_chat_causallm
from serving.prompt.skywork import get_chat as get_chat_skywork
from serving.prompt.bluelm import get_chat as get_chat_bluelm
from serving.prompt.yi import get_chat as get_chat_yi