# -*- coding: utf-8 -*-
# @Time:  23:28
# @Author: tk
# @File：main.py
import sys
sys.path.append('..')
from serving.serving.api_server import runner

if __name__ == '__main__':
    runner()