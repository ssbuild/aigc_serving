# -*- coding: utf-8 -*-
# @Time:  23:28
# @Author: tk
# @File：main.py
import os
import sys
root_dir = os.path.join(os.path.dirname(__file__),"..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)

from serving.serve.api_server import main

if __name__ == '__main__':
    print('starting...')
    main()