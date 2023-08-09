# -*- coding: utf-8 -*-
# @Time:  23:28
# @Author: tk
# @File：main.py
import os
import shutil
import signal
import sys
root_dir = os.path.join(os.path.dirname(__file__),"..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)

import time
import uvicorn
from config.main import global_serve_args
from serving.utils import logger
from serving.serve.api_serving import global_instance,app

def remove_dir(path_dir):
    try:
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        for root,paths,files in os.walk(path_dir):
            for f in files:
                os.remove(os.path.join(root,f))
    except OSError as e:
        logger.warning("warning: {0}; path: {1}".format(path_dir, e.strerror))


if __name__ == '__main__':
    tmp_dir = './tmp'
    remove_dir(tmp_dir)
    os.environ['ZEROMQ_SOCK_TMP_DIR'] = tmp_dir

    global_instance().work_node.create()
    config = uvicorn.Config(app, **global_serve_args,lifespan='off')
    try:
        uvicorn.Server(config).run()
    except Exception as e:
        print(e)
    # threading.main_thread().is_alive()
    # signal.pthread_kill(threading.main_thread().ident, signal.SIGTSTP)
    global_instance().work_node.release()
