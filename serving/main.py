# -*- coding: utf-8 -*-
# @Time:  23:28
# @Author: tk
# @File：main.py
import os
import signal
import sys
import time
import uvicorn

root_dir = os.path.join(os.path.dirname(__file__),"..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)

from serving.serve.api import global_instance,app

if __name__ == '__main__':
    global_instance().work_node.create()
    config = uvicorn.Config(app, host='0.0.0.0', port=8081, workers=4, lifespan='off')
    try:
        uvicorn.Server(config).run()
    except Exception as e:
        print(e)
    # threading.main_thread().is_alive()
    # signal.pthread_kill(threading.main_thread().ident, signal.SIGTSTP)
    global_instance().work_node.release()
