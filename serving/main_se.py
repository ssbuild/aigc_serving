# # -*- coding: utf-8 -*-
# # @Time:  23:28
# # @Author: tk
# # @File：main.py
# encode 例子
import os
import shutil
import signal
import sys
root_dir = os.path.join(os.path.dirname(__file__),"..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)
import sys,os
from se_imports import se_register_module
se_register_module(root_dir=root_dir)

from serving.serve.backend import WokerLoader
import uvicorn
from serving.config_parser.main import global_serve_args
from serving.utils import logger
from serving.serve.api_serving import global_instance, app


def remove_dir(path_dir):
    try:
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        for root, paths, files in os.walk(path_dir):
            for f in files:
                os.remove(os.path.join(root, f))
    except OSError as e:
        logger.warning("warning: {0}; path: {1}".format(path_dir, e.strerror))

if __name__ == '__main__':
    tmp_dir = os.environ.get('ZEROMQ_SOCK_TMP_DIR', '/tmp/aigc_serving')
    os.environ['ZEROMQ_SOCK_TMP_DIR'] = tmp_dir
    remove_dir(tmp_dir)

    bk_worker = WokerLoader()
    global_instance().set_mapper(bk_worker.queue)
    bk_worker.create()
    config = uvicorn.Config(app, **global_serve_args,lifespan='off')
    try:
        uvicorn.Server(config).run()
    except Exception as e:
        traceback.print_exc()
        print(e)
    # threading.main_thread().is_alive()
    # signal.pthread_kill(threading.main_thread().ident, signal.SIGTSTP)
    bk_worker.release()
