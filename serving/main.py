# # -*- coding: utf-8 -*-
# # @Time:  23:28
# # @Author: tk
# # @File：main.py
import copy
import os
import sys
root_dir = os.path.join(os.path.dirname(__file__),"..")
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)

# cc
if os.path.exists(os.path.join(root_dir,".__data__.pys")):
    from se_imports import se_register_module
    se_register_module(root_dir=root_dir)

import traceback
import shutil
import signal
from serving.serve.backend import WokerLoader
import uvicorn
from serving.config_loader.loader import global_serve_args
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
    tmp_dir = os.environ.get('ZEROMQ_SOCK_TMP_DIR','/tmp/aigc_serving')
    os.environ['ZEROMQ_SOCK_TMP_DIR'] = tmp_dir
    remove_dir(tmp_dir)
    bk_worker = WokerLoader()

    def signal_handler(signum, frame):
        bk_worker.release()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal_handler)

    try:
        global_instance().set_mapper(bk_worker.queue)
        bk_worker.create()
        kwargs = copy.deepcopy(global_serve_args)
        kwargs.pop("api_keys",None)
        config = uvicorn.Config(app, **kwargs, lifespan='off')
        uvicorn.Server(config).run()
    except Exception as e:
        traceback.print_exc()
        print(e)
    finally:
        # threading.main_thread().is_alive()
        # signal.pthread_kill(threading.main_thread().ident, signal.SIGTSTP)
        bk_worker.release()
