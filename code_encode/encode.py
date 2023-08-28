# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/28 10:52

import shutil


def test_se_project(src_dir='/home/aigc_serving', dst_dir='/home/aigc_serving_se', package_name='serving'):
    from se_imports import se_project_crypto

    # 忽略复制文件，文件对工程运行没有用
    ignore = shutil.ignore_patterns('tests', '.git', '.idea', 'setup.py','code_encode','docs','docker','config_template','assets')

    # package_name
    # 如果是pypi包，package_name 需要设置包名,否则可以设置None

    # 加密接受规则
    accept_rules = ['serving/config_parser/*',
                    'serving/config_parser/*/*',
                    'serving/config_parser/*/*/*',
                    'serving/model_handler/*',
                    'serving/model_handler/*/*',
                    'serving/model_handler/*/*/*',
                    'serving/openai_api/*',
                    'serving/openai_api/*/*',
                    'serving/openai_api/*/*/*',
                    'serving/react/*',
                    'serving/react/*/*',
                    'serving/react/*/*/*',
                    'serving/react/*/*/*/*',
                    'serving/serve/*',
                    'serving/serve/*/*',
                    'serving/serve/*/*/*',
                    'serving/utils/*',
                    'serving/utils/*/*',
                    'serving/utils/*/*/*',
                    'serving/utils/*/*/*/*',
                    'serving/workers/*',
                    'serving/workers/*/*',
                    'serving/workers/*/*/*',
                    'serving/workers/*/*/*/*',
                    'serving/main.py',]

    se_project_crypto(
        src_dir,
        dst_dir,
        is_use_root_name=False,
        autoremove_dst_exists=True,
        autoremove_dst_empty_dir=True,
        ignore=ignore,
        accept_rules=accept_rules,
        key=bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        iv=bytes([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    )



src_project = r'E:\algo_project_2023\aigc_serving'

dst_project = r'E:\tmp\aigc_serving'

test_se_project(src_project,dst_project)