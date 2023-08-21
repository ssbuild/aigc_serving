# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/21 9:33

from setuptools import setup, find_packages

setup(
    name = "aigc-serving",
    version = "0.0.1",
    keywords = ["aigc_serving",],
    description = "aigc_serving lightweight and efficient Language service model reasoning",
    long_description = "aigc_serving lightweight and efficient Language service model reasoning",
    license = "MIT Licence",
    url = "https://github.com/ssbuild/aigc_serving",
    author = "ssbuild",
    author_email = "9727464@qq.com",
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = [],
    scripts = [],
    entry_points = {
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)