# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/25 15:25
import typing
from enum import Enum

sign = 200

def test_1():
    for i in range(3):
        yield i
    yield 1000

    print(100)

def test_2():
    return 100

def test():
    if sign == 100:
        yield 10
    return test_1() if sign == 1 else test_2()

x = test()

print(type(x))




