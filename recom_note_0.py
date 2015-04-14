# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import os, sys


# rmse, mae Function
import math, random
from random import randint
random.seed(1428)
records = [
    ['1', 'a', 5, randint(3, 5)],
    ['1', 'b', 5, randint(3, 5)],
    ['1', 'c', 3, randint(3, 5)],
    ['1', 'a', 4, 4],
    ['1', 'b', 5, randint(2, 4)],
    ['1', 'e', 4, randint(3, 5)],
    ['1', 'c', 5, randint(3, 5)]
]
def RMSE(records):
    t = float(len(records))
    return math.sqrt( sum([ (rui-pui) ** 2 for u, i, rui, pui in records])/ t)

def MAE(records):
    t = float(len(records))
    return sum([abs(rui - pui) for u, i, rui, pui in records ]) / t


print('RMSE: ', RMSE(records))
print('MAE : ', MAE(records))
# >> RMSE 加大了对预测不准的惩罚效果.  平方级惩罚



# Recall and Precision
# Top-N 方法的召回率与准确率计算


def PrecisionRecall(test, N):
    '''
    对变量加以解释: test 测试集: 用户-推荐物品列表 的字典 ||
            N 要取的推荐阈值, int; 应小于 最短的推荐物品列表
        代码计算逻辑: 遍历每个用户的推荐结果

        从推荐系统函数(Recommend函数 即为 Top-N推荐, 对每个用户返回其热门top-n, 如果用户不在列表中, 则推荐所有项的热门top-n )
        将推荐结果 与 用户真实行为比对, 交集的元素个数即为 hit  表示推荐中标.
        召回量 += items长度  用户行为计数
        推荐长度 +=  每次都是推荐了N个 所以 +N
        最后计算各自的占比,即为
        返回项中的结果
    '''

    hit, N_recall, N_precision = 0, 0, 0
    for user, items in test.items():
        rank = Recommend(user, N)
        hit += len(rank & items)
        N_recall += len(items)
        N_precision += N

    return [hit / float(N_recall), hit / float(N_precision)]




# coverage

def entropy_gain(p):
    'p 为对应的频率分布列表'
    from math import log
    return sum([- x * log(x) for x in p])


def GiniIndex(p):
    '''
    i_j 表示 按物品流行升序排列后 第j个列表
    '''
    from operator import itemgetter
    j = 1
    n = len(p)
    G = 0
    for item, weight in sorted(p.items(), key=itemgetter(1)):
        G += (2 * j - n - 1) * weight
    return G / float(n - 1)



# Popularity
# 对所有的物品的流行度取对数( 流行度就是用的人数)
# 为了避免0, 用 1+ 处理.
def Popularity(train, test, N):
    from collections import defaultdict
    item_popularity = defaultdict(int)
    for user, item in train.items():
        for item in items.keys():
            item_popularity[item] += 1

    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= (n*1.0)
    return ret