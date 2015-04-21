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



# 用户间的相似度 (使用物项关系方面, 即两人与相同的物项集产生关联(不考虑个数), 则为100% 相似.)

def userSimilarity(train):
    import math
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u != v:
                # Intersect
                W[u][v] = len(train[u] & train[v])
                W[u][v] /= math.sqrt(len(train[u] * len(train[v]) * 1.0)
        return W


# >> 对海量用户群体  复杂度高 难以接受 (实现)
# example 新浪微博 就算只有50 M 实际用户, 那也要算一下子.

# 方法实现依据:  user - item 的正向表

# 简化 ->  建立item - user 的倒排表  引入稀疏矩阵 sparse

# 并只存储那些  "我用的物品A, 与A产生的其它用户"  C[u][v] 的数据

def UserSimilarity(train):
    from collections import defaultdict
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    # 计算用户间的关联物品
    C = defaultdict()
    N = defaultdict(int)
    for i, users in items_users.items():
        # 物项i被一群users用过
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1

    # 计算相似度
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv/math.sqrt(N[u] * N[v])
    return W



# Improved 相似度的公式 过粗, 未考虑 过热门产品(很多人都会拥有, 即使两个人都拥有也未必意为着他们二人相似.)

# 引入惩罚项, 详情可搜索 John S. Breese 公式

#

def UserSimilarity(train):
    from collections import defaultdict
    # 物品用户倒排表
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    # 计算用户间的关联物品
    C = defaultdict()
    N = defaultdict(int)
    for i, users in items_users.items():
        # 物项i被一群users用过
        for u in users:
            N[u] += 1
            for v in users:
                if u != v:
                    C[u][v] += 1 / math.log(1+ len(users))

    # calculate final similiarity matrix W
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W


# 上述内容为基于用户的协同过滤思路
# 下面是基于物品的
# usecase: customers who bought this item also bought ...


# userCF 算法 是以 物品-用户的倒排表 出发, itemCF 算法 则是以 用户-物品倒排表出发


# 优化后 带有解释的ItemCF算法
def Recommendation(train, user_id, W, K):
    from operator import itemgetter
    rank = dict()
    ru = train[user_id]
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j].weight += pi * wj
            rank[j].reason[i] = pi * wj
    return rank


