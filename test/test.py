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
from pprint import pprint

def load_matrix():
    matrix = {}
    f = open("test.csv")
    columns = f.readline().split(',')

    for line in f:
        scores = line.split(',')
        for i in range(len(scores))[1:]:
            matrix[(scores[0], columns[i])] = scores[i].strip("\n")

    return matrix

matrix = load_matrix()
pprint(matrix)


def sim_distance(matrix, row1, row2):
    from math import sqrt
    columns = set(map(lambda l: l[1], matrix.keys()))

    si = filter(lambda l: matrix.has_key((row1, l)) and matrix[(row1, l)] != "" and matrix.has_key((row2, l)) and matrix[(row2, l)] != "", columns)

    # si = filter(lambda lambda l: matrix.has_key((row1, l)) and matrix[(row1, l)] != "" and matrix.has_key((row2, 1)) and matrix[(row2, 1)] != "", columns)

    if len(si) == 0:
        return 0
    sum_of_distance = sum([pow(float( matrix[(row1, column)]) - float(matrix[(row2, column)]), 2) for column in si ])

    return 1 /  (1 + sqrt(sum_of_distance))

# print(sim_distance(matrix, 'Kai Zhou', 'Shuai Ge'))

def top_matches(matrix, row, similarity=sim_distance):
    rows = set(map(lambda l: l[0], matrix.keys()))
    scores = [(similarity(matrix, row, r), r) for r in rows if r != row]
    scores.sort()
    scores.reverse()
    return scores


person = 'Kai Zhou'
print('top match for:', person)
print(top_matches(matrix, person))

# similar movies based on movies

def transform(matrix):
    rows = set(map(lambda l: l[0], matrix.keys()))
    columns = set(map(lambda l: l[1], matrix.keys()))

    transform_matrix = {}
    for row in rows:
        for column in columns:
            transform_matrix[(column, row)] = matrix[(row, column)]
    return transform_matrix