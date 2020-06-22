import numpy as np
from tqdm import tqdm
import pickle
from numba import jit, cuda
import json
from util import read_testset

def levenshtein2(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

@jit(nopython=True)
def levenshtein(s, t):
    ''' From Wikipedia article; Iterative with two matrix rows. '''
    if s == t: 
        return 0
    elif len(s) == 0: 
        return len(t)
    elif len(t) == 0: 
        return len(s)
    v0 = np.array([0] * (len(t) + 1))
    v1 = np.array([0] * (len(t) + 1))

    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
            
    return v1[len(t)]


def token2index(lst: list, voc: list) -> list:
    seq_index = []
    for token in lst:
        if token in voc:
            seq_index.append(voc.index(token))
        else:
            seq_index.append(3)
    lst = seq_index
    return lst

def index2token(lst: list, voc: list) -> list:
    seq_token = []
    for num in lst:
        seq_token.append(voc[num])
    lst = seq_token
    return lst

def lev_translate(inp, code_train):
    min_index = -1
    min_lev = 10000
    for index, train in enumerate(code_train):
        seq = [i for i in train if i != 0]
        steps = levenshtein(inp, seq)
        if steps < min_lev:
            min_index = index
            min_lev = steps
#    print("min_index:", min_index)
#    print("min_lev:", min_lev)
    return min_index
            