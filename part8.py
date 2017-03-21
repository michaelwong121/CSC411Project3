from collections import OrderedDict
import re
import os
import numpy as np
from numpy import random
from shutil import copy2
import math
import tensorflow as tf
from pylab import *
import random as rn
import matplotlib.pyplot as plt
import heapq

emb = np.load("embeddings.npz")["emb"] # 41524 x 128

# ind is a dict that map each indixes to words
ind = np.load("embeddings.npz")["word2ind"].flatten()[0]

# 0 randomly pick words
# same amount of 0 as 1
# put in dupliciates multiple times 
# use 500 files for now

# invert the word to index mapping
inv_ind = {v: k for k, v in ind.items()}


def get_cosine_distance(v1, v2):
    return -np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


def get_euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def get_similar_words(input_word, number):
    global emb, ind, inv_ind
    story_emb = emb[inv_ind[input_word]]
    
    heap_cosine = []
    heap_euclidean = []
    
    for i in range(len(emb)):
        if i != inv_ind[input_word]:
            word_emb = emb[i]
            word = ind[i]
            distance_cosine = get_cosine_distance(story_emb, word_emb)
            distance_euclidean = get_euclidean_distance(story_emb, word_emb)
            heapq.heappush(heap_cosine, (distance_cosine, word))
            heapq.heappush(heap_euclidean, (distance_euclidean, word))
    
    print("cosine distance:")
    print([x[1] for x in heapq.nsmallest(number, heap_cosine)])
    print("euclidean distance:")
    print([x[1] for x in heapq.nsmallest(number, heap_euclidean)])


def operation(words, operators):
    global emb, ind, inv_ind
    word0 = emb[inv_ind[words[0]]]
    for i in range(1, len(words)):
        word = emb[inv_ind[words[i]]]
        if operators[i-1] == "+":
            word0 = word0 + word
        else:
            word0 = word0 - word
    return get_word(word0, words)


def get_word(embedding, words):
    global emb, ind, inv_ind
    min_distance = -1
    min_index = 0
    for i in range(len(emb)):
        distance_cosine = get_cosine_distance(emb[i], embedding)
        if (min_distance == -1 or distance_cosine < min_distance) and ind[i] not in words:
            min_distance = distance_cosine
            min_index = i
            
    return ind[min_index]