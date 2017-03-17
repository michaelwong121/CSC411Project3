from collections import Counter
from collections import OrderedDict
import re
import os
import numpy as np
from numpy import random
from shutil import copy2
import math

np.random.seed(0)

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def split_dataset():
    # should put equal amount of reviews into each set 
    # create files
    set = ['train', 'test', 'validation']
    set2 = ['neg', 'pos']
    for s in set:
        for s2 in set2:
            file = s + '/' + s2
            create_dir(file)
    
    pos_path = "txt_sentoken/pos"
    neg_path = "txt_sentoken/neg"
    pos = os.listdir(pos_path)
    neg = os.listdir(neg_path)
    
    # less than 1000 means copy from neg file
    idx = np.array(random.permutation(2000))
    for i in range(1600):
        if (idx[i] >= 1000):
            copy2(pos_path + '/' + pos[idx[i]-1000], 'train/pos')
        else:
            copy2(neg_path + '/' + neg[idx[i]], 'train/neg')
    for i in range(1600, 1800):
        if (idx[i] >= 1000):
            copy2(pos_path + '/' + pos[idx[i]-1000], 'test/pos')
        else:
            copy2(neg_path + '/' + neg[idx[i]], 'test/neg')
    for i in range(1800, 2000):
        if (idx[i] >= 1000):
            copy2(pos_path + '/' + pos[idx[i]-1000], 'validation/pos')
        else:
            copy2(neg_path + '/' + neg[idx[i]], 'validation/neg')

def get_wordcount(path):
    pos = {}
    neg = {}
    
    # count 'each document the word is in' -> if a word appear in a doc twice, count is still 1
    # increment count by 1/len(doc) to normalize
    # remove all punctuation
    #re.split('\W+', string.lower())
    # list(OrderedDict.fromkeys(l))
    
    for file in os.listdir(path + '/pos'):
        with open(path + '/pos' + '/' + file) as f:
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            file_list = list(OrderedDict.fromkeys(file_list))
            length = len(file_list)
            for word in file_list:
                if word not in pos:
                    pos[word] = 1/length
                else:
                    pos[word] += 1/length
    for file in os.listdir(path + '/neg'):
        with open(path + '/neg' + '/' + file) as f:
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            file_list = list(OrderedDict.fromkeys(file_list))
            length = len(file_list)
            for word in file_list:
                if word not in neg:
                    neg[word] = 1/length
                else:
                    neg[word] += 1/length

    return pos, neg

#TODO this is not working right now...
def predict_review(path, train_pos, train_neg):
    m = 0.001 # m should be float
    k = 10 # k should be integer
    
    n_train_pos = sum(train_pos.values())
    n_train_neg = sum(train_neg.values())
    n_train_total = n_train_pos + n_train_neg
    
    p_train_pos = n_train_pos / n_train_total
    p_train_neg = n_train_neg / n_train_total
    
    # exp(sum(log(P(a_i|class))) + logP(class))) / exp(sum(log(P(a_i))))
    
    correct_prediction = 0
    
    for file in os.listdir(path + '/pos'):
        with open(path + '/pos' + '/' + file) as f:
            log_sum = 0
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            file_list = list(OrderedDict.fromkeys(file_list))
            length = len(file_list)
            for word in file_list:
                count = 0
                if word in train_pos:
                    count = train_pos[word]
                p = (count + m * k) / (n_train_pos + k)
                log_sum += math.log(p)
            if (math.exp(log_sum) * p_train_pos >= 0.5):
                correct_prediction += 1
    print("Pos:", correct_prediction)
    
    correct_prediction = 0
    # should we use pos or neg to predict this?
    for file in os.listdir(path + '/neg'):
        with open(path + '/neg' + '/' + file) as f:
            log_sum = 0
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            file_list = list(OrderedDict.fromkeys(file_list))
            length = len(file_list)
            for word in file_list:
                count = 0
                if word in train_pos:
                    count = train_pos[word]
                p = (count + m * k) / (n_train_pos + k)
                log_sum += math.log(p)
            if (math.exp(log_sum) * p_train_pos < 0.5):
                correct_prediction += 1
    print("Neg:", correct_prediction)





# main

#split_dataset()
train_pos, train_neg = get_wordcount('train')
#test_pos, test_neg = get_wordcount('test')
#val_pos, val_neg = get_wordcount('validation')

#predict_review('test', train_pos, train_neg)

"""
pos = {}
with open('test/pos/cv007_4968.txt') as f:
    file_list = re.split('\W+', f.read().lower())
    file_list.remove('')
    file_list = list(OrderedDict.fromkeys(file_list))
    length = len(file_list)
    for word in file_list:
        if word not in pos:
            pos[word] = 1/length
        else:
            pos[word] += length
"""

    


