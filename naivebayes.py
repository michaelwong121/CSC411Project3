from collections import Counter
from collections import OrderedDict
import re
import os
import numpy as np
from numpy import random
from shutil import copy2
import math

P_pos = 0

train_pos = {}
train_neg = {}
train_total = {}

count_train_pos = 0
count_train_total = 0

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
    
    np.random.seed(0)
    
    # less than 1000 means copy from neg file
    idx_pos = np.array(random.permutation(1000))
    idx_neg = np.array(random.permutation(1000))
    for i in range(800):
        copy2(pos_path + '/' + pos[idx_pos[i]], 'train/pos')
        copy2(neg_path + '/' + neg[idx_neg[i]], 'train/neg')
    for i in range(800, 900):
        copy2(pos_path + '/' + pos[idx_pos[i]], 'test/pos')
        copy2(neg_path + '/' + neg[idx_neg[i]], 'test/neg')
    for i in range(900, 1000):
        copy2(pos_path + '/' + pos[idx_pos[i]], 'validation/pos')
        copy2(neg_path + '/' + neg[idx_neg[i]], 'validation/neg')

def get_wordcount(path):
    pos = {}
    neg = {}
    
    pos_file_count = 0
    neg_file_count = 0
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
            pos_file_count += 1/length
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
            neg_file_count += 1/length
    return pos, neg, pos_file_count, neg_file_count

def P_pos_given_words(word_list, m, k):
    global P_pos
    P_a_all = get_P_a_all(word_list)
    #print(P_a_all)
    #print(P_words_given_pos(word_list, m, k))
    return math.exp(P_words_given_pos(word_list, m, k) - P_a_all) * P_pos
    
def P_words_given_pos(word_list, m, k):
    global train_pos, count_train_pos, train_total
    log_sum = 0
    for word in train_total:
        if word in word_list:
            if word in train_pos:
                word_count_pos = train_pos[word]
            else:
                word_count_pos = 0
            p_word = math.log(word_count_pos + m * k)/(count_train_pos + k)
        else:
            if word in train_pos:
                word_count_pos = train_pos[word]
            else:
                word_count_pos = 0
            p_word = math.log(1 - (word_count_pos + m * k)/(count_train_pos + k))
        log_sum += p_word

    return log_sum

def predict_review(word_list, m, k):
    P_pos_review = P_pos_given_words(word_list, m, k)
    if P_pos_review >= 0.5:
        return 1
    else:
        return 0
        
def get_performance(path, m, k):
    correct_count = 0
    total = len(os.listdir(path + '/pos')) + len(os.listdir(path + '/neg'))
    for file in os.listdir(path + '/pos'):
        f = open(path + '/pos' + '/' + file)
        word_list = re.split('\W+', f.read().lower())
        word_list.remove('')
        word_list = list(OrderedDict.fromkeys(word_list))
        if (predict_review(word_list, m, k) == 1):
            correct_count += 1
    for file in os.listdir(path + '/neg'):
        f = open(path + '/neg' + '/' + file)
        word_list = re.split('\W+', f.read().lower())
        word_list.remove('')
        word_list = list(OrderedDict.fromkeys(word_list))
        if (predict_review(word_list, m, k) == 0):
            correct_count += 1
    return (correct_count * 100 / total)

def get_P_a_all(word_list):
    global train_total, count_train_total
    log_sum = 0.0
    for word in train_total:
        if word in word_list:
            p_word = train_total[word]/count_train_total
        else:
            p_word = 1 - train_total[word]/count_train_total
        log_sum += math.log(p_word)
    #print(log_sum)
    return log_sum

def main():
    
    global train_pos, train_neg, train_total, count_train_pos, P_pos, count_train_total
    
    #split_dataset()
    train_pos, train_neg, count_train_pos, count_train_neg = get_wordcount('train')

    train_total = {k: train_pos.get(k, 0) + train_neg.get(k, 0) for k in set(train_pos) | set(train_neg)}
    
    #print(train_total)
    
    m = 0.1 # m should be float
    k =  10 # k should be integer
    
    count_train_total = count_train_pos + count_train_neg
    
    P_pos = count_train_pos / count_train_total

    train_per = get_performance('train', m, k)
    val_per = get_performance('validation', m, k)
    test_per = get_performance('test', m, k)
    
    #print ("Training performance: "+str(train_per)+"%")
    print ("Validation performance: "+str(val_per)+"%")
    print ("Test performance: "+str(test_per)+"%")

