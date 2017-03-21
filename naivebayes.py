from collections import Counter
from collections import OrderedDict
import re
import os
import numpy as np
from numpy import random
from shutil import copy2
import math
import heapq

P_pos = 0
P_neg = 0

train_pos = {}
train_neg = {}
train_total = {}

count_train_pos = 0
count_train_neg = 0
count_train_total = 0


def main():
    print("Part 1:")
    part1()
    print("-----------------------------")
    print("Part 2:")
    part2()
    print("-----------------------------")
    print("Part 3:")
    part3()
    print("-----------------------------")
    print("Part 6 naive:")
    part6_naive()
    

def part1():
    
    # Count words: best, horrendous, awful in positive and negative reviews
    
    dict_pos = {}
    dict_neg = {}
    word_list = ["best", "horrendous", "awful"]
    for word in word_list:
        dict_pos[word] = 0
        dict_neg[word] = 0
    
    for file in os.listdir('txt_sentoken/pos'):
        with open('txt_sentoken/pos/' + file) as f:
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            file_list = list(OrderedDict.fromkeys(file_list))
            for word in word_list:
                if word in file_list:
                    dict_pos[word] += 1
    
    for file in os.listdir('txt_sentoken/neg'):
        with open('txt_sentoken/neg/' + file) as f:
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            file_list = list(OrderedDict.fromkeys(file_list))
            for word in word_list:
                if word in file_list:
                    dict_neg[word] += 1
    
    print(dict_pos)
    print(dict_neg)


def part2():
    
    init()
    
    m = 0.2
    k = 290

    train_per = get_performance('train', m, k)
    val_per = get_performance('validation', m, k)
    test_per = get_performance('test', m, k)
    
    print ("Training performance: "+str(train_per)+"%")
    print ("Validation performance: "+str(val_per)+"%")
    print ("Test performance: "+str(test_per)+"%")


def part3():
    
    global train_pos, train_neg, train_total
    
    if len(train_total) == 0:
        init()
    
    m = 0.2
    k = 290
    
    heap_pos = []
    heap_neg = []
    
    for word in train_pos:
        P_pos_given_word = P_class_given_word(1, word, m, k)
        heapq.heappush(heap_pos, (P_pos_given_word, word))
    
    for word in train_neg:
        P_neg_given_word = P_class_given_word(0, word, m, k)
        heapq.heappush(heap_neg, (P_neg_given_word, word))
        
    top_ten_pos = heapq.nlargest(10, heap_pos)
    top_ten_neg = heapq.nlargest(10, heap_neg)
    
    print("Top 10 words that predicts positive:")
    print([x[1] for x in top_ten_pos])
    print("Top 10 words that predicts negative:")
    print([x[1] for x in top_ten_neg])
    

def part6_naive():
    
    global train_pos, train_neg, train_total
    
    if len(train_total) == 0:
        init()
    
    m = 0.2
    k = 290
    
    heap = []
    dict_class = {}
    
    for word in train_total:
        word_list = [word]
        diff = log_sum_P_words_given_class(word_list, m, k, 1) - \
            log_sum_P_words_given_class(word_list, m, k, 0)
        # predicts positive
        if diff >= 0:
            dict_class[word] = 1
        # predicts negative
        else:
            dict_class[word] = 0
        heapq.heappush(heap, (abs(diff), word))
        
    top_100_diff = heapq.nlargest(100, heap)
    f = open("part6_naive.txt", "w")
    counter = 1
    # Print theta
    thetas = np.array([x[0] for x in top_100_diff])
    print("theta average = %f" % np.average(abs(thetas)))
    print("theta standard deviation = %f" % np.std(thetas))
    for x in top_100_diff:
        if (dict_class[x[1]] == 1):
            f.write("positive %s" % x[1])
        else:
            f.write("negative %s" % x[1])
        if (counter < 100):
            f.write("\n")
        counter += 1
    f.close()
    

def init():
    """Set up the global variables"""
    
    global train_pos, train_neg, train_total
    global count_train_pos, count_train_neg, count_train_total
    global P_pos, P_neg
    
    #split_dataset()
    train_pos, train_neg, count_train_pos, count_train_neg = get_wordcount('train')
    
    train_total = {k: train_pos.get(k, 0) + train_neg.get(k, 0) for k in set(train_pos) | set(train_neg)}
    
    count_train_total = count_train_pos + count_train_neg
    
    P_pos = count_train_pos / count_train_total
    P_neg = count_train_neg / count_train_total


def P_class_given_word(class_type, word, m, k):
    global train_pos, train_neg, train_total
    global count_train_pos, count_train_neg, count_train_total
    global P_pos, P_neg
    
    word_list = [word]
    P_word = math.log(train_total[word]/count_train_total)
    P_word_given_class = log_P_class_given_words(word_list, m, k, class_type)
        
    return P_word_given_class - P_word


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


def log_P_class_given_words(word_list, m, k, cla):
    global P_pos, P_neg
    P_words_cla = log_sum_P_words_given_class(word_list, m, k, cla)
    if cla == 1:
        P_class = P_pos
    else:
        P_class = P_neg
    return P_words_cla + math.log(P_class)
    
    
def log_sum_P_words_given_class(word_list, m, k, cla):
    global train_pos, train_neg, count_train_pos, count_train_neg
    log_sum = 0
    for word in word_list:
        # class is positive
        if cla == 1:
            if word in train_pos:
                word_count = train_pos[word]
            else:
                word_count = 0
            p_word = math.log((word_count + m * k)/(count_train_pos + k))
        # class is negative
        else:
            if word in train_neg:
                word_count = train_neg[word]
            else:
                word_count = 0
            p_word = math.log((word_count + m * k)/(count_train_neg + k))
        log_sum += p_word

    return log_sum


def predict_review(word_list, m, k):
    P_pos_review = log_P_class_given_words(word_list, m, k, 1)
    P_neg_review = log_P_class_given_words(word_list, m, k, 0)
    if P_pos_review >= P_neg_review:
        return 1
    else:
        return 0
        
        
def get_performance(path, m, k):
    pos_correct_count = 0
    neg_correct_count = 0
    pos_total = len(os.listdir(path + '/pos'))
    neg_total = len(os.listdir(path + '/neg'))
    for file in os.listdir(path + '/pos'):
        f = open(path + '/pos' + '/' + file)
        word_list = re.split('\W+', f.read().lower())
        word_list.remove('')
        word_list = list(OrderedDict.fromkeys(word_list))
        predict = predict_review(word_list, m, k)
        if (predict == 1):
            pos_correct_count += 1
    print(path+" positive performance: "+str(pos_correct_count*100/pos_total)+"%")
    for file in os.listdir(path + '/neg'):
        f = open(path + '/neg' + '/' + file)
        word_list = re.split('\W+', f.read().lower())
        word_list.remove('')
        word_list = list(OrderedDict.fromkeys(word_list))
        predict = predict_review(word_list, m, k)
        if (predict == 0):
            neg_correct_count += 1
    print(path+" negative performance: "+str(neg_correct_count*100/neg_total)+"%")
    return ((pos_correct_count+neg_correct_count) * 100 / (pos_total+neg_total))
    
    
if __name__ == "__main__":
    main()