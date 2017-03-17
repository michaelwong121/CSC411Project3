from collections import OrderedDict
import re
import os
import numpy as np
from numpy import random
from shutil import copy2
import math

def get_keyword_set():
    set = ['pos', 'neg']
    keyword = []
    for dir in set:
        path = "txt_sentoken/" + dir
        for file in os.listdir(path):
            with open(path + '/' + file) as f:
                file_list = re.split('\W+', f.read().lower())
                file_list.remove('')
                keyword += file_list
    return list(OrderedDict.fromkeys(keyword))
    
def setup_x_and_y(path, keyword):
    x_shape = len(keyword)
    x = np.zeros((0, x_shape))
    y = np.zeros((0, 1))
    this_x = np.zeros((1, x_shape))
    
    set = ['pos', 'neg']
    for dir in set:
        for file in os.listdir(path + '/' + dir):
            with open(path + '/' + dir + '/' + file) as f:
                file_list = re.split('\W+', f.read().lower())
                file_list.remove('')
                for word in file_list:
                    this_x[0][keyword.index(word)] = 1
                x = np.vstack((x, this_x))
                if (dir == "pos"):
                    y = np.vstack((y, 1))
                else:
                    y = np.vstack((y, 0))
                this_x = np.zeros((1, x_shape)) #reset
    return x, y
    
keyword = get_keyword_set()
x_train, y_train = setup_x_and_y('test', keyword)