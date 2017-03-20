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

def get_keyword_set():
    set = ['pos', 'neg']
    keyword = []
    for dir in set:
        path = "train/" + dir
        for file in os.listdir(path):
            with open(path + '/' + file) as f:
                file_list = re.split('\W+', f.read().lower())
                file_list.remove('')
                keyword += file_list
    return list(OrderedDict.fromkeys(keyword))
    
def setup_x_and_y(path, keyword):
    # use multinomial for y instead
    x_shape = len(keyword)
    x = np.zeros((0, x_shape))
    y = np.zeros((0, 2))
    this_x = np.zeros((1, x_shape))
    
    set = ['pos', 'neg']
    for dir in set:
        for file in os.listdir(path + '/' + dir):
            with open(path + '/' + dir + '/' + file) as f:
                file_list = re.split('\W+', f.read().lower())
                file_list.remove('')
                for word in file_list:
                    if word in keyword:
                        this_x[0][keyword.index(word)] = 1
                x = np.vstack((x, this_x))
                one_hot = np.zeros(2)
                if (dir == "pos"):
                    one_hot[0] = 1
                else:
                    one_hot[1] = 1
                y = np.vstack((y, one_hot))
                this_x = np.zeros((1, x_shape)) #reset
    return x, y

def get_train_batch(n, x_train, y_train): # n = image per actor in batch
    x = zeros((0,x_train.shape[1]))
    y = zeros((0,y_train.shape[1]))
    
    idx = rn.sample(range(x_train.shape[0]), n)
    
    for k in range(n):
        x = vstack((x, x_train[idx[k]]))
        y = vstack((y, y_train[idx[k]]))
    return x, y


def grad_descent(x_test, y_test, x_val, y_val, x_train, y_train, alpha, \
    max_iter, print_iter, mini_batch_size, lam, W0, b0):
    #global train_performance, test_performance, val_performance
    
    x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        
    layer1 = tf.nn.sigmoid(tf.matmul(x, W0)+b0)
    
    y = tf.nn.softmax(layer1)
    y_ = tf.placeholder(tf.float32, [None, y_train.shape[1]])
    
    # regularization/penalty
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    train_step = tf.train.AdamOptimizer(alpha).minimize(reg_NLL)
    
    # Done setting up architecture. Actually run the network now
    # init will init W0, B0, W1, B1 to random value
    #init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    for i in range(max_iter+1):
        # <-change size of mini batch here. Max is 75 for now
        batch_xs, batch_ys = get_train_batch(mini_batch_size, x_train, y_train) 
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        
        if i % print_iter == 0:
            print ("i=",i)
            print ("Cost:", sess.run(reg_NLL, feed_dict={x: x_train, y_:y_train}))
            acc_tr = sess.run(accuracy,feed_dict={x: x_train, y_: y_train})
            print ("Train:", acc_tr)
            
            acc_t = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            print ("Test:", acc_t)
        
            acc_v = sess.run(accuracy,feed_dict={x: x_val, y_: y_val})
            print ("Validation:", acc_v)
        
            print ("Penalty:", sess.run(decay_penalty))
               
    return sess.run(W0)

keyword = get_keyword_set() # get training set keyword
if not os.path.exists("part4_x_train.txt"):
    x_train, y_train = setup_x_and_y('train', keyword)
    np.savetxt("part4_x_train.txt", x_train)
    np.savetxt("part4_y_train.txt", y_train)
else:
    x_train = np.loadtxt("part4_x_train.txt")
    y_train = np.loadtxt("part4_y_train.txt")


if not os.path.exists("part4_x_train.txt"):
    x_test, y_test = setup_x_and_y('test', keyword)
    np.savetxt("part4_x_test.txt", x_test)
    np.savetxt("part4_y_test.txt", y_test)
else:
    x_test = np.loadtxt("part4_x_test.txt")
    y_test = np.loadtxt("part4_y_test.txt")
    
if not os.path.exists("part4_x_train.txt"):
    x_val, y_val = setup_x_and_y('validation', keyword)
    np.savetxt("part4_x_val.txt", x_val)
    np.savetxt("part4_y_val.txt", y_val)
else:
    x_val = np.loadtxt("part4_x_val.txt")
    y_val = np.loadtxt("part4_y_val.txt")
    
print("Done loading")

    
alpha = 0.00001
max_iter = 30000      
print_iter = 500 # print every 500 iterations
mini_batch_size = 50
lam = 0.000001

np.random.seed(100)
W0 = tf.Variable(np.random.normal(0.0, 0.1, \
    (36307, 2)).astype(float32))
np.random.seed(101)
b0 = tf.Variable(np.random.normal(0.0, 0.1, \
    (2)).astype(float32))

grad_descent(x_test, y_test, x_val, y_val, x_train, y_train, alpha, \
    max_iter, print_iter, mini_batch_size, lam, W0, b0)
