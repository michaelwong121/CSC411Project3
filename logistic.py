from collections import OrderedDict
import re
import os
import numpy as np
from numpy import random
from shutil import copy2
import math
import tensorflow as tf

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
    
def grad_descent(x_test, y_test, x_val, y_val, x_train, y_train, nhid, alpha, \
    max_iter, mini_batch_size, lam, W0, b0, W1, b1, part):
    global train_performance, test_performance, val_performance
    
    x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        
    layer1 = tf.nn.sigmoid(tf.matmul(x, W0)+b0)
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, y_train.shape[1]])
    
    # regularization/penalty
    # according to class, weight penalty is used when the network is overfitting
    # to create overfitting in this current architecture, can increase number of 
    # neurons (nhid)?
    
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
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
        
        if part == 7 and i % 200 == 0:
            print ("i=",i)
            print ("Cost:", sess.run(reg_NLL, feed_dict={x: x_train, y_:y_train}))
            acc_tr = sess.run(accuracy,feed_dict={x: x_train, y_: y_train})
            train_performance.append(acc_tr * 100)
            print ("Train:", acc_tr)
            
            acc_t = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            test_performance.append(acc_t * 100)
            print ("Test:", acc_t)
        
            acc_v = sess.run(accuracy,feed_dict={x: x_val, y_: y_val})
            val_performance.append(acc_v * 100)
            print ("Validation:", acc_v)
        
        if part == 8 and i % 500 == 0:
            print ("i=",i)
            print ("Cost:", sess.run(reg_NLL, feed_dict={x: x_train, y_:y_train}))
            acc_tr = sess.run(accuracy,feed_dict={x: x_train, y_: y_train})
            print ("Train:", acc_tr)
            
            acc_t = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            print ("Test:", acc_t)
        
            acc_v = sess.run(accuracy,feed_dict={x: x_val, y_: y_val})
            print ("Validation:", acc_v)
        
            print ("Penalty:", sess.run(decay_penalty))
               
    return sess.run(W0), sess.run(W1)
    
keyword = get_keyword_set()
x_test, y_test = setup_x_and_y('test', keyword)