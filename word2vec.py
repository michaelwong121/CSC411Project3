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

emb = np.load("embeddings.npz")["emb"] # 41524 x 128

# ind is a dict that map each indixes to words
ind = np.load("embeddings.npz")["word2ind"].flatten()[0]

# 0 randomly pick words
# same amount of 0 as 1
# put in dupliciates multiple times 
# use 500 files for now

# invert the word to index mapping
inv_ind = {v: k for k, v in ind.items()}


def get_sorted_embedding(word1, word2):
    if (word1 < word2):
        this_x = np.concatenate((emb[inv_ind[word1]], emb[inv_ind[word2]]))
    else:
        this_x = np.concatenate((emb[inv_ind[word2]], emb[inv_ind[word1]]))
    return this_x

def get_training_set(subset_no):
    
    x = np.zeros((0,256))
    y = np.zeros((0,2))
    rn.seed(0)
    idx = rn.sample(range(1000), subset_no) # take n random reviews from the file
    print(idx)
    dir = os.listdir('txt_sentoken/pos')
    for i in range(subset_no):
        with open('txt_sentoken/pos/' + dir[idx[i]]) as f:
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            for k in range(len(file_list) - 1): # -1 because last word dont have next word
                if (file_list[k] in inv_ind) and (file_list[k+1] in inv_ind):
                    this_x = get_sorted_embedding(file_list[k], file_list[k+1])
                    x = np.vstack((x, this_x))
                    y = np.vstack((y, np.array([1,0])))
                    
                    rand_idx = k
                    while (abs(rand_idx - k) <= 1) or (file_list[rand_idx] not in inv_ind):
                        rand_idx = rn.randint(0, len(file_list)-1)
                    this_x = get_sorted_embedding(file_list[k], file_list[rand_idx])
                    x = np.vstack((x, this_x))
                    y = np.vstack((y, np.array([0,1])))
    return x, y

def get_train_batch(n, x_train, y_train):
    x = zeros((0,x_train.shape[1]))
    y = zeros((0,y_train.shape[1]))
    
    rn.seed(100)
    idx = rn.sample(range(x_train.shape[0]), n)
    
    for k in range(n):
        x = vstack((x, x_train[idx[k]]))
        y = vstack((y, y_train[idx[k]]))
    return x, y

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)
    
def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10   #EPS = 10**(-10)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        iter += 1
        if (iter % 500 == 0):
            print ("Cost:", f(x, y, t))
    return t


x_train, y_train = get_training_set(3)

print("finish loading!")

"""
x_train = x_train.T
y_train = y_train.T
t0 = np.array(np.zeros((257,), dtype=double)) # need to change to rand number
theta = grad_descent(f, df, x_train, y_train, t0, 0.000002)





"""
train_performance = []

alpha = 1e-10
max_iter = 30000      
print_iter = 1000 
mini_batch_size = 1500
lam = 0.00000

np.random.seed(100)
W0 = tf.Variable(np.random.normal(0.0, 0.1, \
    (256, 2)).astype(float32))
np.random.seed(101)
b0 = tf.Variable(np.random.normal(0.0, 0.1, \
    (2)).astype(float32))

x  = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        
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
        #print ("y: ", sess.run(y , feed_dict={x: x_train, y_:y_train}))
        #print ("argmax:", sess.run(tf.argmax(y, 1), feed_dict={x: x_train, y_:y_train}))
        #print ("prediction:", sess.run(correct_prediction, feed_dict={x: x_train, y_:y_train}))
        acc_tr = sess.run(accuracy,feed_dict={x: x_train, y_: y_train})
        #train_performance.append(acc_tr * 100)
        print ("Train:", acc_tr)
        """
        acc_t = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
        test_performance.append(acc_t * 100)
        print ("Test:", acc_t)
    
        acc_v = sess.run(accuracy,feed_dict={x: x_val, y_: y_val})
        val_performance.append(acc_v * 100)
        print ("Validation:", acc_v)
    
        print ("Penalty:", sess.run(decay_penalty))
        """