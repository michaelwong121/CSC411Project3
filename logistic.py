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

#global
train_performance = []
test_performance = []
val_performance = []

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
    global train_performance, test_performance, val_performance
    
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
            train_performance.append(acc_tr * 100)
            print ("Train:", acc_tr)
            
            acc_t = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            test_performance.append(acc_t * 100)
            print ("Test:", acc_t)
        
            acc_v = sess.run(accuracy,feed_dict={x: x_val, y_: y_val})
            val_performance.append(acc_v * 100)
            print ("Validation:", acc_v)
        
            print ("Penalty:", sess.run(decay_penalty))
               
    return sess.run(W0)

keyword = get_keyword_set() # get training set keyword

if not os.path.exists("part4_theta.txt"):
    if not os.path.exists("part4_x_train.txt"):
        x_train, y_train = setup_x_and_y('train', keyword)
        np.savetxt("part4_x_train.txt", x_train)
        np.savetxt("part4_y_train.txt", y_train)
    else:
        x_train = np.loadtxt("part4_x_train.txt")
        y_train = np.loadtxt("part4_y_train.txt")
    
    
    if not os.path.exists("part4_x_test.txt"):
        x_test, y_test = setup_x_and_y('test', keyword)
        np.savetxt("part4_x_test.txt", x_test)
        np.savetxt("part4_y_test.txt", y_test)
    else:
        x_test = np.loadtxt("part4_x_test.txt")
        y_test = np.loadtxt("part4_y_test.txt")
        
    if not os.path.exists("part4_x_val.txt"):
        x_val, y_val = setup_x_and_y('validation', keyword)
        np.savetxt("part4_x_val.txt", x_val)
        np.savetxt("part4_y_val.txt", y_val)
    else:
        x_val = np.loadtxt("part4_x_val.txt")
        y_val = np.loadtxt("part4_y_val.txt")
        
    print("Done loading")

    alpha = 0.0001
    max_iter = 10000      
    print_iter = 1000 # print every 1000 iterations
    mini_batch_size = 50
    lam = 0.00001
    
    np.random.seed(100)
    W0 = tf.Variable(np.random.normal(0.0, 0.1, \
        (36307, 2)).astype(float32))
    np.random.seed(101)
    b0 = tf.Variable(np.random.normal(0.0, 0.1, \
        (2)).astype(float32))
    
    theta = grad_descent(x_test, y_test, x_val, y_val, x_train, y_train, alpha, \
        max_iter, print_iter, mini_batch_size, lam, W0, b0)
    np.savetxt("part4_theta.txt", theta)
    
    x_axis = np.arange(max_iter / print_iter + 1) * print_iter
    plt.ylim(0,110)
    plt.plot(x_axis, test_performance, label="test")
    plt.plot(x_axis, train_performance, label="training")
    plt.plot(x_axis, val_performance, label="validation")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, \
        mode="expand", borderaxespad=0.)
    plt.xlabel('Iteration')
    plt.ylabel('Correctness(%)')
    plt.savefig("part4.png")

if not os.path.exists("part6_logistic.txt"):
    theta = np.loadtxt("part4_theta.txt")
    heap = []
    dict_class = {}
    for i in range(0, theta.shape[0]):
        word = keyword[i]
        diff = theta[i, 0] - theta[i, 1]
        if diff >= 0:
            # predicts positive
            dict_class[word] = 1
        else:
            # predicts negative
            dict_class[word] = 0
        heapq.heappush(heap, (abs(diff), word))
        
    top_100_diff = heapq.nlargest(100, heap)
    f = open("part6_logistic.txt", "w")
    counter = 1
    for x in top_100_diff:
        if (dict_class[x[1]] == 1):
            f.write("positive %s" % x[1])
        else:
            f.write("negative %s" % x[1])
        if (counter < 100):
            f.write("\n")
        counter += 1
    f.close()
    
# Compare words
f_naive = open("part6_naive.txt", "r")
f_logistic = open("part6_logistic.txt", "r")

positive_words = {}
negative_words = {}
positive_words["naive"] = []
positive_words["logistic"] = []
negative_words["naive"] = []
negative_words["logistic"] = []

for line in f_naive:
    words = line.split()
    if words[0] == "positive":
        positive_words["naive"].append(words[1])
    else:
        negative_words["naive"].append(words[1])
        
for line in f_logistic:
    words = line.split()
    if words[0] == "positive":
        positive_words["logistic"].append(words[1])
    else:
        negative_words["logistic"].append(words[1])

common_positive_words = set(positive_words["naive"]).intersection(positive_words["logistic"])
common_negative_words = set(negative_words["naive"]).intersection(negative_words["logistic"])

print("Common positive words:")
print(common_positive_words)
print("Common negative words:")
print(common_negative_words)
print("Number of naive positive words = "+str(len(positive_words["naive"])))
print("Number of logistic positive words = "+str(len(positive_words["logistic"])))
print("Number of naive negative words = "+str(len(negative_words["naive"])))
print("Number of logistic negative words = "+str(len(negative_words["logistic"])))
print("Number of common positive words = "+str(len(common_positive_words)))
print("Number of common negative words = "+str(len(common_negative_words)))
f_naive.close()
f_logistic.close()