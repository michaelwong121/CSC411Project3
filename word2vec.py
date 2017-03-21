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


def get_sorted_words(word1, word2):
    if (word1 < word2):
        return word1, word2
    else:
        return word2, word1

def get_training_set(subset_no, path):
    
    x = np.zeros((0,256))
    y = np.zeros((0,2))
    rn.seed(0)
    dir = os.listdir(path + '/pos')
    idx = rn.sample(range(len(dir)), subset_no) # take n random reviews from the file
    print(idx)
    pair_list = []
    keyword = []
    for i in range(subset_no):
        with open(path + '/pos/' + dir[idx[i]]) as f:
            
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            keyword += file_list
            for k in range(len(file_list) - 1): # -1 because last word dont have next word
                if (file_list[k] in inv_ind) and (file_list[k+1] in inv_ind):
                    word1, word2 = get_sorted_words(file_list[k], file_list[k+1])
                    this_x = np.concatenate((emb[inv_ind[word1]], emb[inv_ind[word2]]))
                    x = np.vstack((x, this_x))
                    y = np.vstack((y, np.array([1,0])))
                    pair_list.append((word1, word2))
                    
    keyword = list(OrderedDict.fromkeys(keyword))
    size = x.shape[0]
    for i in range(size):
        word1 = keyword[rn.randint(0, len(keyword) - 1)]
        word2 = keyword[rn.randint(0, len(keyword) - 1)]
        word1, word2 = get_sorted_words(word1, word2)
        while word1 not in inv_ind or word2 not in inv_ind or (word1, word2) in pair_list:
            word1 = keyword[rn.randint(0, len(keyword) - 1)]
            word2 = keyword[rn.randint(0, len(keyword) - 1)]
            word1, word2 = get_sorted_words(word1, word2) 
        this_x = np.concatenate((emb[inv_ind[word1]], emb[inv_ind[word2]]))
        x = np.vstack((x, this_x))
        y = np.vstack((y, np.array([0, 1])))
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


if not os.path.exists("part7_x_train.txt"):
    x_train, y_train = get_training_set(20, 'train')
    np.savetxt("part7_x_train.txt", x_train)
    np.savetxt("part7_y_train.txt", y_train)
else:
    x_train = np.loadtxt("part7_x_train.txt")
    y_train = np.loadtxt("part7_y_train.txt")
print("finish loading training!")

if not os.path.exists("part7_x_test.txt"):
    x_test, y_test = get_training_set(4, 'test')
    np.savetxt("part7_x_test.txt", x_test)
    np.savetxt("part7_y_test.txt", y_test)
else:
    x_test = np.loadtxt("part7_x_test.txt")
    y_test = np.loadtxt("part7_y_test.txt")
print("finish loading test!")

if not os.path.exists("part7_x_val.txt"):
    x_val, y_val = get_training_set(4, 'validation')
    np.savetxt("part7_x_val.txt", x_val)
    np.savetxt("part7_y_val.txt", y_val)
else:
    x_val = np.loadtxt("part7_x_val.txt")
    y_val = np.loadtxt("part7_y_val.txt")
print("finish loading val!")


train_performance = []
test_performance = []
val_performance = []

alpha = 1e-4
max_iter = 15000      
print_iter = 1000
mini_batch_size = 500
lam = 0.00001

np.random.seed(100)
W0 = tf.Variable(np.random.normal(0.0, 0.1, \
    (256, 2)).astype(float32)/math.sqrt(256 * 2))
np.random.seed(101)
b0 = tf.Variable(np.random.normal(0.0, 0.1, \
    (2)).astype(float32)/math.sqrt(2))

x  = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        
layer1 = tf.sigmoid(tf.matmul(x, W0)+b0)

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
    batch_xs, batch_ys = get_train_batch(mini_batch_size, x_train, y_train) 
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    
    if i % print_iter == 0:
        print ("i=",i)
        print ("Cost:", sess.run(reg_NLL, feed_dict={x: x_train, y_:y_train}))
        #print ("Mat mul:", sess.run(tf.matmul(x, W0)+b0, feed_dict={x: x_train, y_:y_train}))
        #print ("y: ", sess.run(y , feed_dict={x: x_train, y_:y_train}))
        #print ("argmax:", sess.run(tf.argmax(y, 1), feed_dict={x: x_train, y_:y_train}))
        #print ("prediction:", sess.run(correct_prediction, feed_dict={x: x_train, y_:y_train}))
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

plt.figure(1)
x_axis = np.arange(max_iter / print_iter + 1) * print_iter
plt.ylim(0,110)
plt.plot(x_axis, test_performance, label="test")
plt.plot(x_axis, train_performance, label="training")
plt.plot(x_axis, val_performance, label="validation")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, \
    mode="expand", borderaxespad=0.)
plt.xlabel('Iteration')
plt.ylabel('Correctness(%)')
plt.savefig("part7.png")