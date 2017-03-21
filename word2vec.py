import numpy as np
import os
import random as rn

emb = np.load("embeddings.npz")["emb"] # 41524 x 128

# ind is a dict that map each indixes to words
ind = np.load("embeddings.npz")["word2ind"].flatten()[0]


# input to neural network is (word2vec(w), word2vec(t))
# word2vec(w) is 128 long

# when making training set, take a word w, then
# pick a word t next to w and form tuple (w, t). Label it positive
# pick a word t2 randomly from the file and form tuple (w, t2). Label it negative
# if t2 is adjacent to w, need to handle it properly

# need to build up the context, target pair. Can use subset of the review dataset


# 0 randomly pick words
# same amount of 0 as 1
# put in dupliciates multiple times 
# use 500 files for now

# invert the word to index mapping
inv_ind = {v: k for k, v in ind.items()}


def get_sorted_embedding(word1, word2):
    this_x = []
    if (word1 < word2):
        this_x = emb[inv_ind[word1]] + emb[inv_ind[word2]]
    else:
        this_x = emb[inv_ind[word2]] + emb[inv_ind[word1]]
    return this_x

def get_training_set(subst_no):
    
    np.random.seed(0)
    
    x = np.zeros((0,256))
    y = np.zerors((0,1))
    idx = rn.sample(range(1000), subset_no) # take n random reviews from the file
    dir = os.listdir('txt_sentoken/pos')
    for i in range(subset_no):
        with open('txt_sentoken/pos/' + dir[idx[i]]) as f:
            file_list = re.split('\W+', f.read().lower())
            file_list.remove('')
            for k in range(len(file_list) - 1): # -1 because last word dont have next word
                this_x = get_sorted_embedding(file_list[k], file_list[k+1])
                x = np.vstack((x, this_x))
                y = np.vstack((y, [1]))
                
                rand_idx = k
                while (abs(rand_idx - k) <= 1):
                    rand_idx = rn.randint(0, len(file_list)-1)
                this_x = get_sorted_embedding(file_list[k], file_list[rand_idx])
                x = np.vstack((x, this_x))
                y = np.vstack((y, [0]))
    return x, y

