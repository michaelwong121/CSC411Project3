import numpy as np

emb = np.load("embeddings.npz")["emb"] # 41524 x 128

# ind is a dict that map each indixes to words
ind =np. load("embeddings.npz")["word2ind"].flatten()[0]


# input to neural network is (word2vec(w), word2vec(t))
# word2vec(w) is 128 long

# when making training set, take a word w, then
# pick a word t next to w and form tuple (w, t). Label it positive
# pick a word t2 randomly from the file and form tuple (w, t2). Label it negative
# if t2 is adjacent to w, need to handle it properly

# need to build up the context, target pair. Can use subset of the review dataset