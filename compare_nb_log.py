# Compare words
# Note: part6_naive.txt is generated in naivebayes.py
#       part6_logistic.txt is generated in logistic.py

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