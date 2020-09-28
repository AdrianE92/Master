import os
import conllu
import matplotlib.pyplot as plt

train = os.scandir("../norec/data/conllu/train/")
dev = os.scandir("../norec/data/conllu/dev/")
test = os.scandir("../norec/data/conllu/test/")

train_rating = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
train_lng = {'nb': 0, 'nn': 0}
train_total = 0

for i in train:
    with open(i, "r") as f:
        data = f.read()
        data = data.split()
        lng = data[3]
        rating = data[7]
        train_rating[rating] += 1
        train_lng[lng] += 1
        train_total += 1


lngs = ["nb", "nn"]
ratings = ["1", "2", "3", "4", "5", "6"]
rt_values = [i for i in train_rating.values()]
lng_values = [i for i in train_lng.values()]
print(rt_values)
print(lng_values)
print(train_total)

"""
tot = 28158
train = 353, 1973, 5061, 9174, 9905, 1692
train = 27752, 406

tot = 3518
dev = 44, 186, 588, 1131, 1343, 226
dev = 3461, 57

tot = 3513
test = 25, 206, 576, 1147, 1355, 204
test = 3443, 70

plt.bar(ratings, rt_values)
plt.show()
plt.close()

plt.bar(lngs, lng_values)
plt.show()
plt.close()
"""