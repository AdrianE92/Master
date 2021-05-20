import torch
l1 = [1, 4, 5, 2, 3]
l2 = [1, 2, 4, 2, 3]
acc = 0
for i, j in zip(l1, l2):
    if i == j:
        acc += 1
print(acc/len(l1))