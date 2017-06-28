
# coding: utf-8

# In[ ]:

print("HI")
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

print("HI")
result = open("./C_none_result1/test_pr_12.txt")
precision = []
recall = []
print("HI")
while True:
    line = result.readline()
    if not line: break
        
    p, r = line.split(' ')
    precision.append(p)
    recall.append(r)

print("HI")
# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, lw=2, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example')
plt.legend(loc="lower left")
plt.show()


# In[ ]:



