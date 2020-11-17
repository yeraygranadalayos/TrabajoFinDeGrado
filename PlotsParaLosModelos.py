# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:48:32 2020

@author: Yeray
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sb

print(df.groupby('target').size())

df.hist()
plt.show()

sb.factorplot('target',data=df,kind="count", aspect=3)

print(confusion_matrix(y_test, predicted_y))
print(classification_report(y_test, predicted_y))

plot_confusion_matrix(y_test, predicted_y, df.target, normalize=True)

# Generate scatter plot for training data 
plt.scatter(X_train[:,0], X_train[:,1])
plt.title('Linearly separable data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()