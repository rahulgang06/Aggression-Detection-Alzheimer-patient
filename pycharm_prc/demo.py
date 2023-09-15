import numpy as np
from matplotlib import style
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
# create color dictionary
iris = pd.read_csv('a_data2.csv')
print(iris.head())
colors = {'|__acc':'r', '|__gyro':'g'}
# create a figure and axis
fig, ax = plt.subplots()
# plot each data-point
for i in range (0,100):
    ax.scatter(iris[1][i], iris[5][i],color=colors[iris['class'][i]])
# set a title and labels
ax.set_title('Iris Dataset')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.plot()
plt.show()
