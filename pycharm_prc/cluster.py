import numpy as np
import pandas as pd
import sklearn
import pickle
from matplotlib import pyplot as plt
from matplotlib import style


data1 = pd.read_csv("a_data2.csv")
print(data1.columns)
# print(data1)
# df = data1.head()
# print(df)

summary = data1.describe()
summary = summary.transpose()
df = summary.head()
# print(df)

# pl_1 = data1["|__acc"]
# pl_1.plot()
# plt.show()


x = data1.iloc[:,[1,5]].values
print(x)
plt.rcParams['figure.figsize'] = (10,5)
style.use('ggplot')
plt.scatter(x[:,0],x[:,1],c='black',s=7)
plt.show()
