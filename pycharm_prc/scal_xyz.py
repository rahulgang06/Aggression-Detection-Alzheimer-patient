import numpy as np
from matplotlib import style
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

style.use('ggplot')
sc = StandardScaler()

df = pd.read_csv('punchtest.csv')
# Feature Scaling
df[["|__acc","|__acc-x","|__acc-y","|__acc-z","|__gyro","|__gyro-x","|__gyro-y","|__gyro-z"]] = sc.fit_transform(df[["|__acc","|__acc-x","|__acc-y","|__acc-z","|__gyro","|__gyro-x","|__gyro-y","|__gyro-z"]])
# df = df.set_index()
print(df)

# df_1 = df[["|","|__acc"]]
# df_2 = df[["|","|__gyro"]]

# df = pd.concat([df_1, df_2], axis=1)
# print(df)

df.plot()
plt.show()
print(df.corr())

# df_1=df_1.dropna()
# df_2=df_2.dropna()

# df_1 = df_1.groupby("|").mean()
# df_2 = df_2.groupby("|").mean()
# print(df_1)
# df = pd.concat([df_1, df_2], axis=1)
# print(df)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(df_1)
# X_test = sc.transform(df_2)
# # print(X_train)
# # print(X_test)
# # df.to_csv(r'C:\Users\Gang\Desktop\new_data\data.csv')
# plt.plot(X_train,X_test)
# # df_2.plot()
# plt.show()
