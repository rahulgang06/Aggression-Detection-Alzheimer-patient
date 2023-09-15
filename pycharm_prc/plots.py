import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd


style.use('ggplot')

# x,y = np.loadtxt('fitbitodu-export.csv',unpack=True,delimiter=',')

#plt.plot(x,y)

#plt.bar(x,y)
#plt.show()

df = pd.read_csv('117-152..csv')
df["|"] = df["|"].apply(lambda x: x.split()[0])
print(df.columns)
df_1 = df[["|","|__acc"]]
df_2 = df[["|","|__gyro"]]

# Feature Scaling



df_1=df_1.dropna()
df_2=df_2.dropna()

df_1 = df_1.groupby("|").mean()
df_2 = df_2.groupby("|").mean()
print(df_1)
print(df_2)
df = pd.concat([df_1, df_2], axis=1)
# print(df)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(df_1)
# X_test = sc.transform(df_2)
# print(X_train)
# print(X_test)
df.to_csv(r'C:\Users\Gang\Desktop\new_data\data1.csv')
# df.plot()
# df_2.plot()
# plt.show()




