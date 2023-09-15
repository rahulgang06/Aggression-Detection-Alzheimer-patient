import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd


style.use('ggplot')

# x,y = np.loadtxt('fitbitodu-export.csv',unpack=True,delimiter=',')

#plt.plot(x,y)

#plt.bar(x,y)
#plt.show()

df = pd.read_csv('a_data1.csv')
df["|"] = df["|"].apply(lambda x: x.split()[0])
print(df.columns)
df_1 = df[["|","|__acc"]]
df_5 = df[["|","|__gyro"]]
# df_2 = df[["|","|__acc-x"]]
# df_3 = df[["|","|__acc-y"]]
# df_4 = df[["|","|__acc-z"]]
#
# df_6 = df[["|","|__gyro-x"]]
# df_7 = df[["|","|__gyro-y"]]
# df_8 = df[["|","|__gyro-z"]]




df_1=df_1.dropna()
df_5=df_5.dropna()
# df_2=df_2.dropna()
# df_3=df_3.dropna()
# df_4=df_4.dropna()
#
# df_6=df_6.dropna()
# df_7=df_7.dropna()
# df_8=df_8.dropna()



df_1 = df_1.groupby("|").mean()
df_5 = df_5.groupby("|").mean()

# df_2 = df_2.groupby("|").mean()
# df_3 = df_3.groupby("|").mean()
# df_4 = df_4.groupby("|").mean()
# df_6 = df_6.groupby("|").mean()
# df_7 = df_7.groupby("|").mean()
# df_8 = df_8.groupby("|").mean()



print(df_1)
print(df_5)

# print(df_2)
# print(df_3)
# print(df_4)
# print(df_6)
# print(df_7)
# print(df_8)
df = pd.concat([df_1,df_5], axis=1)
print(df)
df.fillna(df.mean(), inplace=True)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(df_1)
# X_test = sc.transform(df_2)
# print(X_train)
# print(X_test)
df.to_csv(r'C:\Users\Gang\Desktop\new_data\a_data1.csv')
# df.plot()
# df_2.plot()
# plt.show()




