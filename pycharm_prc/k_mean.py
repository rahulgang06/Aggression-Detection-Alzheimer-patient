import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
# from matplotlib import style
from sklearn.cluster import KMeans

#
# def eucliedian(a,b) :
#     return np.linalg.norm(a-b)
#
data_1 = pd.read_csv("data5.csv")
data_2 = pd.read_csv("a_data3.csv")
data =[data_1, data_2]
data1 = pd.concat(data)
x = data1.iloc[:,[1,5]].values

kmeans = KMeans(n_clusters=2,random_state=0).fit(x)
print(x)
x_0 = x[kmeans.labels_ == 0]
x_1 = x[kmeans.labels_ == 1]
plt.plot(x_0[:,0], x_0[:,1], "bo")
plt.plot(x_1[:,0], x_1[:,1], "ro")
plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], "g*")
plt.xlabel("acc")
plt.ylabel("gyro")
plt.show()

# summary = data1.describe()
# summary = summary.transpose()
# df = summary.head()
# x = data1.iloc[:,[1,5]].values
# plt.rcParams['figure.figsize'] = (6,6)
# style.use('ggplot')
# plt.scatter(x[:,1],x[:,5],c='black',s=7)
#
#
# def main():
#     centroid = x[np.random.choice(x.shape[0],2,replace=False)]
#     print("Centroid are:",centroid)
#
#     total = x.shape
#     distance_1 = np.zeros(total[0])
#     distance_2 = np.zeros(total[0])
#     belongs_to = np.zeros(total[0])
#     c_old = np.zeros(centroid.shape)
#     error = eucliedian(centroid,c_old)
#     mean = np.zeros(centroid.shape)
#     iterator=0
#
#     while error!=0:
#         print("Iteration",iterator+1,":")
#         for i in range(total[0]):
#             distance_1 = eucliedian(x[i],centroid[0])
#             distance_2 = eucliedian(x[i],centroid[1])
#             if(distance_1[i]<distance_2[i]):
#                 belongs_to[i] = 0
#             if(distance_1[i]>distance_2[i]):
#                 belongs_to[i] = 1
#     c_old = deepcopy(centroid)
#     for i in range(len(belongs_to)):
#         if belongs_to[i]==0:
#             mean[0][0]=np.mean(x[i][0])
#             mean[0][1]=np.mean(x[i][1])
#         else:
#             continue
#
#     print("New centroid for cluster 1:",mean[0])
#
#     for i in range(len(belongs_to)):
#         if belongs_to[i]==0:
#             mean[0][0]=np.mean(x[i][0])
#             mean[0][1]=np.mean(x[i][1])
#         else:
#             continue
#     print("New centroid for cluster 1:",mean[1])
#
#     centroid[0]=mean[0]
#     centroid[1]=mean[1]
#     error=eucliedian(centroid,c_old)
#     iterator+=1
#     if error==0 :
#         print("Same centroids again")
#
#
#     plt.rcParams['figure.figsize'] = (6,6)
#     plt.style.use('ggplot')
#     colors =['r','g']
#     fig,ax = plt.subplots()
#     for i in range(2) :
#         points =np.array([x[j] for j in range(len(x)) if belongs_to[j]==i])
#         ax.scatter(points[:,1],points[:,5],c=colors[i],s=7)
#     ax.scatter(centroid[:,0],centroid[:,1],marker='*',s=200,c='#050505')
#
#     plt.show()
#     print(data1.corr())
#
# if __name__ == '__main__':
#     main()
#
#
#
#
#
#
#
#
#
#
#
#
#
# print(data1[["|__acc","|__gyro"]])











