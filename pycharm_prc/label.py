import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
# Labels are the values we want to predict
df = pd.read_csv("new_data.csv")
df.hist()
plt.show()
# df_1 = df.iloc[:, 1:5].values# Remove the labels from the features
# # axis 1 refers to the columns
# df_2 = df.iloc[:, ].values# Remove the labels from the features

# Saving feature names for later use
# feature_list = list(df.columns)
# Convert to numpy array
# features = df.values
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 20)
#
# rf = RandomForestRegressor(n_estimators = 100, random_state = 20)
# rf.fit(train_features, train_labels)
# predictions = rf.predict(test_features)
# errors = abs(predictions - test_labels)
# mape = 100 * (errors / test_labels)
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')



# forecast_col =['|__acc']
# df.fillna(df.mean,inplace='True')
# forecast_out = int(math.ceil(0.1*len(df)))
# print(forecast_out)
# df['label'] = df[forecast_col].shift(-forecast_out)
X = df.iloc[:, 8:9].values
y = df.iloc[:, 12].values
# print(y)
# df['label']= df['label'].fillna(value=0)
# print(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# cut_off = 500
# y_pred_classes = np.zeros_like(y_pred)
# y_pred_classes[y_pred > cut_off] = 1
#
# y_test_classes = np.zeros_like(y_pred)
# y_test_classes[y_test > cut_off] = 1

# print(y_test)
#
# print(y_train)

# print(X_train)
# print(X_test)
# # print(y_train)
# # print(y_test)
#
print('confusion matrix is')
print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
print('accuracy is')
print(accuracy_score(y_test, y_pred))



