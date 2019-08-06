import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

balance_data = pd.read_csv('telcodata.txt',sep= ',', header= None)

print("Dataset Length:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)

X = balance_data.iloc[:,0:36]
Y = balance_data.iloc[:,37]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=1)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("decision stump")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))


print("oversampling")
ros = RandomOverSampler(random_state=0)
X,Y = ros.fit_resample(X, Y)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=4,max_features=2,min_samples_leaf=4)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("decision stump")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))



print("undersampling")
rus = RandomUnderSampler(random_state=0)
X,Y = rus.fit_resample(X, Y)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=4,max_features=2,min_samples_leaf=4)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("decision stump")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))



print("SMOTE")
smote_enn = SMOTEENN(random_state=0)
X,Y = smote_enn.fit_resample(X, Y)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=4,max_features=2,min_samples_leaf=4)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("decision stump")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))
