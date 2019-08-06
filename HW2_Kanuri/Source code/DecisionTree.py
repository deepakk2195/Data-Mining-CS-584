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
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO

balance_data = pd.read_csv('breast-cancerdata.txt',sep= ',', header= None)

print("Dataset Length:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)

enc = OrdinalEncoder()
balance_data=enc.fit_transform(balance_data)


X = balance_data[:, 0:8]
Y = balance_data[:,9]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=1)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("decision stump")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=4)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
##dot_data = StringIO()
##export_graphviz(clf_gini, out_file=dot_data,  
##                filled=True, rounded=True,
##                special_characters=True)
##graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
##Image(graph.create_png())

print("decision tree unpruned")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=4,max_features=2,min_samples_leaf=4)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("decision tree pruned")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))



classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("KNN")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))
