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
import arff
from sklearn.decomposition import PCA
from info_gain import info_gain
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel


dataset = arff.load(open('train_mnist_clean_bestdata.txt'))
data = pd.DataFrame(np.array(dataset['data']))
print("Dataset Length:: ", len(data))
print("Dataset Shape:: ",data.shape)


X = data.loc[:,0:523]
Y = data.loc[:,524]

##for i in X:
##    if (info_gain.info_gain(i, Y) ==0):
        

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

print("given data")
tuned_parameters=[{'C':[10**4,10**-2,10**0,10**2,10**4]}]
model=GridSearchCV(SVC(kernel='linear',gamma='auto'),tuned_parameters,scoring='f1_micro',cv=5)
model.fit(X_train, y_train)
print(model.best_estimator_)
print(model.score(X_test,y_test))
##


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
sfm = SelectFromModel(clf_entropy, threshold=0.01)
sfm.fit(X_train, y_train)


print("information gain")
tuned_parameters=[{'C':[10**4,10**-2,10**0,10**2,10**4]}]
model=GridSearchCV(SVC(kernel='linear',gamma='auto'),tuned_parameters,scoring='f1_micro',cv=5)
model.fit(X_train, y_train)
print(model.best_estimator_)
print(model.score(X_test,y_test))


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)
print("PCA")
pca = PCA(.90)
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)



tuned_parameters=[{'C':[10**4,10**-2,10**0,10**2,10**4]}]
model=GridSearchCV(SVC(kernel='linear',gamma='auto'),tuned_parameters,scoring='f1_micro',cv=5)
model.fit(X_train, y_train)
print(model.best_estimator_)
print(model.score(X_test,y_test))
