import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix,f1_score
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


input_data = pd.read_csv('traindrugs.txt',sep= '	', header= None)
vectorizer=CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=120000,token_pattern=r"(?u)\b\w+\b")
corpus=[]

for x in range(1,100001,1):
    corpus.append(str(x))

X=vectorizer.fit(corpus)
Y=(vectorizer.vocabulary_)

output=vectorizer.transform(input_data.loc[:,1])
df=pd.DataFrame(output.todense())

#test data
# Read test data
test_data = pd.read_csv('testdrugs.txt', sep=',', header=None)

# test_data.tail(1)
test_data.columns

test_output = vectorizer.transform(test_data.loc[:,0])

test_features_df = pd.DataFrame(test_output.todense(), columns=vectorizer.get_feature_names())

test_features_df.head()
print(test_features_df.shape)













scaler = StandardScaler()

scaler.fit(df)

df = scaler.transform(df)


print("Dataset Length:: ", len(df))
print("Dataset Shape:: ", df.shape)

target_df = input_data.loc[:, 0]
##X = df.loc[:,0:99998]
##Y = df.loc[:,99999]

##print(balance_data)


X_train, X_test, y_train, y_test = train_test_split( df, target_df, test_size = 0.2, random_state = 100)

print("PCA")
pca = PCA(.90)
pca.fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)

print("SMOTE")
#ros = RandomOverSampler(random_state=0)
#X,Y = ros.fit_resample(df, target_df)
smote_enn = SMOTEENN()
X_train,y_train = smote_enn.fit_resample(X_train,y_train)
#X_train,test_features_df = smote_enn.fit_resample(X_train,test_features_df)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
print("yooo",y_train.value_counts())


svclassifier = SVC(kernel='linear',gamma='auto')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))
print("f1score:",f1_score(y_test, y_pred))


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=4,max_features=2,min_samples_leaf=4)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print("decision tree pruned")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))
print("f1score:",f1_score(y_test, y_pred))


clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("forest")
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
print("Report : ",classification_report(y_test, y_pred))
print("f1score:",f1_score(y_pred, y_test))
##testing
bonda=clf.predict(test_features_df)
print(bonda)
