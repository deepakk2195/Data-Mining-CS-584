# %%

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# %%
training_data = pd.read_csv('traindrugs.txt', sep='	', header=None)
vectorizer = CountVectorizer(
    tokenizer=None,
    preprocessor=None,
    stop_words=None,
    max_features=120000,
    token_pattern=r"(?u)\b\w+\b")

corpus = []

for x in range(1, 100001, 1):
    corpus.append(str(x))

X = vectorizer.fit([' '.join(corpus)])

output = vectorizer.transform(training_data.loc[:, 1])

features_df = pd.DataFrame(output.todense(), columns=vectorizer.get_feature_names())
target_df = training_data.loc[:, 0]

target_df.value_counts()

features_df.shape
# features_df.loc[0, '62214']


# %%
# Read test data
test_data = pd.read_csv('testdrugs.txt', sep=',', header=None)

# test_data.tail(1)
test_data.columns

test_output = vectorizer.transform(test_data.loc[:, 0])

test_features_df = pd.DataFrame(test_output.todense(), columns=vectorizer.get_feature_names())

# test_features_df.head()
# test_features_df.shape
# %%

# Combining features in train and test dataframes
combined_features_df = pd.concat([features_df, test_features_df], keys=['x', 'y'])

features_df.shape
test_features_df.shape

combined_features_df.shape
combined_features_df.loc['y']
# %%
# 3.b PCA
# PCA for dimensionality reduction

scaler = StandardScaler()

scaler.fit(combined_features_df)

# scaled_combined_data = scaler.transform(features_df)
scaled_combined_data = scaler.transform(combined_features_df)

pca = PCA(.90)
pca.fit(scaled_combined_data)

pca.n_components_
pca_combined_df = pd.DataFrame(pca.transform(scaled_combined_data))

pca_combined_df.shape
# Out[18]: (800, 645)

pca_test_df = pca_combined_df.iloc[800:]
pca_train_df = pca_combined_df.iloc[:800]

pca_test_df.shape
pca_train_df.shape

X_train, X_test, y_train, y_test = train_test_split(pca_train_df, target_df, train_size=.8)

pca_train_df.shape
X_train.shape
X_test.shape
type(X_train)
type(y_train)
# %%



from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_train, y_train = ros.fit_sample(X_train, y_train)

X_train.shape
y_train.shape
pd.DataFrame(y_train)[0].value_counts()


# %%

pd.DataFrame(X_train).columns
X_test.shape

tuning_params = [{'n_estimators': [10, 50, 100, 200],
                  'class_weight': ['balanced_subsample', 'balanced']}]
rand_forest = RandomForestClassifier(random_state=99)

rand_forest = GridSearchCV(rand_forest, tuning_params,
                           scoring='f1', cv=5, n_jobs=3)

rand_forest.fit(X_train, y_train)

f1_score(rand_forest.predict(X_test), y_test)

rand_forest_output = rand_forest.predict(pca_test_df)

rand_for_output_df = pd.Series(rand_forest_output)

rand_for_output_df.value_counts()

print(rand_forest.best_estimator_)
print(rand_forest.score(X_test, y_test))

f1_score(rand_forest.predict(X_test), y_test)

# %%

# X_train.shape
# import numpy as np
#
# np.bincount(y_train)

base_model = xgb.XGBClassifier(random_state=99)

tuning_params = [{'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]
                  # ,'scale_pos_weight': [3, 5, 7, 9, 11, 13, 15]
                  }]

xgb_model = GridSearchCV(base_model,
                         tuning_params,
                         scoring='f1', cv=5, n_jobs=2)
#%%
xgb_model.fit(X_train, y_train)

print(xgb_model.best_estimator_)
print(xgb_model.score(X_test, y_test))

#
# %%

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(loss='exponential', n_estimators=600, learning_rate=0.5,
                                 max_depth=5, random_state=0).fit(X_train, y_train)
f1_score(clf.predict(X_test), y_test)

X_test.shape
gbdt_test_output = clf.predict(pca_test_df)


gbdt_test_output_df = pd.Series(gbdt_test_output)


gbdt_test_output_df.to_csv('/team-submission-GBDT.csv', index=False)
gbdt_test_output_df.value_counts()
# %%




