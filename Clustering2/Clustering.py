from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
import joblib

from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('dataset2.csv', sep=';',  quoting=csv.QUOTE_NONE, encoding='utf-8')
#data = data.drop('ID', axis=1)

import warnings

warnings.filterwarnings("ignore")

# Seu código aqui


print(data.groupby('label').size())

X = data.loc[:,data.columns[:-1]]
y = data.label
#X = X.values
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)


#####
#See if all positions are given

pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression()),
    ])

#pipeline.fit(X_train, y_train)
'''
print(pipeline.score(X_test, y_test))


####
#Semi-Supervised Classification


n_labeled = 100

pipeline.fit(X_train[:n_labeled], y_train[:n_labeled])
print(pipeline.score(X_test, y_test))
'''
###############3
#Semi-Supervised Clustering


X_train, X_test, y_train, y_test = train_test_split(X, y)


k=100
kmeans = KMeans(n_clusters=k)
X_dist = kmeans.fit_transform(X_train) 
representative_idx = np.argmin(X_dist, axis=0) 
X_representative = X_train.values[representative_idx]

y_representative = [list(y_train)[x] for x in representative_idx]
pipeline.fit(X_representative, y_representative)


#print(pipeline.score(X_test, y_test))
print(pipeline.predict(X_test))

#pickle.dump(pipeline, open("kmean_clustering.pkl", "wb"))
joblib.dump(pipeline, "./kmeans.joblib")

