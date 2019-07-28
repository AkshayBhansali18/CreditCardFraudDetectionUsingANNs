import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn

dataset=pd.read_csv('C:\\Users\\aksha\\Documents\\ML documents\\creditcard.csv')
dataset=dataset.sample(frac=0.1,random_state=1)
print(dataset.shape)
fraud=dataset[dataset['Class']==1]
valid=dataset[dataset['Class']==0]
outlier_fraction=len(fraud)/float(len(valid))
print("Fraud cases {}".format(len(fraud)))
print("Valid cases {}".format(len(valid)))
cormat=dataset.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=0.8,square=True)
plt.show()

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(x.shape)
print(y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

classifiers={"IsolationForest":IsolationForest(max_samples=len(x),contamination=outlier_fraction,random_state=1),
             "LocalOutlierFraction":LocalOutlierFactor(contamination=outlier_fraction)}
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if(clf_name=="LocalOutlierFraction"):
        y_pred=clf.fit_predict(x)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        from sklearn.metrics import confusion_matrix

        cm1 = confusion_matrix(y, y_pred)
    else:
        clf.fit(x)
        score_pred=clf.decision_function(x)
        y_pred=clf.predict(x)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        from sklearn.metrics import confusion_matrix

        cm2= confusion_matrix(y, y_pred)

