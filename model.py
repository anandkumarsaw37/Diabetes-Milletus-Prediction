import os
import pickle
os.getcwd()
os.chdir("C:\\Users\\Ani\\Desktop\\Project")
os.getcwd()
import numpy as np
import pandas as pd
from sklearn import preprocessing
dataset=pd.read_csv('pima-indians-diabetes.csv')
dataset.head()
X=dataset.loc[:, ['preg', 'plas', 'pres','skin','test','mass','pedi','age']]
Y=dataset.loc[:, ['class']]
X.head()
Y.head()
X=dataset.iloc[:, 0:8]
Y=dataset.iloc[:, -1]
X.head()
Y.head()
from sklearn import model_selection, neighbors
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=37)
X_train.head()
X_test.head()
Y_train.head()
Y_test.head()
from sklearn.model_selection import cross_val_score, cross_val_predict
minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(X).transform(X)
X_train_std=minmax.fit_transform(X_train)
X_test_std=minmax.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rmf=RandomForestClassifier(max_depth=3, random_state=1)
rmf_clf=rmf.fit(X_train, Y_train)
pickle.dump(rmf, open('model.pkl', 'wb'))
model=pickle.load(open('model.pkl', 'rb'))
print('Class is:', model.predict([[6,148,72,35,0,33.6,0.627,50]]))
rmf_clf_acc=cross_val_score(rmf_clf, X_train_std, Y_train, cv=3, scoring="accuracy", n_jobs=-1)
rmf_proba=cross_val_predict(rmf_clf, X_train_std, Y_train, cv=3, method="predict_proba")
rmf_clf_scores=rmf_proba[:, 1]
#predict on test data
Y_pred=rmf.predict(X_test)
Y_pred
print("Actual diabetes milletus:")
print(Y_test.values)
#Accuracy score on Test and Train
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
print("\n Accuracy Score: %f" %(accuracy_score(Y_test, Y_pred)*100))
print("\n Recall Score: %f" %(recall_score(Y_test, Y_pred)*100))
print("\n ROC Score: %f" %(roc_auc_score(Y_test, Y_pred)*100))
print(confusion_matrix(Y_test, Y_pred))
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
def ROC_curve(title, Y_train, scores, label=None):
    #calculate the roc score
    fpr, tpr, thresholds = roc_curve(Y_train, scores)
    print('AUC Score({}): {:.2f} '.format(title, roc_auc_score(Y_train, scores)))
    
    #plot the ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=label, color='b')
    plt.xlabel("False positive rate", fontsize=16)
    plt.ylabel("False negative rate", fontsize=16)
    plt.title('ROC curve {}:'.format(title), fontsize=16)
    plt.show()
ROC_curve('Random Forest Classifier', Y_train, rmf_clf_scores)
