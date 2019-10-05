# -*- coding: utf-8 -*-

#IMPORT DATA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer


cancer=load_breast_cancer()
df_cancer=pd.DataFrame(np.c_[cancer["data"],cancer["target"]],columns=np.append(cancer["feature_names"],["target"]))

print(cancer["feature_names"])

#VISUALIZING DATA
sns.pairplot(df_cancer,hue="target",vars=['mean radius','mean texture', 'mean perimeter' ,'mean area','mean smoothness'])
sns.countplot(df_cancer["target"])
sns.scatterplot(x="mean area",y="mean smoothness",hue="target",data=df_cancer)
#FEATURE SCALLING 
from sklearn.preprocessing import minmax_scale
X_transformed=minmax_scale(X)


#TRAINGING

X=df_cancer.drop(["target"],axis=1)
y=df_cancer["target"]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_transformed,y,test_size=0.33)


from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,f1_score

clf=SVC(C= 10, gamma= 0.1, kernel='rbf')
clf.fit(X_train,y_train)

pred=clf.predict(X_test)
cm=confusion_matrix(y_test,pred)
sns.heatmap(cm,annot=True)
print(clf.score(X_test,y_test))
F1=f1_score(y_test,pred)



#impoving the model again
param_grid={"C":[0.1,1,10,100],"gamma":[1,0.1,0.01,0.001],"kernel":["rbf"]}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)

grid.fit(X_train,y_train)
print(clf.F)
print(grid.best_params_)
