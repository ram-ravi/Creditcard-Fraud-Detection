# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:06:12 2020

@author: ramravi
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

#importing the dataset
df= pd.read_csv("creditcard.csv")


#get some info
df.describe()
df.info() #no missing values


#analyse the fraud and non-fraud cases
print('no.of non-fraud are ', round(df['Class'].value_counts()[0]/len(df)*100,2),'%percent of dataset')
print('no of fraud are, ', round(df['Class'].value_counts()[1]/len(df)*100,2),'%percent of dataset')
#we can se that there is an imbalance in the dataset

#lets plot to know more:
fig,ax= plt.subplots(1,1,figsize=(20,4)) 
sns.countplot(x=df['Class'],data=df, ax=ax)
ax.set_title('distribution of Time', fontsize=14)
ax.set_xlim([min(df['Time'], max(df['Time']))])
plt.title('count of fraud and non-fraud')
plt.show()

#this is indeed the proof for imbalance in the dataset


#lets see how time and amount are distributed
fig,ax=plt.subplots(1,2, figsize=(18,4))
sns.distplot(df['Amount'], ax=ax[0],color='r')
ax[0].set_title('Distribution of Amount', fontsize=14)
ax[0].set_xlim([min(df['Amount']), max(df['Amount'])])

sns.distplot(df['Time'],ax=ax[1], color='b')
plt.show()
#we can see thedistribution of Amount is highly positively skewed.and the dsitribution of Time is mutimodel with different peaks

#lets see the correlation between amount and class

plt.scatter(df['Amount'],df['Class'])
#it is seen that there is more chances of fraud transaction and lesser amount

#lets scale the amount and time values
std_scaler=StandardScaler()
robust_scaler=RobustScaler()
df['scaled_amount']= robust_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time']= robust_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

scaled_time=df['scaled_time'].values
scaled_amount=df['scaled_amount'].values
df.drop(['scaled_time','scaled_amount'], axis=1, inplace=True)

#bringing it to the beginning
df.insert(0,'scaled_time',scaled_time)
df.insert(1,'scaled_amount', scaled_amount)



fig,ax=plt.subplots(1,2,figsize=(18,4))
sns.distplot(df['scaled_amount'], ax=ax[0],color='r')
ax[0].set_title('distribution of  scaled Amount')

sns.distplot(df['scaled_time'],ax=ax[1], color='b')
ax[1].set_title('Distribution of scaled time')
plt.show()
#as we can see the values have been standardized between ranges.

#lets create a equal number of sample with fraud and non-fraud cases:
X=df.drop('Class', axis=1)
y=df['Class']

sss=StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

for train_index, test_index in sss.split(X,y):
    print('train:', train_index, 'test:', test_index)
    original_X_train,original_X_test= X.iloc[train_index],X.iloc[test_index]
    original_y_train, original_y_test=y.iloc[train_index], y.iloc[test_index]
    
    
orginal_X_train=original_X_train.values   
original_X_test=original_X_test.values
original_y_train=original_y_train.values
original_y_test=original_y_train.values

#implementing random under sampling to have a balanced dataset

df=df.sample(frac=1)

fraud_df=df.loc[df['Class']==1]
non_fraud_df=df.loc[df['Class']==0][:492]

normal_df=pd.concat([fraud_df, non_fraud_df])
new_df= normal_df.sample(frac=1, random_state=42)
new_df.head()
colors=[]
print(new_df['Class'].value_counts()/len(new_df))
sns.countplot('Class',data=new_df)
plt.title('Equally Distributed Class', fontsize=10)
plt.show()


#correlation matrix
fig,(ax1, ax2)=plt.subplots(2,1,figsize=(24,20))
corr=df.corr()
sns.heatmap(corr,cmap='coolwarm_r',annot_kws={'size':20}, ax=ax1)
ax1.set_title('Imbalanced dataset correlations', fontsize=14)

sample_corr=new_df.corr()
sns.heatmap(sample_corr, cmap='coolwarm_r', annot_kws={'size':20},ax=ax2)
ax2.set_title('Correlation of balanced data', fontsize=14)
plt.show()
#we can see that in the imbalanced dataset we can merely see any correlation but in
# the second corr, we can see a couple of correlation

#lets see the negative and positive correlation using a box plot
#negative corrleation
fig, axes= plt.subplots(ncols=4, figsize=(20,4))
sns.boxplot(x='Class', y='V12', data=new_df, ax=axes[0])
axes[0].set_title('Relation between V12 and class')

sns.boxplot(x='Class', y='V17', data=new_df, ax=axes[1])
axes[1].set_title('Relation between V17 and class')

sns.boxplot(x='Class', y='V14', data=new_df,ax=axes[2])
axes[2].set_title('Relation between V14 and class')

sns.boxplot(x='Class', y='V10', data=new_df, ax=axes[3])
axes[3].set_title('Relation between V10 and class')

plt.show()

#positve correlations:
fig, axes= plt.subplots(ncols=4, figsize=(20,4))
sns.boxplot(x='Class', y='V11', data=new_df, ax=axes[0])
axes[0].set_title('Relation between V11 and class')

sns.boxplot(x='Class', y='V4', data=new_df, ax=axes[1])
axes[1].set_title('Relation between V4 and class')

sns.boxplot(x='Class', y='V2', data=new_df,ax=axes[2])
axes[2].set_title('Relation between V2 and class')

sns.boxplot(x='Class', y='V19', data=new_df,ax=axes[3])
axes[3].set_title('Relation between V19 and class')

plt.show()
#if can see the box plots, the lesser the negative correlation, the more likely is the fraud 
#transaction
from scipy.stats import norm

f, (ax1, ax2, ax3)= plt.subplots(1,3,figsize=(20,4))
v14_fraud_case= new_df['V14'].loc[new_df['Class']==1].values
sns.distplot(v14_fraud_case,fit=norm, ax=ax1, color='#FB8861')
ax1.set_title('Relationship  between V14 and  fraud transaction')

v12_fraud_case= new_df['V12'].loc[new_df['Class']==1].values
sns.distplot(v12_fraud_case, fit=norm,ax=ax2, color='#56F9BB')
ax2.set_title('Relationship  between V12 and fraud transaction')

v10_fraud_case= new_df['V10'].loc[new_df['Class']==1].values
sns.distplot(v10_fraud_case, fit=norm,ax=ax3, color='#C5B3F9')
ax3.set_title('Relationship  between V10 and fraud transaction')


#Removing Outliers (Highest Negative Correlated with labels)
v14_fraud=new_df['V14'].loc[new_df['Class']==1].values
q25,q75=np.percentile(v14_fraud,25), np.percentile(v14_fraud, 75)
print('Quarentile 25: {} | Quarentile 75: {}' .format(q25,q75))
v14_iqr=q75-q25
print('iqr: {}' .format(v14_iqr))

v14_cut_off=v14_iqr*1.5
v14_lower,v14_upper=q25-v14_cut_off, q75 + v14_cut_off
print('cut_off:{}' .format(v14_cut_off))
print('lower threshold: {}' .format(v14_lower))
print('upper threshold: {}' .format(v14_upper))

outliers= [x for x in v14_fraud if x < v14_lower or x> v14_upper]
print('the length of outliers {}' .format(len(outliers)))
print('the outliers are  {}' .format(outliers))

new_df= new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

sns.boxplot(x=new_df['Class'], y=new_df['V14'])
plt.show()


#remove for V10
v10_fraud= new_df['V10'].loc[new_df['Class']==1].values
q25, q75=np.percentile(v10_fraud,25), np.percentile(v10_fraud,75)
print('Qurentile 25: {} | Quarentile 75.{}' .format(q25,q75))
v10_iqr= q75-q25
print('iqr: {}' .format(v10_iqr))

v10_cut_off= v10_iqr*1.5
v10_lower,v10_upper=q25-v10_cut_off, q75 + v10_cut_off

print('cut_off:{}' .format(v10_cut_off))
print('lower threshold: {}' .format(v10_lower))
print('upper threshold: {}' .format(v10_upper))

outliers= [x for x in v10_fraud if x < v10_lower or x> v10_upper]
print('the length of outliers {}' .format(len(outliers)))
print('the outliers are  {}' .format(outliers))
 
new_df= new_df.drop(new_df[(new_df['V10'] > v10_upper )| (new_df['V10'] < v10_lower)].index)

sns.boxplot(x=new_df['Class'], y=new_df['V10'])
plt.show()

#remove v12 outliers
v12_fraud= new_df['V12'].loc[new_df['Class']==1].values
q25, q75=np.percentile(v12_fraud,25), np.percentile(v12_fraud,75)
print('Qurentile 25: {} | Quarentile 75.{}' .format(q25,q75))
v12_iqr= q75-q25
print('iqr: {}' .format(v12_iqr))

v12_cut_off= v12_iqr*1.5
v12_lower,v12_upper=q25-v12_cut_off, q75 + v12_cut_off

print('cut_off:{}' .format(v12_cut_off))
print('lower threshold: {}' .format(v12_lower))
print('upper threshold: {}' .format(v12_upper))

outliers= [x for x in v12_fraud if x < v12_lower or x> v12_upper]
print('the length of outliers {}' .format(len(outliers)))
print('the outliers are  {}' .format(outliers))
 
new_df= new_df.drop(new_df[(new_df['V12'] > v12_upper )| (new_df['V12'] < v12_lower)].index)


#plotting with removing outliers
f, (ax1, ax2, ax3)=plt.subplots(1,3, figsize=(20,4))

sns.boxplot(x=new_df['Class'], y=new_df['V12'], ax=ax1)
ax1.set_title('Removal outliers for V12 and class')

sns.boxplot(x=new_df['Class'], y=new_df['V14'], ax=ax2)
ax2.set_title('Removal outliers for V14 and class')

sns.boxplot(x=new_df['Class'], y=new_df['V10'], ax=ax3)
ax3.set_title('Removal outliers for V10 and class')
plt.show()

#lets implement dimentionality reduction:
#t-sne:
X_new_df=new_df.drop('Class', axis=1)
y_new_df=new_df['Class']
#t-sne
t0=time.time()
tsne_reduced=TSNE(n_components=2, random_state=42).fit_transform(X_new_df.values)
t1=time.time()
print('tsne took {:.2} s' .format(t1-t0))

#PCA
to=time.time()
PCA_reduced=PCA(n_components=2, random_state=12).fit_transform(X_new_df.values)
t1=time.time()
print('PCA took :{:.2} s' .format((t1-t0)))

#TruncatedSVD
t0=time.time()
tsvd_reduced=TruncatedSVD(n_components=2, random_state=11).fit_transform(X_new_df.values)
t1=time.time()
print('truncSVD took {:.2} s' .format((t1-t0)))

#lets visualize dimentionality reduction
#tsne
fig,(ax1,ax2,ax3)= plt.subplots(1,3,figsize=(24,4))
fig.suptitle('Dimentionality reduction using different techs', fontsize=14)

blue_patch=mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch= mpatches.Patch(color='#AF0000', label='Fraud')

ax1.scatter(x=tsne_reduced[:,0],y=tsne_reduced[:,1], c=(y_new_df==0),cmap='coolwarm', 
            label='No Fraud', linewidths=2)
ax1.scatter(x=tsne_reduced[:,0],y=tsne_reduced[:,1], c=(y_new_df==1), cmap='coolwarm',
            label='Fraud', linewidths=2)
ax1.set_title('TSNE DR', fontsize=14)
ax1.grid(True)
ax1.legend(handles=[blue_patch, red_patch])

#PCA

ax2.scatter(x=PCA_reduced[:,0],y=PCA_reduced[:,1], c=(y_new_df==0),cmap='coolwarm', 
            label='No Fraud', linewidths=2)
ax2.scatter(x=PCA_reduced[:,0],y=PCA_reduced[:,1], c=(y_new_df==1), cmap='coolwarm',
            label='Fraud', linewidths=2)
ax2.set_title('PCA DR', fontsize=14)
ax2.grid(True)
ax2.legend(handles=[blue_patch, red_patch])

#TSVE
ax3.scatter(x=tsvd_reduced[:,0],y=tsvd_reduced[:,1], c=(y_new_df==0),cmap='coolwarm', 
            label='No Fraud', linewidths=2)
ax3.scatter(x=tsvd_reduced[:,0],y=tsvd_reduced[:,1], c=(y_new_df==1), cmap='coolwarm',
            label='Fraud', linewidths=2)
ax3.set_title('tsve DR', fontsize=14)
ax3.grid(True)
ax3.legend(handles=[blue_patch, red_patch])
plt.show()

#Classifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train,y_test= train_test_split(X_new_df, y_new_df, test_size=.2, 
                                                  random_state=3)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

classifiers={
    'logisticreg':LogisticRegression(),
    'knearest': KNeighborsClassifier(),
    'SVC': SVC(),
    'decisiontree': DecisionTreeClassifier()}

for key, classifier in classifiers.items():
    classifier.fit(X_train,y_train)
    training_score= cross_val_score(classifier, X_train, y_train, cv=5)
    print('Classifier:', classifier.__class__.__name__,'has a training score of ',
          round(training_score.mean(),2)*100, '% accuracy score')

#Grid search cv:
from sklearn.model_selection import GridSearchCV

log_reg_params={'penalty':['l1','l2'], 'C':[0.001, 0.01,0.1,1, 10,100, 1000]}

grid_log_reg=GridSearchCV(LogisticRegression(),log_reg_params)
grid_log_reg.fit(X_train, y_train)

log_reg_best_estim=grid_log_reg.best_estimator_

#knn
knears_params={'n_neighbors' : list(range(2,5,1)), 'algorithm':['auto','ball_tree',
                                                             'kd_tree', 'brute']}

grid_knn=GridSearchCV(KNeighborsClassifier(),knears_params)
grid_knn.fit(X_train, y_train)
knn_best_estim=grid_knn.best_estimator_

#svc()

svc_params={'C':[0.5, 0.7,0.9,1], 'kernel':['rbf', 'poly','sigmoid','linear']}
grid_svc=GridSearchCV(SVC(),svc_params)
grid_svc.fit(X_train,y_train)

svc_best_estim=grid_svc.best_estimator_

#decisiontree

dt_params={'criterion':['gini','entropy'], 'max_depth' : list(range(2,4,1)),
           'min_samples_leaf': list(range(5,7,1))}
dt_grid=GridSearchCV(DecisionTreeClassifier(),dt_params)
dt_grid.fit(X_train,y_train)
dt_best_estim= dt_grid.best_estimator_

#lets compute the cross_val_score

log_reg_score= cross_val_score(log_reg_best_estim,X_train, y_train,cv=5)
print('log reg with best estimator having:',
      round(log_reg_score.mean()*100,2).astype(str) + '%')

knn_score=cross_val_score(knn_best_estim, X_train,y_train, cv=5)
print('knn has a score of:', round(knn_score.mean()*100,2).astype(str) + '%')

svc_score=cross_val_score(svc_best_estim,X_train,y_train,cv=5)
print('svc has a score of: ' , round(svc_score.mean()*100,2).astype(str) + '%')

dt_score=cross_val_score(dt_best_estim,X_train,y_train,cv=5)
print('dt has a score of: ' , round(dt_score.mean()*100,2).astype(str) + '%')

#lets oversample and undersample the data to see the performance.
#we should be careful to oversample or undersample the data during cross validation
#not before so that we dont have any cross validation on duplicate data.

undersample_X= new_df.drop('Class', axis=1)
undersample_y= new_df['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print('train:' ,train_index, 'test: ', test_index)
    undersample_Xtrain, undersample_Xtest= undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest= undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
    
undersample_Xtrain= undersample_Xtrain.values
undersample_Xtest= undersample_Xtest.values
undersample_ytrain= undersample_ytrain.values
undersample_ytest= undersample_ytest.values

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

# for train,test in sss.split(undersample_Xtrain, undersample_ytrain):
#     imba_pipline=imbalanced_make_pipeline(SMOTE(sampling_strategy='majority'),
#                                           log_reg_best_estim)
#     undersample_model= imba_pipline.fit(undersample_Xtrain[train], undersample_ytrain[train])
#     undersample_prediction=undersample_model.predict(undersample_Xtrain[test])
#     undersample_accuracy.append(imba_pipline.score(undersample_Xtrain[test], undersample_ytrain[test]))
#     undersample_precision.append(precision_score(original_y_train[test],undersample_prediction))
#     undersample_recall.append(recall_score(original_y_train[test],undersample_prediction))
#     undersample_f1.append(f1_score(original_y_train[test], undersample_prediction))
#     undersample_auc.append(roc_auc_score(original_y_train[test], undersample_prediction))
    
                                
near_miss_X= SMOTE()
near_X, near_y=near_miss_X.fit_sample(undersample_Xtrain, undersample_ytrain.ravel())




















































































































































































































































































































































































































    






































