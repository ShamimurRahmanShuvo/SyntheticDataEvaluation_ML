#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:23:44 2020

@author: shuvo
"""
import numpy as np
import pandas as pd
from numpy import sqrt, argmax
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score
import time
import csv

np.random.seed(42)

plt.figure(figsize=(25, 17))

#Load Real Data
data = pd.read_csv("./input/Pasyim_new.csv")
data1 = data.drop(['type', 'nameOrig', 'nameDest', 'isFlaggedFraud'],axis=1)
y_true = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(data1, y_true, test_size=0.3, random_state=42)

#Load Synthetic Data
dataS = pd.read_csv("./input/Paysim_new_synt.csv")
dataS1 = dataS.drop(['type', 'nameOrig', 'nameDest', 'isFlaggedFraud'],axis=1)
yS_true = dataS['isFraud']
XS_train, XS_test, yS_train, yS_test = train_test_split(dataS1, yS_true, test_size=0.3, random_state=42)

#undersampling original data
# rus = RandomUnderSampler()
# X_train, y_train = rus.fit_sample(X_train, y_train)
# X_test, y_test = rus.fit_sample(X_test, y_test)

#undersampling synthetic data
# XS_train, yS_train = rus.fit_sample(X_train, y_train)
# XS_test, yS_test = rus.fit_sample(X_test, y_test)

#oversampling original data
# ada = ADASYN()
# X_train, y_train = ada.fit_sample(X_train, y_train)
# X_test, y_test = ada.fit_sample(X_test, y_test)

#oversampling synthetic data
# XS_train, yS_train = ada.fit_sample(XS_train, yS_train)
# XS_test, yS_test = ada.fit_sample(XS_test, yS_test)

classifiers = {
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20, novelty = True)
}

# ******************************** With Original Data ********************************

print('*'*25 +'Training with Original Data, Testing With Original Data' + '*'*25)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        tic1=time.time()
        clf.fit(X_train)
        tic2 = time.time()
        elapsedtime=tic2-tic1
        print("")
        print("Training Time Taken : "+str(elapsedtime)+"seconds")
        y_pred = clf.predict(X_test)
        tic3 = time.time()
        elapsedtime=tic3-tic2
        print("")
        print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
    else:
        clf.fit(X_train)
        scores_prediction = clf.decision_function(X_train)
        y_pred = clf.predict(X_test)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    scn1_tn,scn1_fp,scn1_fn,scn1_tp=confusion_matrix(y_test,y_pred).ravel()
    
    total_records1 = scn1_tn + scn1_fp + scn1_fn + scn1_tp
    scn1_tn = (scn1_tn/total_records1) * 100
    scn1_fp = (scn1_fp/total_records1) * 100
    scn1_fn = (scn1_fn/total_records1) * 100
    scn1_tp = (scn1_tp/total_records1) * 100 

    scoring = clf.decision_function(X_train) 
    fpr, tpr, thresholds = roc_curve(y_train, scoring)
    j = sqrt(tpr * (1 - fpr))
    ix = argmax(j)   

    precision, recall = precision_recall_curve(y_train, scoring)[:2]

    AUC = auc(fpr, tpr)
    AUPR = auc(recall, precision)
    
    scn1_acc = accuracy_score(y_test, y_pred) * 100
    scn1_ap = average_precision_score(y_test, y_pred) * 100
    scn1_p = precision_score(y_test, y_pred) * 100
    scn1_r = recall_score(y_test, y_pred) * 100
    scn1_f = f1_score(y_test, y_pred) * 100
    scn1_roc = roc_auc_score(y_test, y_pred) * 100
    scn1_pr = AUPR * 100
    scn1_thresh = thresholds[ix]

    #plt.subplot(121)
    plt.plot(fpr, tpr, lw=1, label='%s (area = %0.3f)' % (clf_name, AUC))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('ROC curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()

    #plt.subplot(122)
    plt.plot(recall, precision, lw=1, label='%s (area = %0.3f)'
                  % (clf_name, AUPR))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.title('PR curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()

# ******************************** With Synthetic Data ********************************

print('*'*25 +'Training with Synthetic Data, Testing With Synthetic Data' + '*'*25)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        tic1=time.time()
        clf.fit(XS_train)
        tic2 = time.time()
        elapsedtime=tic2-tic1
        print("")
        print("Training Time Taken : "+str(elapsedtime)+"seconds")
        y_predS = clf.predict(XS_test)
        tic3 = time.time()
        elapsedtime=tic3-tic2
        print("")
        print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
    else:
        clf.fit(XS_train)
        scores_prediction = clf.decision_function(XS_train)
        y_predS = clf.predict(XS_test)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_predS[y_predS == 1] = 0
    y_predS[y_predS == -1] = 1
    
    scn2_tn,scn2_fp,scn2_fn,scn2_tp=confusion_matrix(yS_test,y_predS).ravel()
    
    total_records2 = scn2_tn + scn2_fp + scn2_fn + scn2_tp
    scn2_tn = (scn2_tn/total_records2) * 100
    scn2_fp = (scn2_fp/total_records2) * 100
    scn2_fn = (scn2_fn/total_records2) * 100
    scn2_tp = (scn2_tp/total_records2) * 100 

    scoringS = clf.decision_function(XS_train) 
    fpr, tpr, thresholds = roc_curve(yS_train, scoringS)
    j = sqrt(tpr * (1 - fpr))
    ix = argmax(j)   

    precision, recall = precision_recall_curve(yS_train, scoringS)[:2]

    AUC = auc(fpr, tpr)
    AUPR = auc(recall, precision)
    
    scn2_acc = accuracy_score(yS_test, y_predS) * 100
    scn2_ap = average_precision_score(yS_test, y_predS) * 100
    scn2_p = precision_score(yS_test, y_predS) * 100
    scn2_r = recall_score(yS_test, y_predS) * 100
    scn2_f = f1_score(yS_test, y_predS) * 100
    scn2_roc = roc_auc_score(yS_test, y_predS) * 100
    scn2_pr = AUPR * 100
    scn2_thresh = thresholds[ix]

    #plt.subplot(121)
    plt.plot(fpr, tpr, lw=1, label='%s (area = %0.3f)' % (clf_name, AUC))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('ROC curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()

    #plt.subplot(122)
    plt.plot(recall, precision, lw=1, label='%s (area = %0.3f)'
                  % (clf_name, AUPR))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.title('PR curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()

# ******************************** Testing Synthetic Data ********************************

print('*'*25 +'Training with Real Data, Testing With Synthetic Data' + '*'*25)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        tic1=time.time()
        clf.fit(X_train)
        tic2 = time.time()
        elapsedtime=tic2-tic1
        print("")
        print("Training Time Taken : "+str(elapsedtime)+"seconds")
        y_predS = clf.predict(XS_test)
        tic3 = time.time()
        elapsedtime=tic3-tic2
        print("")
        print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
    else:
        clf.fit(X_train)
        scores_prediction = clf.decision_function(XS_train)
        y_predS = clf.predict(XS_test)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_predS[y_predS == 1] = 0
    y_predS[y_predS == -1] = 1
    
    scn3_tn,scn3_fp,scn3_fn,scn3_tp=confusion_matrix(yS_test,y_predS).ravel()
    
    total_records3 = scn3_tn + scn3_fp + scn3_fn + scn3_tp
    scn3_tn = (scn3_tn/total_records3) * 100
    scn3_fp = (scn3_fp/total_records3) * 100
    scn3_fn = (scn3_fn/total_records3) * 100
    scn3_tp = (scn3_tp/total_records3) * 100 

    scoringS = clf.decision_function(XS_train) 
    fpr, tpr, thresholds = roc_curve(yS_train, scoringS)
    j = sqrt(tpr * (1 - fpr))
    ix = argmax(j)
        
    precision, recall = precision_recall_curve(yS_train, scoringS)[:2]

    AUC = auc(fpr, tpr)
    AUPR = auc(recall, precision)
    
    scn3_acc = accuracy_score(yS_test, y_predS) * 100
    scn3_ap = average_precision_score(yS_test, y_predS) * 100
    scn3_p = precision_score(yS_test, y_predS) * 100
    scn3_r = recall_score(yS_test, y_predS) * 100
    scn3_f = f1_score(yS_test, y_predS) * 100
    scn3_roc = roc_auc_score(yS_test, y_predS) * 100
    scn3_pr = AUPR * 100
    scn3_thresh = thresholds[ix]

    #plt.subplot(121)
    plt.plot(fpr, tpr, lw=1, label='%s (area = %0.3f)' % (clf_name, AUC))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('ROC curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()

    #plt.subplot(122)
    plt.plot(recall, precision, lw=1, label='%s (area = %0.3f)'
                  % (clf_name, AUPR))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.title('PR curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()

# ******************************** Testing Original Data ********************************

print('*'*25 +'Training with Synthetic Data, Testing With Real Data' + '*'*25)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        tic1=time.time()
        clf.fit(XS_train)
        tic2 = time.time()
        elapsedtime=tic2-tic1
        print("")
        print("Training Time Taken : "+str(elapsedtime)+"seconds")
        y_predS = clf.predict(X_test)
        tic3 = time.time()
        elapsedtime=tic3-tic2
        print("")
        print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
    else:
        clf.fit(XS_train)
        scores_prediction = clf.decision_function(X_train)
        y_predS = clf.predict(X_test)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_predS[y_predS == 1] = 0
    y_predS[y_predS == -1] = 1
    
    scn4_tn,scn4_fp,scn4_fn,scn4_tp=confusion_matrix(y_test,y_predS).ravel()
    
    total_records4 = scn4_tn + scn4_fp + scn4_fn + scn4_tp
    scn4_tn = (scn4_tn/total_records4) * 100
    scn4_fp = (scn4_fp/total_records4) * 100
    scn4_fn = (scn4_fn/total_records4) * 100
    scn4_tp = (scn4_tp/total_records4) * 100

    scoringS = clf.decision_function(X_train) 
    fpr, tpr, thresholds = roc_curve(y_train, scoringS)
    j = sqrt(tpr * (1 - fpr))
    ix = argmax(j)   

    precision, recall = precision_recall_curve(y_train, scoringS)[:2]

    AUC = auc(fpr, tpr)
    AUPR = auc(recall, precision)
    
    scn4_acc = accuracy_score(y_test, y_predS) * 100
    scn4_ap = average_precision_score(y_test, y_predS) * 100
    scn4_p = precision_score(y_test, y_predS) * 100
    scn4_r = recall_score(y_test, y_predS) * 100
    scn4_f = f1_score(y_test, y_predS) * 100
    scn4_roc = roc_auc_score(y_test, y_predS) * 100
    scn4_pr = AUPR * 100
    scn4_thresh = thresholds[ix]

    #plt.subplot(121)
    plt.plot(fpr, tpr, lw=1, label='%s (area = %0.3f)' % (clf_name, AUC))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('ROC curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()

    #plt.subplot(122)
    plt.plot(recall, precision, lw=1, label='%s (area = %0.3f)'
                 % (clf_name, AUPR))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.title('PR curve', fontsize=25)
    plt.legend(loc="lower right", prop={'size': 14})
    plt.show()
    
# ******************************** Normalized Difference ********************************

# ************ Normalized difference for TP,FN,TN,FP ************
# ******** In-In Vs Out-Out ********
diff1_tp = scn2_tp - scn1_tp
diff1_fn = scn2_fn - scn1_fn
diff1_tn = scn2_tn - scn1_tn
diff1_fp = scn2_fp - scn1_fp

# ******** In-In Vs In-Out ********
diff2_tp = scn3_tp - scn1_tp
diff2_fn = scn3_fn - scn1_fn
diff2_tn = scn3_tn - scn1_tn
diff2_fp = scn3_fp - scn1_fp

# ******** In-In Vs Out-In ********
diff3_tp = scn4_tp - scn1_tp
diff3_fn = scn4_fn - scn1_fn
diff3_tn = scn4_tn - scn1_tn
diff3_fp = scn4_fp - scn1_fp


# ************ Accuracy, Average Precision, Precision, Recall, F1-score, ROC, PR ************
# ******** In-In Vs Out-Out ********
diff1_acc = scn2_acc - scn1_acc
diff1_ap = scn2_ap - scn1_ap
diff1_p = scn2_p - scn1_p
diff1_r = scn2_r - scn1_r
diff1_f = scn2_f - scn1_f
diff1_roc = scn2_roc - scn1_roc
diff1_pr = scn2_pr - scn1_pr

# ******** In-In Vs In-Out ********
diff2_acc = scn3_acc - scn1_acc
diff2_ap = scn3_ap - scn1_ap
diff2_p = scn3_p - scn1_p
diff2_r = scn3_r - scn1_r
diff2_f = scn3_f - scn1_f
diff2_roc = scn3_roc - scn1_roc
diff2_pr = scn3_pr - scn1_pr

# ******** In-In Vs Out-In ********
diff3_acc = scn4_acc - scn1_acc
diff3_ap = scn4_ap - scn1_ap
diff3_p = scn4_p - scn1_p
diff3_r = scn4_r - scn1_r
diff3_f = scn4_f - scn1_f
diff3_roc = scn4_roc - scn1_roc
diff3_pr = scn4_pr - scn1_pr

# ******************************** Print The Results ********************************
     
h1 = "Paysim_synt_TP_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h2 = "Paysim_synt_TN_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h3 = "Paysim_synt_FP_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h4 = "Paysim_synt_FN_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h5 = "Paysim_synt_Accuracy_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h6 = "Paysim_synt_AveragePrecision_LocalOutlierFactor_n_neighbors = 20_novelty = True" 
h7 = "Paysim_synt_Precision_LocalOutlierFactor_n_neighbors = 20_novelty = True" 
h8 = "Paysim_synt_Recall_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h9 = "Paysim_synt_F1-Score_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h10 = "Paysim_synt_ROC-Area_LocalOutlierFactor_n_neighbors = 20_novelty = True"
h11 = "Paysim_synt_PR-Area_LocalOutlierFactor_n_neighbors = 20_novelty = True"

tp = [[h1,scn1_tp, scn2_tp, diff1_tp], [h1,scn1_tp, scn3_tp, diff2_tp], [h1,scn1_tp, scn4_tp, diff3_tp]]
tn = [[h2,scn1_tn, scn2_tn, diff1_tn], [h2,scn1_tn, scn3_tn, diff2_tn], [h2,scn1_tn, scn4_tn, diff3_tn]]
fp = [[h3,scn1_fp, scn2_fp, diff1_fp], [h3,scn1_fp, scn3_fp, diff2_fp], [h3,scn1_fp, scn4_fp, diff3_fp]]
fn = [[h4,scn1_fn, scn2_fn, diff1_fn], [h4,scn1_fn, scn3_fn, diff2_fn], [h4,scn1_fn, scn4_fn, diff3_fn]]
acc = [[h5,scn1_acc, scn2_acc, diff1_acc], [h5,scn1_acc, scn3_acc, diff2_acc], [h5,scn1_acc, scn4_acc, diff3_acc]]
ap = [[h6,scn1_ap, scn2_ap, diff1_ap], [h6,scn1_ap, scn3_ap, diff2_ap], [h6,scn1_ap, scn4_ap, diff3_ap]]
p = [[h7,scn1_p, scn2_p, diff1_p], [h7,scn1_p, scn3_p, diff2_p], [h7,scn1_p, scn4_p, diff3_p]]
r = [[h8,scn1_r, scn2_r, diff1_r], [h8,scn1_r, scn3_r, diff2_r], [h6,scn1_r, scn4_r, diff3_r]]
f = [[h9,scn1_f, scn2_f, diff1_f], [h9,scn1_f, scn3_f, diff2_f], [h9,scn1_f, scn4_f, diff3_f]]
roc = [[h10,scn1_roc, scn2_roc, diff1_roc], [h10,scn1_roc, scn3_roc, diff2_roc], [h10,scn1_roc, scn4_roc, diff3_roc]]
pr = [[h11,scn1_pr, scn2_pr, diff1_pr], [h11,scn1_pr, scn3_pr, diff2_pr], [h11,scn1_pr, scn4_pr, diff3_pr]]

cs = np.concatenate((tp,tn,fp,fn,acc,ap,p,r,f,roc,pr))
myFile = open('paysim_localOutlierFactor.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(cs)

print('*'*25 +' Scenario 1 Results ' + '*'*25) # training with original, testing with original
print("tn =%.3f" %scn1_tn,"fp =%.3f" %scn1_fp)
print("fn =%.3f" %scn1_fn,"tp =%.3f" %scn1_tp)
print("Accuracy Score: %.3f" %scn1_acc)
print("Average Precision Score: %.3f" %scn1_ap)
print("Precision Score: %.3f" %scn1_p)
print("Recall Score: %.3f" %scn1_r)
print("F1-Score: %.3f" %scn1_f)
print("Optimal Threshold: %.3f" %scn1_thresh)

print('*'*25 +' Scenario 2 Results ' + '*'*25) # training with synthetic, testing with synthic
print("tn =%.3f" %scn2_tn,"fp =%.3f" %scn2_fp)
print("fn =%.3f" %scn2_fn,"tp =%.3f" %scn2_tp)
print("Accuracy Score: %.3f" %scn2_acc)
print("Average Precision Score: %.3f" %scn2_ap)
print("Precision Score: %.3f" %scn2_p)
print("Recall Score: %.3f" %scn2_r)
print("F1-Score: %.3f" %scn2_f)
print("Optimal Threshold: %.3f" %scn2_thresh)

print('*'*25 +' Scenario 3 Results ' + '*'*25) # training with original, testing with synthetic
print("tn =%.3f" %scn3_tn,"fp =%.3f" %scn3_fp)
print("fn =%.3f" %scn3_fn,"tp =%.3f" %scn3_tp)
print("Accuracy Score: %.3f" %scn3_acc)
print("Average Precision Score: %.3f" %scn3_ap)
print("Precision Score: %.3f" %scn3_p)
print("Recall Score: %.3f" %scn3_r)
print("F1-Score: %.3f" %scn3_f)
print("Optimal Threshold: %.3f" %scn3_thresh)

print('*'*25 +' Scenario 4 Results ' + '*'*25) # training with synthetic, testing with original
print("tn =%.3f" %scn4_tn,"fp =%.3f" %scn4_fp)
print("fn =%.3f" %scn4_fn,"tp =%.3f" %scn4_tp)
print("Accuracy Score: %.3f" %scn4_acc)
print("Average Precision Score: %.3f" %scn4_ap)
print("Precision Score: %.3f" %scn4_p)
print("Recall Score: %.3f" %scn4_r)
print("F1-Score: %.3f" %scn4_f)
print("Optimal Threshold: %.3f" %scn4_thresh)

print('*'*25 +' Normalized difference for In-In Vs Out-Out ' + '*'*25) 
print("Difference TP: %.3f" %diff1_tp)
print("Difference FN: %.3f" %diff1_fn)
print("Difference TN: %.3f" %diff1_tn)
print("Difference FP: %.3f" %diff1_fp)
print("Difference Accuracy: %.3f" %diff1_acc)
print("Difference Average Precision: %.3f" %diff1_ap)
print("Difference Precision: %.3f" %diff1_p)
print("Difference Recall: %.3f" %diff1_r)
print("Difference F1-score: %.3f" %diff1_f)
print("Difference Area Under ROC Curve: %.3f" %diff1_roc)
print("Difference Area Under PR Curve: %.3f" %diff1_pr)


print('*'*25 +' Normalized difference for In-In Vs In-Out ' + '*'*25) 
print("Difference TP: %.3f" %diff2_tp)
print("Difference FN: %.3f" %diff2_fn)
print("Difference TN: %.3f" %diff2_tn)
print("Difference FP: %.3f" %diff2_fp)
print("Difference Accuracy: %.3f" %diff2_acc)
print("Difference Average Precision: %.3f" %diff2_ap)
print("Difference Precision: %.3f" %diff2_p)
print("Difference Recall: %.3f" %diff2_r)
print("Difference F1-score: %.3f" %diff2_f)
print("Difference Area Under ROC Curve: %.3f" %diff2_roc)
print("Difference Area Under PR Curve: %.3f" %diff2_pr)

print('*'*25 +' Normalized difference for In-In Vs Out-In ' + '*'*25) 
print("Difference TP: %.3f" %diff3_tp)
print("Difference FN: %.3f" %diff3_fn)
print("Difference TN: %.3f" %diff3_tn)
print("Difference FP: %.3f" %diff3_fp)
print("Difference Accuracy: %.3f" %diff3_acc)
print("Difference Average Precision: %.3f" %diff3_ap)
print("Difference Precision: %.3f" %diff3_p)
print("Difference Recall: %.3f" %diff3_r)
print("Difference F1-score: %.3f" %diff3_f)
print("Difference Area Under ROC Curve: %.3f" %diff3_roc)
print("Difference Area Under PR Curve: %.3f" %diff3_pr)
      