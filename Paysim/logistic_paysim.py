#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:37:09 2020

@author: shuvo
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import norm
import matplotlib.patches as mpatches
import time
import csv
from numpy import argmax
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
import collections
# Other Libraries
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score, auc
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# plot no skill and model roc curves
def plot_roc_curve(test_y, naive_probs, model_probs):
	# plot naive skill roc curve
	fpr, tpr, _ = roc_curve(test_y, naive_probs)
	plt.plot(fpr, tpr, linestyle='--', label='No Skill')
	# plot model roc curve
	fpr, tpr, _ = roc_curve(test_y, model_probs)
	plt.plot(fpr, tpr, marker='.', label='Logistic')
	# axis labels
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()
    

def plot_pr_curve(test_y, model_probs):
	# calculate the no skill line as the proportion of the positive class
	no_skill = len(test_y[test_y==1]) / len(test_y)
	# plot the no skill precision-recall curve
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	# plot model precision-recall curve
	precision, recall, _ = precision_recall_curve(y_test, model_probs)
	plt.plot(recall, precision, marker='.', label='Logistic')
	# axis labels
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()


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

# ******************************** With Original Data ********************************

print('*'*25 +'Training with Original Data, Testing With Original Data' + '*'*25)
tic1=time.time()

clf = LogisticRegression()
clf.fit(X_train, y_train)

tic2 = time.time()
elapsedtime=tic2-tic1
print("")
print("Training Time Taken : "+str(elapsedtime)+"seconds")

pred_y = clf.predict(X_test)

scn1_tn,scn1_fp,scn1_fn,scn1_tp=confusion_matrix(y_test,pred_y).ravel()

total_records1 = scn1_tn + scn1_fp + scn1_fn + scn1_tp
scn1_tn = (scn1_tn/total_records1) * 100
scn1_fp = (scn1_fp/total_records1) * 100
scn1_fn = (scn1_fn/total_records1) * 100
scn1_tp = (scn1_tp/total_records1) * 100

tic3 = time.time()
elapsedtime=tic3-tic2
print("")
print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
pred_y_probs = clf.predict_proba(X_test)
naive_probs = pred_y_probs[:,1]
roc_auc = roc_auc_score(y_test, naive_probs)
print('No skill ROC accuracy score %.2f: ' %roc_auc)
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
yhat = model.predict_proba(X_test)
model_probs = yhat[:, 1]
# calculate roc auc
roc_auc = roc_auc_score(y_test, model_probs)
print('Logistic ROC AUC %.3f' % roc_auc)
plot_roc_curve(y_test, naive_probs, model_probs)

# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(y_test, model_probs)
fscore = (2 * precision * recall)/ (precision + recall)
ix = argmax(fscore)
auc_score = auc(recall, precision)
#print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(y_test, model_probs)

scn1_acc = accuracy_score(y_test, pred_y) * 100
scn1_ap = average_precision_score(y_test, pred_y) * 100
scn1_p = precision_score(y_test, pred_y) * 100
scn1_r = recall_score(y_test, pred_y) * 100
scn1_f = f1_score(y_test, pred_y) * 100
scn1_roc = roc_auc_score(y_test, pred_y) * 100
scn1_pr = auc_score * 100
scn1_threshold = _[ix]
scn1_fscore = fscore[ix]

# # ******************************** With Synthetic Data ********************************

print('*'*25 +'Training with Synthetic Data, Testing With Synthetic Data' + '*'*25)
tic4=time.time()

clf = LogisticRegression()
clf.fit(XS_train, yS_train)

tic5 = time.time()
elapsedtime=tic5-tic4
print("")
print("Training Time Taken : "+str(elapsedtime)+"seconds")

predS_y = clf.predict(XS_test)

scn2_tn,scn2_fp,scn2_fn,scn2_tp=confusion_matrix(yS_test,predS_y).ravel()

total_records2 = scn2_tn + scn2_fp + scn2_fn + scn2_tp
scn2_tn = (scn2_tn/total_records2) * 100
scn2_fp = (scn2_fp/total_records2) * 100
scn2_fn = (scn2_fn/total_records2) * 100
scn2_tp = (scn2_tp/total_records2) * 100

tic6 = time.time()
elapsedtime=tic6-tic5
print("")
print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
pred_y_probsS = clf.predict_proba(XS_test)
naive_probsS = pred_y_probsS[:,1]
roc_aucS = roc_auc_score(yS_test, naive_probsS)
print('No skill ROC accuracy score %.2f: ' %roc_aucS)
model = LogisticRegression(solver='lbfgs')
model.fit(XS_train, yS_train)
yhatS = model.predict_proba(XS_test)
model_probsS = yhatS[:, 1]
# calculate roc auc
roc_aucS = roc_auc_score(yS_test, model_probsS)
print('Logistic ROC AUC %.3f' % roc_auc)
plot_roc_curve(yS_test, naive_probsS, model_probsS)

# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(yS_test, naive_probsS)
auc_score = auc(recall, precision)
print('No Skill PR AUC: %.3f' % auc_score)

# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(y_test, model_probs)
fscore = (2 * precision * recall)/ (precision + recall)
ix = argmax(fscore)
auc_score = auc(recall, precision)
#print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(y_test, model_probs)

scn2_acc = accuracy_score(yS_test, predS_y) * 100
scn2_ap = average_precision_score(yS_test, predS_y) * 100
scn2_p = precision_score(yS_test, predS_y) * 100
scn2_r = recall_score(yS_test, predS_y) * 100
scn2_f = f1_score(yS_test, predS_y) * 100
scn2_roc = roc_auc_score(yS_test, predS_y) * 100
scn2_pr = auc_score * 100
scn2_threshold = _[ix]
scn2_fscore = fscore[ix]

# ******************************** Testing Synthetic Data ********************************

print('*'*25 +'Training with Original Data, Testing With Synthetic Data' + '*'*25)
tic7=time.time()

clf = LogisticRegression()
clf.fit(X_train, y_train)

tic8 = time.time()
elapsedtime=tic8-tic7
print("")
print("Training Time Taken : "+str(elapsedtime)+"seconds")

predS_y = clf.predict(XS_test)

scn3_tn,scn3_fp,scn3_fn,scn3_tp=confusion_matrix(yS_test,predS_y).ravel()

total_records3 = scn3_tn + scn3_fp + scn3_fn + scn3_tp
scn3_tn = (scn3_tn/total_records3) * 100
scn3_fp = (scn3_fp/total_records3) * 100
scn3_fn = (scn3_fn/total_records3) * 100
scn3_tp = (scn3_tp/total_records3) * 100

tic9 = time.time()
elapsedtime=tic9-tic8
print("")
print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
pred_y_probsS = clf.predict_proba(XS_test)
naive_probsS = pred_y_probsS[:,1]
roc_aucS = roc_auc_score(yS_test, naive_probsS)
print('No skill ROC accuracy score %.2f: ' %roc_aucS)
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
yhatS = model.predict_proba(XS_test)
model_probsS = yhatS[:, 1]
# calculate roc auc
roc_aucS = roc_auc_score(yS_test, model_probsS)
print('Logistic ROC AUC %.3f' % roc_auc)
plot_roc_curve(yS_test, naive_probsS, model_probsS)

# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(y_test, model_probs)
fscore = (2 * precision * recall)/ (precision + recall)
ix = argmax(fscore)
auc_score = auc(recall, precision)
#print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(y_test, model_probs)

scn3_acc = accuracy_score(yS_test, predS_y) * 100
scn3_ap = average_precision_score(yS_test, predS_y) * 100
scn3_p = precision_score(yS_test, predS_y) * 100
scn3_r = recall_score(yS_test, predS_y) * 100
scn3_f = f1_score(yS_test, predS_y) * 100
scn3_roc = roc_auc_score(yS_test, predS_y) * 100
scn3_pr = auc_score * 100
scn3_threshold = _[ix]
scn3_fscore = fscore[ix]

# ******************************** Testing Original Data ********************************

print('*'*25 +'Training with Synthetic Data, Testing With Original Data' + '*'*25)
tic7=time.time()

clf = LogisticRegression()
clf.fit(XS_train, yS_train)

tic8 = time.time()
elapsedtime=tic8-tic7
print("")
print("Training Time Taken : "+str(elapsedtime)+"seconds")

predS_y = clf.predict(X_test)

scn4_tn,scn4_fp,scn4_fn,scn4_tp=confusion_matrix(y_test,predS_y).ravel()

total_records4 = scn4_tn + scn4_fp + scn4_fn + scn4_tp
scn4_tn = (scn4_tn/total_records4) * 100
scn4_fp = (scn4_fp/total_records4) * 100
scn4_fn = (scn4_fn/total_records4) * 100
scn4_tp = (scn4_tp/total_records4) * 100

tic9 = time.time()
elapsedtime=tic9-tic8
print("")
print("Prediction Time Taken : "+str(elapsedtime)+"seconds")
pred_y_probsS = clf.predict_proba(X_test)
naive_probsS = pred_y_probsS[:,1]
roc_aucS = roc_auc_score(y_test, naive_probsS)
print('No skill ROC accuracy score %.2f: ' %roc_aucS)
model = LogisticRegression(solver='lbfgs')
model.fit(XS_train, yS_train)
yhatS = model.predict_proba(X_test)
model_probsS = yhatS[:, 1]
# calculate roc auc
roc_aucS = roc_auc_score(y_test, model_probsS)
print('Logistic ROC AUC %.3f' % roc_aucS)
plot_roc_curve(y_test, naive_probsS, model_probsS)

# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(y_test, model_probs)
fscore = (2 * precision * recall)/ (precision + recall)
ix = argmax(fscore)
auc_score = auc(recall, precision)
#print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(y_test, model_probs)

scn4_acc = accuracy_score(y_test, predS_y) * 100
scn4_ap = average_precision_score(y_test, predS_y) * 100
scn4_p = precision_score(y_test, predS_y) * 100
scn4_r = recall_score(y_test, predS_y) * 100
scn4_f = f1_score(y_test, predS_y) * 100
scn4_roc = roc_auc_score(y_test, predS_y) * 100
scn4_pr = auc_score * 100
scn4_threshold = _[ix]
scn4_fscore = fscore[ix]

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
     
h1 = "Paysim_synt_TP_LogisticRegression_solver='lbfgs'"
h2 = "Paysim_synt_TN_LogisticRegression_solver='lbfgs'"
h3 = "Paysim_synt_FP_LogisticRegression_solver='lbfgs'"
h4 = "Paysim_synt_FN_LogisticRegression_solver='lbfgs'"
h5 = "Paysim_synt_Accuracy_LogisticRegression_solver='lbfgs'"
h6 = "Paysim_synt_AveragePrecision_LogisticRegression_solver='lbfgs'"
h7 = "Paysim_synt_Precision_LogisticRegression_solver='lbfgs'"
h8 = "Paysim_synt_Recall_LogisticRegression_solver='lbfgs'"
h9 = "Paysim_synt_F1-Score_LogisticRegression_solver='lbfgs'"
h10 = "Paysim_synt_ROC-Area_LogisticRegression_solver='lbfgs'"
h11 = "Paysim_synt_PR-Area_LogisticRegression_solver='lbfgs'"

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
myFile = open('paysim_logistic.csv', 'w')
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
print("Optimal Threshold: %.3f" %scn1_threshold)
print("Optimal F-score: %.3f" %scn1_fscore)

print('*'*25 +' Scenario 2 Results ' + '*'*25) # training with synthetic, testing with synthic
print("tn =%.3f" %scn2_tn,"fp =%.3f" %scn2_fp)
print("fn =%.3f" %scn2_fn,"tp =%.3f" %scn2_tp)
print("Accuracy Score: %.3f" %scn2_acc)
print("Average Precision Score: %.3f" %scn2_ap)
print("Precision Score: %.3f" %scn2_p)
print("Recall Score: %.3f" %scn2_r)
print("F1-Score: %.3f" %scn2_f)
print("Optimal Threshold: %.3f" %scn2_threshold)
print("Optimal F-score: %.3f" %scn2_fscore)

print('*'*25 +' Scenario 3 Results ' + '*'*25) # training with original, testing with synthetic
print("tn =%.3f" %scn3_tn,"fp =%.3f" %scn3_fp)
print("fn =%.3f" %scn3_fn,"tp =%.3f" %scn3_tp)
print("Accuracy Score: %.3f" %scn3_acc)
print("Average Precision Score: %.3f" %scn3_ap)
print("Precision Score: %.3f" %scn3_p)
print("Recall Score: %.3f" %scn3_r)
print("F1-Score: %.3f" %scn3_f)
print("Optimal Threshold: %.3f" %scn3_threshold)
print("Optimal F-score: %.3f" %scn3_fscore)

print('*'*25 +' Scenario 4 Results ' + '*'*25) # training with synthetic, testing with original
print("tn =%.3f" %scn4_tn,"fp =%.3f" %scn4_fp)
print("fn =%.3f" %scn4_fn,"tp =%.3f" %scn4_tp)
print("Accuracy Score: %.3f" %scn4_acc)
print("Average Precision Score: %.3f" %scn4_ap)
print("Precision Score: %.3f" %scn4_p)
print("Recall Score: %.3f" %scn4_r)
print("F1-Score: %.3f" %scn4_f)
print("Optimal Threshold: %.3f" %scn4_threshold)
print("Optimal F-score: %.3f" %scn4_fscore)

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
