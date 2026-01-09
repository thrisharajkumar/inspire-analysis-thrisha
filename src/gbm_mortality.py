# Taken from https://github.com/vitaldb/inspire/blob/main/gbm_mortality.py
"""
24. Lee, H. A machine learning-based prediction model for 30-day mortality after surgery using data from
INSPIRE. Github https://github.com/vitaldb/inspire/blob/main/gbm_mortality.py (2023).
"""
import numpy as np
import pandas as pd

# Set variables
OUTCOME_VAR = 'inhosp_death_30day'
INPUT_VARS = ['age', 'sex', 'emop', 'bmi', 'andur',
              'preop_hb', 'preop_platelet', 'preop_wbc', 
              'preop_aptt', 'preop_ptinr', 'preop_glucose',
              'preop_bun', 'preop_albumin', 'preop_ast', 
              'preop_alt', 'preop_creatinine', 'preop_sodium', 
              'preop_potassium']

# Load operations
df = pd.read_csv('C:\\Users\\pc\\Desktop\\dataset\\operations.csv')

# find the first operation for each patient
df.sort_values('orin_time', inplace=True)
df = df.loc[df[['op_id','subject_id']].groupby('subject_id')['op_id'].idxmin()]

print(f"TYPE 'inhosp_death_time' {df['inhosp_death_time'].dtype}")
print(f"TYPE 'orout_time' {df['orout_time'].dtype}")

df[OUTCOME_VAR] = (df['inhosp_death_time'] < df['orout_time'] + 30 * 24 * 60)
df = df[(df['asa'] < 6)]
df.loc[:, 'andur'] = df['anend_time'] - df['anstart_time']

valid_mask = df['height'] > 10
df['bmi'] = np.nan
df.loc[valid_mask, 'bmi'] = df.loc[valid_mask, 'weight'] / (df.loc[valid_mask, 'height'] / 100) ** 2

# Load labs
df_lab = pd.read_csv("C:\\Users\\pc\\Desktop\\dataset\\labs.csv")
for item_name in ('hb', 'platelet', 'aptt', 'wbc', 'ptinr', 'glucose', 'bun', 'albumin', 'ast', 'alt', 'creatinine', 'sodium', 'potassium'):
    df = pd.merge_asof(df.sort_values('orin_time'),
                    df_lab.loc[df_lab['item_name'] == item_name].sort_values('chart_time'),
                    left_on='orin_time', right_on='chart_time', by='subject_id',
                    tolerance=6* 30 * 24 * 60, suffixes=('', '_'))
    df.drop(columns=['chart_time', 'item_name'], inplace=True)
    df.rename(columns={'value':f'preop_{item_name}'}, inplace=True)






df['sex'] = df['sex'] == 'M'

#-----------------------------------------------#
# Debugging
#-----------------------------------------------#
# print("DF Features")
# for col_name, col_type in zip(df.columns, df.dtypes):
#     print(f"Column: {col_name}, Type: {col_type}")
# print(df)
df.to_csv('gbm_mortality.csv', index=False)
#-----------------------------------------------#

#print(df.astype({'inhosp_death_30day':int}).quantile([0, 0.25, 0.5, 0.75, 1]))
# Split a dataset into train and test sets
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
ntrain = int(len(df) * 0.7)
y_train = df.loc[:ntrain, OUTCOME_VAR]
x_train = df.loc[:ntrain, INPUT_VARS].astype(float)
y_test = df.loc[ntrain:, OUTCOME_VAR]
x_test = df.loc[ntrain:, INPUT_VARS].astype(float)

# Print the number of train and test sets
print(f'{sum(y_train)}/{len(y_train)} ({np.mean(y_train)*100:.2f}%) train, {sum(y_test)}/{len(y_test)} ({np.mean(y_test)*100:.2f}%) test, {x_train.shape[1]} features', flush=True)

# ASA class
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve

# ASA healthy = 1, then worse, e.g. 4 may not survive
# 3, 4, 5 is very sick and iffy will survive surgery
# 6 is max and patient is brain dead
y_pred_asa = df.loc[ntrain:, 'asa']

# Compute area under the ROC AUC from prediction scores
auroc_asa = roc_auc_score(y_test, y_pred_asa)

# Compute the precision recall curve
prc_asa, rec_asa, thresholds = precision_recall_curve(y_test, y_pred_asa)

# Compute area under the precision-recall curve
auprc_asa = auc(rec_asa, prc_asa)
print('ASA auroc: {:.3f}, auprc: {:.3f}'.format(auroc_asa, auprc_asa), flush=True)

# Logistic regression using SimpleImputer()
from sklearn.impute import SimpleImputer
imp = SimpleImputer().fit(x_train)
x_train_imputed = imp.transform(x_train)
x_test_imputed = imp.transform(x_test)

# Logistic regression using LogisticRegression()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000).fit(x_train_imputed, y_train)
y_pred_lr = model.predict_proba(x_test_imputed)[:, 1]

"""
Since the model is performing binary classification, the output is a 2D
array where each row corresponds to a test sample and each row contains
two probabilities: the probability of the sample belonging to the
negative class (class 0) and the probability of belonging to the
positive class (class 1).

The [:, 1] indexing selects only the second column of this array,
which contains the predicted probabilities for the positive class (class 1).
This is a common practice when the primary interest is in the likelihood
of the positive outcome, such as the probability of a customer churning
or a patient having diabetes.
"""

# Compute AUROC and AUPRC
auroc_lr = roc_auc_score(y_test, y_pred_lr)
prc_lr, rec_lr, thresholds = precision_recall_curve(y_test, y_pred_lr)
auprc_lr = auc(rec_lr, prc_lr)
print('LR auroc: {:.3f}, auprc: {:.3f}'.format(auroc_lr, auprc_lr), flush=True)

# # Gradient Boosting using XGBClassifier()
# from xgboost import XGBClassifier
# model = XGBClassifier(max_depth=4, n_estimators=50, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
# model.fit(x_train, y_train)
# y_pred_gbm = model.predict_proba(x_test)[:, 1]
#
# # Compute AUROC and AUPRC
# auroc_gbm = roc_auc_score(y_test, y_pred_gbm)
# prc_gbm, rec_gbm, thresholds = precision_recall_curve(y_test, y_pred_gbm)
# auprc_gbm = auc(rec_gbm, prc_gbm)
# print(f'GBM auroc: {auroc_gbm:.3f}, auprc: {auprc_gbm:.3f}', flush=True)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fpr_asa, tpr_asa, _ = roc_curve(y_test, y_pred_asa)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
# fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_pred_gbm)

plt.figure(figsize=(5,5))
plt.plot(fpr_asa, tpr_asa, label='ASA = {:0.3f}'.format(auroc_asa))
plt.plot(fpr_lr, tpr_lr, label='LR = {:0.3f}'.format(auroc_lr))
# plt.plot(fpr_gbm, tpr_gbm, label='GBM = {:0.3f}'.format(auroc_gbm))
plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('gbm_auroc.png')
plt.show()

# Compute the confusion matrix, only for hard classification!!!
# class_names = [False, True]
# cm = confusion_matrix(y_test, y_pred_lr, labels=class_names)
#
# # Create and plot the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot()
# plt.show()