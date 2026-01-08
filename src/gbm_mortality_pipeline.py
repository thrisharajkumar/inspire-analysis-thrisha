"""
Analyses json subject files (derived from INSPIRE dataset).
This implementation mirrors the gbm_mortality.py module, that uses Pandas operations,
but replaces with preprocessing/data preparation using objects.  This approach is
not only more intuitive but also more general (i.e. can be used for many others approaches).
"""
import math

from stopwatch import Stopwatch
import subject as subject_module

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve

# This was an attempt to compare the "gbm" features against a deep learning approach.
# However, a very poor approach, only 18 features, not much dnn can learn.
# We have since switched to looking at all the data (not just these 18 manual features)
# as input to dnn learning models.  It is not an "apples to apples" comparison now
# but still possible to assess the efficacy of dnns for feature learning for mortality prediction.
# import old_dnn_training


# def convert_vitals_to_dictionary(vitals):
#     vitals_dict = dict()
#     for subject_vital in vitals:
#         # op_id      = subject_vital['op_id']
#         # subject_id = subject_vital['subject_id']
#         chart_time = int(subject_vital['chart_time'])
#         item_name = subject_vital['item_name']
#         value = float(subject_vital['value'])
#
#         if item_name not in vitals_dict:
#             vitals_dict[item_name] = dict()
#         series = vitals_dict[item_name]
#         series[chart_time] = value
#     return vitals_dict


# def equal_dicts(dict1, dict2):
#     """
#     Unfortunately == operator requires all key/types to be same (if not, throws error).
#     :param dict1:
#     :param dict2:
#     :return: True if equal, False otherwise
#     """
#     keys_to_check = [
#         # 'subject_id',
#         'age',
#         'sex',
#         'emop',
#         'bmi',
#         'andur',
#         'preop_hb',
#         'preop_platelet',
#         'preop_aptt',
#         'preop_wbc',
#         'preop_ptinr',
#         'preop_glucose',
#         'preop_bun',
#         'preop_albumin',
#         'preop_ast',
#         'preop_alt',
#         'preop_creatinine',
#         'preop_sodium',
#         'preop_potassium',
#         'inhosp_death_30day',
#     ]
#     for key in keys_to_check:
#         d1value = dict1[key]
#         d2value = dict2[key]
#         if type(d1value) is str:
#             raise ValueError(f'Expected dict1 {key} to be number, got str {d1value}')
#         if type(d2value) is str:
#             raise ValueError(f'Expected dict2 {key} to be number, got str {d2value}')
#
#         if math.isnan(d1value) and math.isnan(d2value):
#             # print(f"Key {key} are both nan {d1value} == {d2value}")
#             return True
#
#         if d1value != d2value:
#             print(f"Key {key} not equal {d1value} != {d2value}")
#             return False
#     return True


def calculate_bmi(weight_kg, height_m):
    """
    Calculate Body Mass Index (BMI).  Not valid if height <= 10

    Parameters:
    weight_kg (float): Weight in kilograms
    height_m (float): Height in meters

    Returns:
    float: The BMI value or None if not valid
    """
    # valid_mask = df['height'] > 10
    # df['bmi'] = np.nan
    # df.loc[valid_mask, 'bmi'] = df.loc[valid_mask, 'weight'] / (df.loc[valid_mask, 'height'] / 100) ** 2
    if height_m < 10/100 or weight_kg <= 0:
        return None

    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    if weight_kg <= 0: # can't be reached because of check above but keep just in case
        raise ValueError("Weight must be greater than zero.")

    bmi = weight_kg / (height_m ** 2)
    return bmi




def main():
    # Set variables
    OUTCOME_VAR = 'inhosp_death_30day'
    INPUT_VARS = ['age', 'sex', 'emop', 'bmi', 'andur',
                  'preop_hb', 'preop_platelet', 'preop_wbc',
                  'preop_aptt', 'preop_ptinr', 'preop_glucose',
                  'preop_bun', 'preop_albumin', 'preop_ast',
                  'preop_alt', 'preop_creatinine', 'preop_sodium',
                  'preop_potassium']

    parent_dir = '../inspire_subjects'

    stopwatch = Stopwatch()
    subjects = subject_module.read_subjects(parent_dir)
    print(f"Reading took {stopwatch.elapsedTime() / 60:.2f} minutes")

    # Print some stats
    subject_module.subjects_statistics(subjects)

    # We want a dict of dicts (easy to create Pandas dataframe, list of dicts is also easy)
    rows = dict()
    for subject_id in subjects.keys():
        subject = subjects[subject_id]
        first_operation = subject.get_first_operation()

        # Could copy each iem over and add couple of items,
        # but easier to make copy of operation
        # shallow copy is sufficient
        row = first_operation.copy()

        # Convert age from str to int
        row['age'] = int(row['age'])

        # Convert sex to a number (well, a bool which extends int)
        # our_df['sex'] = (our_df['sex'] == 'M')
        row['sex'] = (row['sex'] == 'M')

        # Convert emop to a number, most seem to be 0 or 1
        row['emop'] = int(row['emop'])

        # First the last lab before the first operation
        orin_time    = int(row['orin_time'].strip())
        # orout_time   = int(first_operation['orout_time'].strip())
        # opstart_time = int(first_operation['opstart_time'].strip())
        # opend_time   = int(first_operation['opend_time'].strip())

        # Convert asa from str to int for "asa classifier"
        asa_str = row['asa'].strip()
        if len(asa_str) > 0:
            row['asa'] = int(asa_str)
        else:
            row['asa'] = None # will be imputed, likely 1?

        row['preop_hb'] = subject.get_most_recent_lab('hb', orin_time)
        row['preop_platelet'] = subject.get_most_recent_lab('platelet', orin_time)
        row['preop_aptt'] = subject.get_most_recent_lab('aptt', orin_time)
        row['preop_wbc'] = subject.get_most_recent_lab('wbc', orin_time)
        row['preop_ptinr'] = subject.get_most_recent_lab('ptinr', orin_time)
        row['preop_glucose'] = subject.get_most_recent_lab('glucose', orin_time)
        row['preop_bun'] = subject.get_most_recent_lab('bun', orin_time)
        row['preop_albumin'] = subject.get_most_recent_lab('albumin', orin_time)
        row['preop_ast'] = subject.get_most_recent_lab('ast', orin_time)
        row['preop_alt'] = subject.get_most_recent_lab('alt', orin_time)
        row['preop_creatinine'] = subject.get_most_recent_lab('creatinine', orin_time)
        row['preop_sodium'] = subject.get_most_recent_lab('sodium', orin_time)
        row['preop_potassium'] = subject.get_most_recent_lab('potassium', orin_time)

        # gbm does this, note this is duration of operation attribute, is this predictive?
        # df.loc[:, 'andur'] = df['anend_time'] - df['anstart_time']
        anend_time_str = row['anend_time']
        anstart_time_str = row['anstart_time']
        if len(anend_time_str) > 0 and len(anstart_time_str) > 0:
            anend_time = int(anend_time_str)
            anstart_time = int(anstart_time_str)
            row['andur'] = anend_time - anstart_time
        else:
            row['andur'] = None

            # Compute bmi fromn weight and height, can return None if not valid, will be replaced impute
        weight_kg_str = row['weight']
        height_m_str = row['height'] # NB: this is in cm!!!  Needs to be meters to computer BMI
        if len(weight_kg_str) > 0 and len(height_m_str) > 0:
            weight_kg = float(weight_kg_str)
            height_m  = float(height_m_str) / 100.0
            bmi = calculate_bmi(weight_kg, height_m)
            row['bmi'] = bmi
            # print(f"{subject_id} BMI( weight {weight_kg},  height {height_m}) = {bmi}")
        else:
            row['bmi'] = None

        row[OUTCOME_VAR] = subject.inhosp_death_30day()

        # rows.append(row)
        rows[int(subject_id)] = row
    # df = pd.DataFrame(rows)
    our_df = pd.DataFrame.from_dict(rows, orient='index')
    print(our_df.head())

    # -------------------------------------------#
    # For some reason we exclude asa 6 (already know they likely will die?)
    # Barely changes classification results, not sure why this was done but negligible.
    # -------------------------------------------#
    # print(f"Rows before drop asa 6? {len(our_df)}")
    # X_train (69920, 18)  total 99886
    # X_test (29966, 18)
    # y_train (69920,)
    # y_test (29966,)

    our_df = our_df[(our_df['asa'] < 6)]
    # print(f"Rows after drop  asa 6? {len(our_df)}")
    # Rows before drop asa 6? 99886
    # Rows after drop  asa 6? 97260
    # X_train (68082, 18)
    # X_test (29178, 18)
    # y_train (68082,)
    # y_test (29178,)

    # -------------------------------------------#
    # Pick subset of features to mirror gbm model
    # -------------------------------------------#

    # Need to handle about 10k rows with nans, either drop or impute?
    # print(f"Rows before drop nans? {len(our_df)}")
    # our_df = our_df.dropna()
    # print(f"Rows after drop nans? {len(our_df)}")

    # our_df['sex'] = (our_df['sex'] == 'M')

    df = our_df[INPUT_VARS+[OUTCOME_VAR]]

    #-------------------------------------------#
    # Sanity Check
    # -------------------------------------------#
    # # We want to see if we derive the same features as gmb_mortality (i.e. the INSPIRE paper)
    # gbm_dataframe = pd.read_csv('gbm_mortality.csv')
    # # for col_name, col_type in zip(gbm_dataframe.columns, gbm_dataframe.dtypes):
    # #     print(f"Column: {col_name}, Type: {col_type}")
    # count = 0
    # for index, gmb_row in gbm_dataframe.iterrows():
    #     subject_id = gmb_row['subject_id']
    #     if subject_id in our_df.index:
    #         # Sadly, gbm converted from str to int in many cases, including subject_id
    #         our_row = df.loc[subject_id]
    #
    #         our_label = our_row[OUTCOME_VAR]
    #         gbm_label = gmb_row[OUTCOME_VAR]
    #
    #         our_label_type = type(our_label)
    #         gbm_label_type = type(gbm_label)
    #
    #
    #         # print("")
    #         # print(f"GBM: {gmb_row}")
    #         # print(f"OUR: {our_row}")
    #         # print(f"Equal: {our_row == gmb_row}")
    #         print( f"{count} Equal {subject_id}: {equal_dicts(our_row, gmb_row)}   label gmb {gbm_label}  our {our_label} our label type={our_label_type}  gbm type {gbm_label_type}" )
    #     else:
    #         in_our_dict = subject_id in rows
    #         print(f"{count} Could not find {subject_id} in our dataframe, in our dict? {in_our_dict}")
    #
    #     if count > 10:
    #         break
    #     count += 1
    # # I used to save this to compare against the saved gbm_mortality (Pandas) csv.
    # # I believe both are effectively the exact same now, so no longer need to save.
    # df.to_csv('dnn_mortality.csv', index=True, header=True)
    # -------------------------------------------#

    df_y = df[OUTCOME_VAR]  # Target variable

    # Separate features (X) and target (y)
    df_X = df.drop( OUTCOME_VAR, axis=1)  # Features (without 'target')

    # Print feature names and their types using a for loop
    # for col_name, col_type in X.dtypes.items():
    #     print(f"Before Feature: {col_name}, Type: {col_type}")
    df_X = df_X.astype(float)


    # Print feature names and their types using a for loop
    # for col_name, col_type in X.dtypes.items():
    #     print(f"After Feature: {col_name}, Type: {col_type}")

    # Split both X and y into train and test sets
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)

    # Convert to numpy (i think the split used to do this but now returns dataframes)
    X_train = df_X_train.to_numpy()
    X_test  = df_X_test.to_numpy()
    y_train = df_y_train.to_numpy()
    y_test  = df_y_test.to_numpy()

    # Handle missing or nan values using imputer
    imp = SimpleImputer().fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)

    # Outputs are NumPy arrays
    print(f"X_train {X_train.shape}")  # <class 'numpy.ndarray'>
    print(f"X_test {X_test.shape}")  # <class 'numpy.ndarray'>
    print(f"y_train {y_train.shape}")  # <class 'numpy.ndarray'>
    print(f"y_test {y_test.shape}")  # <class 'numpy.ndarray'>

    # -------------------------------------------#
    # Modelling
    # -------------------------------------------#
    # Instantiate the LogisticRegression model
    # logreg = LogisticRegression(max_iter=5000, random_state=16)
    logreg = LogisticRegression(max_iter=5000)
    # Fit the model on the training data
    logreg.fit(X_train, y_train)

    # Predict the target values for the test set
    # -------------------------------------------#
    # Calculate metrics for soft classification (NB: can adjust/select threshold)
    # -------------------------------------------#
    y_pred_lr = logreg.predict_proba(X_test)[:, 1]  # soft classification
    # Compute AUROC and AUPRC
    auroc_lr = roc_auc_score(y_test, y_pred_lr)
    prc_lr, rec_lr, thresholds = precision_recall_curve(y_test, y_pred_lr)
    print(f"LR Precision {prc_lr}  Recall {rec_lr}")
    auprc_lr = auc(rec_lr, prc_lr)
    print('LR auroc: {:.3f}, auprc: {:.3f}'.format(auroc_lr, auprc_lr), flush=True)

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    # fpr_gbm, tpr_gbm, _ = roc_curve(y_test, y_pred_gbm)

    plt.figure(figsize=(5, 5))
    #plt.plot(fpr_asa, tpr_asa, label='ASA = {:0.3f}'.format(auroc_asa))
    plt.plot(fpr_lr, tpr_lr, label='LR = {:0.3f}'.format(auroc_lr))
    # plt.plot(fpr_gbm, tpr_gbm, label='GBM = {:0.3f}'.format(auroc_gbm))
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('auroc.png')
    plt.show()


    """
    GBM Results
    ASA auroc: 0.859, auprc: 0.105
    LR auroc: 0.949, auprc: 0.203
    
    OUR Results (noting we used a different train/test split)
    LR auroc: 0.938, auprc: 0.141
    """
    # print("Starting deep learning experiment")
    # epochs = 32
    # old_dnn_training.deep_learning_experiment(X_train, y_train, X_test, y_test, epochs)




if __name__ == "__main__":
    main()

"""
filename = "140696052.json"

TAKEN FROM gbm_mortality.py
op_id	subject_id	hadm_id	case_id	opdate	age	sex	weight	height	race	asa	emop	department	antype	icd10_pcs	orin_time	orout_time	opstart_time	opend_time	admission_time	discharge_time	anstart_time	anend_time	cpbon_time	cpboff_time	icuin_time	icuout_time	inhosp_death_time	allcause_death_time	inhosp_death_30day	andur	bmi	preop_hb	preop_platelet	preop_aptt	preop_wbc	preop_ptinr	preop_glucose	preop_bun	preop_albumin	preop_ast	preop_alt	preop_creatinine	preop_sodium	preop_potassium
403543518	164935053	251633071		0	45	M	65.0	180.0	Asian	1.0	1	OS	General	0R900	5	310	65.0	300.0	0	63355	15.0	295.0							FALSE	280.0	20.061728395061700	14.2	245.0	25.1		0.95		18.0	4.1	108.0	35.0	0.92	137.0	4.5
429493668	196269490	269094492		0	55	F	60.0	160.0	Asian	2.0	1	NS	General	009T0	10	300	120.0	285.0	0	10075	50.0	290.0			290.0	2575.0			FALSE	240.0	23.437500000000000	11.9	233.0	25.1	13.79	0.95	120.0	16.0	3.9	11.0	13.0	0.58	143.0	3.8
490288058	143981214	268056914	1211.0	0	75	M			Asian	3.0	1	GS	General	03CY0	10	435	65.0	430.0	0	18715	20.0	430.0			430.0	5175.0			FALSE	410.0		12.2	291.0	32.3	9.51	1.03	153.0	36.0	3.9	14.0	9.0	3.04	139.0	4.0
453052475	140696052	251635295		0	75	F	50.0	155.0	Asian	3.0	1	RAD	General	03LG0	20	275	100.0	270.0	0	15835	70.0	270.0			1000.0	15100.0	15100.0	18720.0	TRUE	200.0	20.811654526534900	11.0	200.0	29.6	7.9	0.9	110.0	50.0	3.6	19.0	11.0	5.55	132.0	5.3
"""









