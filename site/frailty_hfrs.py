"""
Functions related to vitals from the INSPIRE dataset.
"""

import sys
import pandas as pd
from subject import Subject
import os
import inspire_dataset


def get_vital_names():
    vital_names = {
        'ffp',
        'epii',
        'phe',
        'ppfi',
        'pap_dbp',
        'etsevo',
        'hes',
        'hr',
        'd10w',
        'svi',
        'cpat',
        'rfti',
        'ds',
        'hns',
        'nibp_dbp',
        'ci',
        'ppf',
        'etdes',
        'mlni',
        'pepi',
        'fio2',
        'pap_sbp',
        'sti',
        'o2',
        'pc',
        'spo2',
        'art_dbp',
        'psa',
        'etiso',
        'nibp_mbp',
        'pap_mbp',
        'epi',
        'eph',
        'd5w',
        'hs',
        'alb20',
        'vt',
        'pip',
        'rr',
        'bis',
        'art_sbp',
        'aft',
        'alb5',
        'bt',
        'art_mbp',
        'dobui',
        'ns',
        'ntgi',
        'uo',
        'etgas',
        'dopai',
        'pplat',
        'vaso',
        'mdz',
        'ebl',
        'cryo',
        'n2o',
        'd50w',
        'etco2',
        'rbc',
        'ftn',
        'cvp',
        'peep',
        'pheresis',
        'sft',
        'stiii',
        'cbro2',
        'nepi',
        'air',
        'stv5',
        'nibp_sbp',
        'stii',
        'pmean',
        'minvol'}
    return vital_names



"""
Reads all packets from pcap into memory.
"""
def main():

    if len(sys.argv) != 3:
        print("Usage <directory of dataset> <subjects subset size>")
        print("Example ../dataset 50")
        return
    dataset_dir = sys.argv[1]

    #dataset_path = "./dataset/operations.csv"
    operations = inspire_dataset.read_csv_as_list( os.path.join(dataset_dir, "operations.csv") )
    df_operations = pd.DataFrame(operations)
    subject_operation_count = df_operations['subject_id'].value_counts(dropna=False)


    # =================================================== #
    # Create a separate object for each subject to hold
    #   labs, vitals, ward_vitals, and operations
    # =================================================== #
    subjects = dict()
    for subject_id in subject_operation_count.keys():
        #print(f"    {subject_id}")
        subjects[subject_id] = Subject(subject_id) # or create an object


    # =================================================== #
    # Read the operations, vitals, ward_vitals, labs, and diagnosis for each subject
    # =================================================== #
    vital_item_names = inspire_dataset.read_subjects_vitals(dataset_dir, subjects)
    print("ITEM: vital_item_names")
    print(vital_item_names)




if __name__ == "__main__":
    main()

