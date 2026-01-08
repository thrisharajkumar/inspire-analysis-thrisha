# Utility class to read parameters.csv in the INSPIRE dataset


"""
Table,Label,Unit,Description
labs,albumin,g/dL,Albumin
labs,alp,IU/L,Alkaline phosphatase
labs,alt,IU/L,Alanine transaminase
labs,aptt,sec,Activated partial thromboplastin time
labs,ast,IU/L,Aspartate transaminase
labs,be,mmol/L,Base excess in arterial blood gas analysis
labs,bun,mg/dL,Blood urea nitrogen
labs,calcium,mg/dL,Calcium
labs,chloride,mmol/L,Chloride

There are only three unique Tables {labs, vitals, ward_vitals}

NB Some rows/labels are duplicated between tables
However, all have the same (or largely same) description
Therefore, we decide to simply map Label -> Description ans omit the table

ward_vitals	nibp_dbp	mmHg	Non-invasive diastolic blood pressure
vitals	nibp_mbp	mmHg	Non-invasive mean blood pressure
ward_vitals	nibp_mbp	mmHg	Non-invasive mean blood pressure
vitals	nibp_sbp	mmHg	Non-invasive systolic blood pressure
ward_vitals	nibp_sbp	mmHg	Non-invasive systolic blood pressure
"""

import inspire_dataset
import sys
import os


def read_parameters( dataset_dir = "../dataset/"):
    parameters = inspire_dataset.read_csv_as_list(os.path.join(dataset_dir, "parameters.csv"))
    label2unit = dict()  # just to make sure units are same between tables
    label2desc = dict()
    for parameter in parameters:
        table = parameter['Table']
        label = parameter['Label']
        unit = parameter['Unit']
        desc = parameter['Description']
        # print(parameter)
        if label not in label2desc:
            label2desc[label] = desc
            label2unit[label] = unit
        # else:
        #     orig_desc = label2desc[label]
        #     orig_unit = label2unit[label]
        #     if desc != orig_desc:
        #         print(f"Duplicate for {label}, new {desc} orig {orig_desc}, keeping orig")
        #     if unit != orig_unit:
        #         print(f"Duplicate for {label}, new {unit} orig {orig_unit}, keeping orig")
    return label2desc, label2unit


def main():
    if len(sys.argv) != 2:
        print("Usage <directory of dataset>")
        print("Example ../dataset")
        return
    dataset_dir = sys.argv[1]

    parameters = inspire_dataset.read_csv_as_list(os.path.join(dataset_dir, "parameters.csv"))

    label2unit = dict() # just to make sure units are same between tables
    label2desc = dict()
    for parameter in parameters:
        table = parameter['Table']
        label = parameter['Label']
        unit  = parameter['Unit']
        desc  = parameter['Description']
        #print(parameter)
        if label in label2desc:
            orig_desc = label2desc[label]
            orig_unit = label2unit[label]
            if desc != orig_desc:
                print(f"Duplicate for {label}, new {desc} orig {orig_desc}, keeping orig")
            if unit != orig_unit:
                print(f"Duplicate for {label}, new {unit} orig {orig_unit}, keeping orig")
        else:
            label2desc[label] = desc
            label2unit[label] = unit

    for label in label2desc.keys():
        desc = label2desc[label]
        print(f"{label} -> {desc}")


if __name__ == "__main__":
    main()