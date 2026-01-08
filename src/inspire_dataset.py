"""
Analyses csv files, from INSPIRE dataset.
"""

from stopwatch import Stopwatch

import sys
import csv
import pandas as pd
import os
from subject import Subject
import vitals

DEPARTMENTS = {"CTS", #Cardio-Thoracic Surgery
                "GS", #General Surgery
                "NS", #Neurosurgery
                "OG", #Obstetrics & Gynecology
                "OL", #Otolaryngology
                "OS", #Orthopedic Surgery
                "OT", #Ophthalmology
                "PS", #Plastic Surgery
                "UR", #Urology
               }


def read_csv_as_list(csv_file_name, subjects=None):
    """
    Reads csv_file_name and returns as a list of dict objects (compatible with Dataframe(list).
    :param csv_file_name:
    :param subjects: None to read all subjects, dict with subject_id as key to read
    :return: list of dicts (if subjects not none, only those subjects)
    """
    # easier to just create list of dict and then create dataframe
    csv_list = list()
    #NB: the 'utf-8-sig' removes the byte order mark BOM from first column header '\ufeffsubject_id'
    with open(csv_file_name, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if subjects is None:
                csv_list.append(row)
            elif row["subject_id"] in subjects:
                csv_list.append(row)

    return csv_list

# Old way of removing BOM
# df.columns = [remove_byte_order_mark(col) for col in df.columns]
def remove_byte_order_mark(col):
    return col.replace('\ufeff', '')




def read_subjects_vitals( dataset_dir, subjects ):

    # Sadly, multiple threads did not help, disk must be bottleneck

    # =================================================== #
    # Linearly go through operations adding to subject
    # =================================================== #
    file_path = os.path.join( dataset_dir, 'operations.csv')
    operations = read_csv_as_list(file_path, subjects)
    for operation_dict in operations:

        # op = Operation()
        # op.fromJson(operation_dict)
        # print(f"OPERATION: {op}")

        subject_id = operation_dict["subject_id"]
        # if subject_id not in subjects:
        #    raise ValueError(f"Subject {subject_id} must have an operation, subject dict was created from operations")
        if subject_id in subjects:
            subject = subjects[subject_id]
            subject.add_operation(operation_dict)

            # op = Operation()
            # op.fromJson(operation_dict)
            # subject.add_operation(op)

    # ====================== ============================= #
    # Linearly go through labs adding to subject
    # =================================================== #
    file_path = os.path.join(dataset_dir, 'labs.csv')
    labs = read_csv_as_list(file_path, subjects)
    for lab in labs:
        subject_id = lab["subject_id"]
        #if subject_id not in subjects:
        #    print(f"LABS: Could not find subject {subject_id}, does not have operation, creating with no operation")
        #    subjects[subject_id] = Subject(subject_id)
        if subject_id in subjects:
            subject = subjects[subject_id]
            subject.add_lab(lab)

    # =================================================== #
    # Linearly go through ward vitals adding to subject
    # =================================================== #
    file_path = os.path.join(dataset_dir, 'ward_vitals.csv')
    ward_vitals = read_csv_as_list(file_path, subjects)
    for ward_vital in ward_vitals:
        subject_id = ward_vital["subject_id"]
        #if subject_id not in subjects:
        #    print(f"WARD_VITALS: Could not find subject {subject_id}, does not have operation, creating with no operation")
        #    subjects[subject_id] = Subject(subject_id)
        if subject_id in subjects:
            subject = subjects[subject_id]
            subject.add_ward_vital(ward_vital)

    # =================================================== #
    # Linearly go through vitals (specific to an operation) adding to subject
    # =================================================== #
    vital_item_names = set()
    file_path = os.path.join(dataset_dir, 'vitals.csv')
    vitals = read_csv_as_list(file_path, subjects)
    for vital in vitals:
        subject_id = vital["subject_id"]
        item_name = vital["item_name"]
        vital_item_names.add(item_name)
        # if subject_id not in subjects:
        #    raise ValueError(f"Subject {subject_id} must have an vital associated with operation, subject dict was created from operations")
        if subject_id in subjects:
            subject = subjects[subject_id]
            subject.add_vital(vital)

    # =================================================== #
    # Linearly go through diagnosis adding to subject
    # =================================================== #
    file_path = os.path.join(dataset_dir, 'diagnosis.csv')
    diagnoses = read_csv_as_list(file_path, subjects)
    for diagnosis in diagnoses:
        subject_id = diagnosis["subject_id"]
        #chart_time = diagnosis["chart_time"]
        #icd10_cm   = diagnosis["icd10_cm"]
        # if subject_id not in subjects:
        #    raise ValueError(f"Subject {subject_id} must have an diagnosis associated with operation, subject dict was created from operations")
        if subject_id in subjects:
            subject = subjects[subject_id]
            subject.add_diagnosis(diagnosis)

    # =================================================== #
    # Linearly go through medications adding to subject
    # =================================================== #
    file_path = os.path.join(dataset_dir, 'medications.csv')
    medications = read_csv_as_list(file_path, subjects)
    for medication in medications:
        subject_id = medication["subject_id"]
        # if subject_id not in subjects:
        #    raise ValueError(f"Subject {subject_id} must have an diagnosis associated with operation, subject dict was created from operations")
        if subject_id in subjects:
            subject = subjects[subject_id]
            subject.add_medication(medication)

    return vital_item_names


# def print_department_operations(inspire_operation_csv_filepath):
#     """
#     Reads and prints CSV file assumed to be from the INSPIRE dataset.
#     :param inspire_operation_csv_filepath:
#     :return:
#     """
#     #
#     # if len(sys.argv) != 2:
#     #     print("Usage <csv_file>")
#     #     return
#     # inspire_csv_filepath = sys.argv[1]
#     operations_filepath = "../dataset/operations.csv"
#     departments_filepath = "../dataset/department.csv"
#
#     #df = read_operations(department={"GS"})
#     depts = read_csv_as_list(departments_filepath)
#
#     for department in depts.keys():
#         df = read_operations(department)
#         #df = read_all_operations()
#         #print(df)
#         print(f"DEPARTMENT {department} = {df.shape}")



def print_csv(inspire_csv_filepath):
    """
    Reads and prints CSV file assumed to be from the INSPIRE dataset.
    :param inspire_csv_filepath:
    :return:
    """
    stopwatch = Stopwatch()

    operations = read_csv_as_list(inspire_csv_filepath)
    df_operations = pd.DataFrame(operations)

    print(f"CSV File {inspire_csv_filepath}")
    print(df_operations)
    print(f"Reading took {stopwatch.elapsedTime():.2f} seconds")



def main():
    """
    Reads raw INSIPRE dataset csv files and save each subject as
    JSON file in either died or survived output directory.
    :return:
    """
    # python inspire_dataset_subjects.py ../dataset ../inspire_subjects
    if len(sys.argv) != 3:
        print("Usage <directory of dataset> <directory of output json files>")
        print("Example ../dataset ../inspire_subjects")
        return
    dataset_dir = sys.argv[1] # directory of INSPIRE dataset (i.e. where the csv files are located)
    output_dir  = sys.argv[2] # directory where to save the subject JSON files

    stopwatch = Stopwatch()

    #dataset_path = "./dataset/operations.csv"
    operations = read_csv_as_list( os.path.join(dataset_dir, "operations.csv") )
    df_operations = pd.DataFrame(operations)
    subject_operation_count = df_operations['subject_id'].value_counts(dropna=False)
    # 123456 -> 2   (subject_id 123456 had 2 operations)

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
    vital_item_names = read_subjects_vitals(dataset_dir, subjects)

    dir_died     = os.path.join(output_dir,     'died')
    dir_survived = os.path.join(output_dir, 'survived')

    # Figure out if patient died?  And save
    total_count    = 0
    death_count    = 0
    survived_count = 0
    for subject_id in subjects.keys():
        subject = subjects[subject_id]

        filename = f"{subject.get_subject_id()}.json"

        total_count += 1
        #print(f"SUBJECT {subject_id}")
        if subject.died():
            death_count += 1
            # Writing to a JSON file
            # output_dir = "../inspire_subjects"

            #filepath = filedir + "/" + filename
            filepath = os.path.join(dir_died,filename)

            operations = subject.get_operations()

            inhosp_death_30day = subject.inhosp_death_30day()
            print(f"Writing subject that died ({len(operations)} operations) to file {filepath} 30 day death? {inhosp_death_30day}")
            subject.toJSON(filepath)
        else:
            survived_count += 1
            filepath = os.path.join(dir_survived, filename)
            subject.toJSON(filepath)

    print(f"Total operations {total_count} (sanity check {death_count+survived_count})")
    print(f"Counted {death_count} operations died")
    print(f"Counted {survived_count} operations survived")
    print(f"Reading took {stopwatch.elapsedTime() / 60:.2f} minutes")


    vital_item_names = vitals.get_vital_names()
    print("ITEM: vital_item_names")
    print(vital_item_names)




if __name__ == "__main__":
    main()

"""
DEPARTMENT AN = (   68, 29)
DEPARTMENT CTS= ( 8754, 29)
DEPARTMENT DM = (    1, 29)
DEPARTMENT EM = (    2, 29)
DEPARTMENT GS = (34725, 29)
DEPARTMENT IM = (   89, 29)
DEPARTMENT NS = (10172, 29)
DEPARTMENT OG = (12946, 29)
DEPARTMENT OL = (11702, 29)
DEPARTMENT OS = (17433, 29)
DEPARTMENT OT = (17249, 29)
DEPARTMENT PED= (   38, 29)
DEPARTMENT PS = ( 5157, 29)
DEPARTMENT RAD= (  379, 29)
DEPARTMENT RO = (   15, 29)
DEPARTMENT UR = (12230, 29)


Subjects that had more than one operations 21565
Sum of operations 130960
How many subjects 99886


op_id,subject_id,hadm_id,case_id,opdate,age,sex,weight,height,race,asa,emop,department,antype,icd10_pcs,orin_time,orout_time,opstart_time,opend_time,admission_time,discharge_time,anstart_time,anend_time,cpbon_time,cpboff_time,icuin_time,icuout_time,inhosp_death_time,allcause_death_time
484069807,178742874,229842382,,0,30,F,50,155,Asian,,1,OT,General,09B70,1110,1245,1140,1230,0,7195,1120,1235,,,,,,
446270725,158995752,257857903,,0,70,M,45,170,Asian,,1,GS,General,0WJG0,1340,1550,1370,1540,0,70555,1345,1540,,,1550,19595,69860,106560

CSV File ./dataset/operations.csv
            op_id subject_id    hadm_id case_id   opdate age sex weight  ... anstart_time anend_time cpbon_time cpboff_time icuin_time icuout_time inhosp_death_time allcause_death_time
0       484069807  178742874  229842382                0  30   F     50  ...         1120       1235                                                                                    
1       446270725  158995752  257857903                0  70   M     45  ...         1345       1540                              1550       19595             69860              106560
2       406892271  108553242  200664328            61920  50   F     70  ...        62170      62370                                                                              718560
3       478413008  133278262  277235295                0  35   F     55  ...          215        340                                                                                    
4       468516791  116924034  299190423            17280  45   F     45  ...        17950      18070                                                                                    
...           ...        ...        ...     ...      ...  ..  ..    ...  ...          ...        ...        ...         ...        ...         ...               ...                 ...
130955  449124488  138484174  228449654          4999680  50   F     60  ...      5000390    5000570                                                                                    
130956  461252752  126772283  273139806             2880  70   F     55  ...         3355       3430                                                                                    
130957  471834474  144363433  275833861             2880  65   F     50  ...         3595       3670                                                                                    
130958  419787421  195835964  293939099            12960  85   M     75  ...        13780      13950                             13955       15120                                613440
130959  493136728  148443464  280269654             2880  70   M     80  ...         3360       3430                                                                                    

[130,960 rows x 29 columns]
Reading took 0.48 seconds


CSV File ./dataset/labs.csv
         subject_id chart_time      item_name  value
0         133338290      86155  total_protein    6.9
1         133338290      86155         sodium  140.0
2         133338290      86155      potassium    4.4
3         133338290      86155       platelet  152.0
4         133338290      93150        glucose  120.0
...             ...        ...            ...    ...
19503330  155658270      35270             hb   15.0
19503331  155658270      35270            hct   44.1
19503332  155658270      35270       platelet  245.0
19503333  155658270      35270            seg   79.9
19503334  155658270      35270     lymphocyte   11.8

[19,503,335 rows x 4 columns]
Reading took 20.13 seconds



CSV File ./dataset/medications.csv
        subject_id chart_time       drug_name route drug_name2 drug_name3 atc_code atc_code2 atc_code3
0        117512122    2832985      pregabalin    po                        N03AX16                    
1        117512122    2833610      pregabalin    po                        N03AX16                    
2        117512122    2832985   levetiracetam    po                        N03AX14                    
3        117512122    2833610   levetiracetam    po                        N03AX14                    
4        117512122    2832985      famotidine    po                        A02BA03                    
...            ...        ...             ...   ...        ...        ...      ...       ...       ...
9885567  127026720       2070       ofloxacin    ex                        J01MA01                    
9885568  127026720       2550       ibuprofen    po   arginine             M01AE01   B05XB01          
9885569  127026720       1340  levocetirizine    po                        R06AE09                    
9885570  154044684        675      gentamicin    iv                        J01GB03                    
9885571  154044684        675       cefazolin    iv                        J01DB04                    

[9,885,572 rows x 9 columns]
Reading took 14.69 seconds



CSV File ./dataset/diagnosis.csv
        subject_id chart_time icd10_cm
0        190852492     325440      R06
1        190852492     325440      G20
2        142367193          0      I61
3        178346414      80640      A16
4        178346414      80640      J98
...            ...        ...      ...
2464615  195448932     172800      H02
2464616  195448932     172800      L90
2464617  195448932     172800      S04
2464618  155658270      24480      K81
2464619  155658270      34560      K81

[2,464,620 rows x 3 columns]
Reading took 2.13 seconds




CSV File ./dataset/vitals.csv
              op_id subject_id chart_time item_name  value
0         435959808  181409183       1985    minvol    4.4
1         435959808  181409183       1985        vt  512.0
2         435959808  181409183       1985        rr    9.0
3         435959808  181409183       1985       pip   23.0
4         435959808  181409183       2005    minvol    4.4
...             ...        ...        ...       ...    ...
66127935  447098707  159399111    1511530  nibp_dbp   92.0
66127936  447098707  159399111    1511530  nibp_mbp  119.0
66127937  447098707  159399111    1511530      spo2  100.0
66127938  447098707  159399111    1511535        hr   78.0
66127939  447098707  159399111    1511535      spo2  100.0

[66,127,940 rows x 5 columns]
Reading took 78.56 seconds



CSV File ./dataset/ward_vitals.csv
         subject_id chart_time item_name  value
0         104192463        580      spo2   98.0
1         104192463        580  nibp_sbp  169.0
2         104192463        580        hr   96.0
3         104192463        580  nibp_dbp  100.0
4         104192463        580        bt   36.4
...             ...        ...       ...    ...
45796479  155523991       6465      iabp    1.0
45796480  155523991       6480      iabp    1.0
45796481  155523991       6540      iabp    1.0
45796482  155523991       6570      iabp    1.0
45796483  155523991       6720      iabp    1.0

[45,796,484 rows x 4 columns]
Reading took 47.25 seconds




CSV File ./dataset/parameters.csv
           Table     Label  Unit                            Description
0           labs   albumin  g/dL                                Albumin
1           labs       alp  IU/L                   Alkaline phosphatase
2           labs       alt  IU/L                   Alanine transaminase
3           labs      aptt   sec  Activated partial thromboplastin time
4           labs       ast  IU/L                 Aspartate transaminase
..           ...       ...   ...                                    ...
121  ward_vitals  nibp_sbp  mmHg   Non-invasive systolic blood pressure
122  ward_vitals        rr  /min                       Respiration rate
123  ward_vitals      spo2     %           Peripheral oxygen saturation
124  ward_vitals        uo    mL                           Urine output
125  ward_vitals      vent   0/1          Use of mechanical ventilation

[126 rows x 4 columns]
Reading took 0.00 seconds


CSV File ./dataset/department.csv
   Abbreviations                Full name
0             AN           Anesthesiology
1            CTS  Cardio-Thoracic Surgery
2             DM              Dermatology
3             EM       Emergency Medicine
4             GS          General Surgery
5             IM        Internal Medicine
6             NS             Neurosurgery
7             OG  Obstetrics & Gynecology
8             OL          Oto-laryngology
9             OS       Orthopedic Surgery
10            OT            Ophthalmology
11           PED               Pediatrics
12            PS          Plastic Surgery
13           RAD                Radiology
14            RO       Radiation Oncology
15            UR                  Urology
Reading took 0.00 seconds

Number of subjects per department
DEPARTMENT AN = (   68, 29)
DEPARTMENT CTS= ( 8754, 29)
DEPARTMENT DM = (    1, 29)
DEPARTMENT EM = (    2, 29)
DEPARTMENT GS = (34725, 29)
DEPARTMENT IM = (   89, 29)
DEPARTMENT NS = (10172, 29)
DEPARTMENT OG = (12946, 29)
DEPARTMENT OL = (11702, 29)
DEPARTMENT OS = (17433, 29)
DEPARTMENT OT = (17249, 29)
DEPARTMENT PED= (   38, 29)
DEPARTMENT PS = ( 5157, 29)
DEPARTMENT RAD= (  379, 29)
DEPARTMENT RO = (   15, 29)
DEPARTMENT UR = (12230, 29)


"""