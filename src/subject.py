# -----------------------------------------------------------------------
# subject.py
# -----------------------------------------------------------------------
import json
import os

from sortedcontainers import SortedDict
from collections import Counter


class Subject:

    # Construct subject, must have an id.
    def __init__(self, subject_id=None):
        self._subject_id = subject_id
        self._labs        = list() # records in labs.csv for this subject id
        self._vitals      = list()
        self._operations  = list()
        self._ward_vitals = list()
        self._diagnoses   = list()
        self._medications = list()


    def get_subject_id(self):
        return self._subject_id

    def add_lab(self, lab):
        self._labs.append(lab)

    def get_labs(self):
        return self._labs

    def add_vital(self, vital):
        self._vitals.append(vital)

    def get_vitals(self):
        return self._vitals

    def add_operation(self, operation):
        self._operations.append(operation)

    def get_operations(self):
        return self._operations

    def add_ward_vital(self, ward_vital):
        self._ward_vitals.append(ward_vital)

    def get_ward_vitals(self):
        return self._ward_vitals

    def add_diagnosis(self, diagnosis):
        self._diagnoses.append(diagnosis)

    def get_diagnoses(self):
        return self._diagnoses

    def add_medication(self, medication):
        self._medications.append(medication)

    def get_medications(self):
        return self._medications


    def get_first_operation(self):
        """
        Return subject's first operation (based on orin_time)
        :return:
        """
        # operations = self.get_operations()
        # min_operation = None
        # for operation in operations:
        #     if min_operation is None:
        #         min_operation = operation
        #     else:
        #         if int(operation['orin_time']) < int(min_operation['orin_time']):
        #             min_operation = operation
        # return min_operation
        d = self.get_operations_by_orin_time()
        min_key = min(d.keys())
        return d[min_key]

    def get_last_operation(self):
        d = self.get_operations_by_orin_time()
        max_key = max(d.keys())
        return d[max_key]

    def get_age_last_operation(self):
        operation = self.get_last_operation()
        age = int(operation["age"])
        return age

    def get_operations_by_orin_time(self):
        """
        Return operations as a dict with operation orin_time as the key.
        Note that orin_time is converted to int first, so returned dict key is an int.
        :return: dict int(orin_time) -> operation (which is also a dict)
        """
        operations = self.get_operations()
        ops_orin_time = dict()
        for operation in operations:
            orin_time = int(operation['orin_time'])
            ops_orin_time[orin_time] = operation
        return ops_orin_time



    def get_inhosp_death_time(self):
        """
        If it exists, returns inhosp_death_time (associated with first operation)
        :return: None or int
        """
        # Why is checking the first operation sufficient?  Because if there is a record
        # that the subject died, all operations have the same time.  Example subject with four operations.
        # Note that this subject died but not within 30 days of last operation.
        # Still not clear what allcause_death_time is but seems to be after inhosp_death_time.
        # 100241853 30 days? False operations 4, operation 1 at orout_time=4220  inhosp_death_time=2374000  allcause_death_time=2377440
        # 100241853 30 days? False operations 4, operation 2 at orout_time=19820  inhosp_death_time=2374000  allcause_death_time=2377440
        # 100241853 30 days? False operations 4, operation 3 at orout_time=169515  inhosp_death_time=2374000  allcause_death_time=2377440
        # 100241853 30 days? False operations 4, operation 4 at orout_time=2242950  inhosp_death_time=2374000  allcause_death_time=2377440
        first_operation = self.get_first_operation()
        # Not all operations will have a inhosp_death_time (obviously many patients survive)
        inhosp_death_time = first_operation['inhosp_death_time'].strip()
        if len(inhosp_death_time) == 0:
            return None
        return int(inhosp_death_time)


    def inhosp_death_30day(self):
        """
        Determines if the subject died within 30 days of their last operation.
        :return: True if inhosp_death_time < orout_time (last operation),
                 False otherwise, including if subject survived.
        """
        # I think the gbm_mortality takes the first (smallest chart_time)
        #  operation and uses it?  See get_inhosp_death_time for partial explanation.

        # find the first operation for each patient
        # df.sort_values('orin_time', inplace=True)
        # df = df.loc[df[['op_id', 'subject_id']].groupby('subject_id')['op_id'].idxmin()]
        # df[OUTCOME_VAR] = (df['inhosp_death_time'] < df['orout_time'] + 30 * 24 * 60)
        # inhosp_death_30day
        # TYPE 'inhosp_death_time' float64
        # TYPE 'orout_time' int64
        # first_operation = self.get_first_operation()

        # min_operation always has inhosp_death_time?
        inhosp_death_time = self.get_inhosp_death_time()
        if inhosp_death_time is None:
            return False

        # This is how gbm_mortality computed, but I think this is wrong.
        # All operations have the same (repeated) inhosp_death_time, can just use first operation.
        # But what to compare against? It is orout_time but which operation?
        # Should it not be 30 days from the last operation not the first operation?
        #  Could be that subsequent operations are seen as causal consequence of initial operation?
        #  This still seems wrong, many subjects have multiple operations over 10 years.
        # operation = self.get_first_operation()
        operation = self.get_last_operation()
        orout_time = operation['orout_time'].strip()
        orout_time = int(orout_time)

        # subject may have died, but did they did within 30 days of the operation?
        died = (inhosp_death_time < orout_time + 30 * 24 * 60)
        return died



    def died(self):
        operations = self.get_operations()
        for operation in operations:
            inhosp_death_time = operation['inhosp_death_time'].strip()
            allcause_death_time = operation['allcause_death_time'].strip()
            # if len(inhosp_death_time)>0 or len(allcause_death_time)>0:
            if len(inhosp_death_time) > 0:
                #death_count += 1
                #print(f"SUBJECT {subject_id}: OPERATION {operation}")
                #for op_key in operation:
                #    print(f"    {op_key} -> {operation[op_key]}")
                #break
                return True
        return False


    def get_most_recent_lab(self, lab_name, chart_time):
        """
        Returns value of most recent lab from given chart_time
        :param chart_time:
        :param lab_name:
        :return: float value or None if no lab found before chart_time
        """
        #             subject_id  chart_time      item_name  value
        # 0          133338290       86155  total_protein    6.9
        # 1          133338290       86155         sodium  140.0
        # 2          133338290       86155      potassium    4.4
        # dict(sorted(my_dict.items()))
        # Sort all the labs by chart_time
        sorted_labs = self.get_lab(lab_name)

        # First, see if we have an exact value at that time
        if chart_time in sorted_labs:
            return sorted_labs[chart_time]
        # If not, look for closest preceding value (might be None)
        # print(f"{self._subject_id}: {lab_name} -> {sortedLabs} =   (t={chart_time})")
        preceding_key = find_nearest_preceding(sorted_labs, chart_time)
        if preceding_key is None:
            return None
        preceding_val = sorted_labs[preceding_key]
        # print(f"{self._subject_id}: {lab_name} -> {sorted_labs} = {preceding_key}: {preceding_val}  (t={chart_time})")
        return preceding_val

    def get_lab(self, lab_name):
        """
        Returns value of most recent lab from given chart_time
        :param lab_name:
        :return: float value or None if no lab found before chart_time
        """
        #             subject_id  chart_time      item_name  value
        # 0          133338290       86155  total_protein    6.9
        # 1          133338290       86155         sodium  140.0
        # 2          133338290       86155      potassium    4.4
        # dict(sorted(my_dict.items()))
        sorted_labs = SortedDict()
        for lab in self._labs:
            chart_time = int( lab['chart_time'].strip() )
            item_name  = lab['item_name']
            if item_name == lab_name:
                value = float(lab['value'].strip())
                # Oddly some have two values for same lab at same chart_time
                if chart_time in sorted_labs:
                    # prev_val = sorted_labs[chart_time]
                    # if prev_val != value:
                    #     print(f"Anomaly {self._subject_id}: {lab_name} at {chart_time} already has value {prev_val} overwriting with {value}")
                    # Dodgy solution?  Treat as collision and addd to available previous position
                    adjusted_chart_time = chart_time
                    while adjusted_chart_time in sorted_labs:
                        adjusted_chart_time = adjusted_chart_time - 1
                    # print(f"Anomaly {self._subject_id}: {lab_name} at {chart_time} already has value {prev_val}, adding value {value} at {adjusted_chart_time}")
                    sorted_labs[adjusted_chart_time] = value
                else:
                    sorted_labs[chart_time] = value
        return sorted_labs

    def get_chart_time_range(self):
        """
        Gets the minimum chart_time of any information for this subject.
        :return: (max_chart_time, max_chart_time)
        """
        min_chart_time = None
        max_chart_time = None
        # We always have at least one operation, so only need to check None for operations
        for operation in self.get_operations():
            chart_time = int( operation['orin_time'] )
            if min_chart_time is None or chart_time < min_chart_time:
                min_chart_time = chart_time
            if max_chart_time is None or chart_time > max_chart_time:
                max_chart_time = chart_time
        for medication in self.get_medications():
            chart_time = int( medication['chart_time'] )
            if chart_time < min_chart_time: min_chart_time = chart_time
            if chart_time > max_chart_time: max_chart_time = chart_time
        for vital in self.get_vitals():
            chart_time = int( vital['chart_time'] )
            if chart_time < min_chart_time: min_chart_time = chart_time
            if chart_time > max_chart_time: max_chart_time = chart_time
        for ward_vital in self.get_ward_vitals():
            chart_time = int( ward_vital['chart_time'] )
            if chart_time < min_chart_time: min_chart_time = chart_time
            if chart_time > max_chart_time: max_chart_time = chart_time
        return min_chart_time, max_chart_time






    def toJSON(self, filepath):
        # Add all lists to a dict
        data = dict()
        data["subject_id"]  = self._subject_id
        data["labs"]        = self._labs
        data["vitals"]      = self._vitals
        data["operations"]  = self._operations
        data["ward_vitals"] = self._ward_vitals
        data["diagnoses"]   = self._diagnoses
        data["medications"] = self._medications

        # Writing to a JSON file
        with open(filepath, "w") as file:
            json.dump(data, file, indent=2)

    def fromJSON(self, filepath):
        # Reading from a JSON file
        with open(filepath, "r") as json_file:
            data = json.load(json_file)
        # json_file = open(filepath, "r")
        # data = json.load(json_file)
        # json_file.close()

        self._subject_id  = data["subject_id"]
        self._labs        = data["labs"]
        self._vitals      = data["vitals"]
        self._operations  = data["operations"]
        self._ward_vitals = data["ward_vitals"]
        self._diagnoses   = data["diagnoses"]
        self._medications = data["medications"]



    # Return the elapsed time since creation of self, in seconds.
    def __str__(self):
        return f"{self._subject_id}: LABS {len(self._labs)}, OPS {len(self._operations)}, VITALS {len(self._vitals)}, WARD {len(self._ward_vitals)}, DIAGNOSES {len(self._diagnoses)}, MEDICATIONS {len(self._medications)}"




"""
A common and feasible approach in perioperative and critical care datasets
like INSPIRE is to use the Hospital Frailty Risk Score (HFRS), an
ICD-10-based administrative scoring system validated for hospitalised
patients (including surgical and ICU cohorts). It is fully computable
from the diagnosis table in INSPIRE and correlates well with clinical frailty measures.

Reference for the frailty score
Gilbert T, Neuburger J, Kraindler J, et al. Development and validation of a Hospital
Frailty Risk Score focusing on older people in acute care settings using
electronic hospital records: an observational study.
Lancet. 2018;391(10132):1775-1782. doi:10.1016/S0140-6736(18)30668-8.

Interpretation:
Low risk: <5
Intermediate risk: 5–15
High risk: >15

The score has been externally validated in critically ill patients and predicts
longer stays, readmissions, and mortality.

:param subject:
:return:
"""

import frailty_hfrs

def compute_hfrs(subject):
    """
    Compute the Hospital Frailty Risk Score (HFRS) for a subject.  This is derived from
    the diagnosis codes, where each code contributes "points" to the frailty risk score.
    The points are simply added together from the last two years of diagnosis codes.
    Many codes will not have a point indicating it contributes 0.0 to the score.

    Parameters:
        subject: Object with attribute .diagnoses (List[str]) containing ICD-10 codes.
                 May include demographics if age weighting is desired (optional).

    Returns:
        frailty_score (float): HFRS value.
        risk_category (str): Low/Intermediate/High.
    """
    diagnoses = subject.get_diagnoses()
    if len(diagnoses) == 0:
        return 0.0, "unknown (no diagnoses)"

    # NEED TO FIGURE OUR TWO YEARS OF CODES
    # Normalise codes: remove dots and uppercase
    # print(f"BEFORE: {diagnoses}")
    clean_diagnoses = list()
    for diagnoses_dict in diagnoses:
        code = diagnoses_dict["icd10_cm"]
        clean_code = code.replace('.', '').upper()
        clean_diagnoses.append( clean_code )
    # print(f"AFTER: {clean_diagnoses}")

    # HFRS weighting (subset of the 109 ICD-10 clusters; full list in Gilbert et al., 2018)
    # Values are the points assigned to each cluster
    hfrs_weights = frailty_hfrs.get_hfrs_weights()

    score = 0.0
    for code in clean_diagnoses:
        # Match the first 3 characters (cluster level)
        cluster = code[:3]
        if cluster in hfrs_weights:
            code_info = hfrs_weights[cluster]
            score += code_info["Points awarded"]
        # Also check full code for exact matches if needed

    # Optional: age adjustment (not part of original HFRS but sometimes used)
    # if hasattr(subject, 'age') and subject.age >= 75:
    #     score += 2.0  # example adjustment

    # Risk category
    if score < 5:
        category = "low"
    elif score < 15:
        category = "intermediate"
    else:
        category = "high"

    return score, category



def convert_vitals_to_dictionary(vitals):
    """
    Converts list of vitals, where each vital is a str list
    ['op_id','subject_id','chart_time','item_name','value'],
    to a dict mapping chart_time to value for given 'item_name' (aka vital_name).
    :param vitals:
    :return: series[chart_time] -> value
    """
    vitals_dict = dict()
    for subject_vital in vitals:
        # op_id      = subject_vital['op_id']
        # subject_id = subject_vital['subject_id']
        chart_time = int(subject_vital['chart_time'])
        item_name = subject_vital['item_name']
        value = float(subject_vital['value'])

        if item_name not in vitals_dict:
            vitals_dict[item_name] = dict()
        series = vitals_dict[item_name]
        series[chart_time] = value
    return vitals_dict



def find_nearest_preceding(sorted_dict, lookup_key):
    """
    Finds the key that precedes given lookup_key or None if no preceding key exists
    :param sorted_dict:
    :param lookup_key:
    :return: key that precedes lookup_key or None if no preceding key exists
    """
    # hb -> SortedDict({-2175: 12.9, 6300: 11.3, 9100: 11.3, 11990: 10.7, 29365: 13.7}) = -2175: 12.9  (lookup_key=3500)
    # Find the index where the lookup key would be inserted to maintain order
    index = sorted_dict.bisect_left(lookup_key)
    # If the index is 0, there is no preceding key
    if index == 0:
        # raise KeyError(f"No key preceding {lookup_key} found")
        return None
    # Return the key at the index before the insertion point
    return sorted_dict.iloc[index - 1]
    # sd = SortedDict({0: '0', 4: '4', 8: '8', 12: '12'})
    # print(find_nearest_preceding(sd, 4))  # Output: 0
    # print(find_nearest_preceding(sd, 3))  # Output: 0
    # print(find_nearest_preceding(sd, 12))  # Output: 8


def save_counter_to_json(counter, filename):
    """
    Save a Counter object to a JSON file.
    Args:
        counter (Counter): The Counter object to save.
        filename (str): The path to the output JSON file.
    """
    # Convert Counter to a regular dictionary for JSON serialization
    counter_dict = dict(counter)
    with open(filename, 'w') as f:
        json.dump(counter_dict, f, indent=4)

def subjects_statistics(subjects):
    labs = Counter()
    meds = Counter()
    vitals = Counter()
    wards = Counter()
    for subject_id in subjects.keys():
        subject = subjects[subject_id]

        for lab_dict in subject.get_labs():
            # subject_id  chart_time  item_name  value
            lab_name = lab_dict['item_name']
            labs[lab_name] += 1
        for med_dict in subject.get_medications():
            # subject_id chart_time drug_name route drug_name2 drug_name3 atc_code atc_code2 atc_code3
            med_name = med_dict['drug_name']
            meds[med_name] += 1
        for vit_dict in subject.get_vitals():
            # op_id subject_id chart_time item_name value
            vit_name = vit_dict['item_name']
            vitals[vit_name] += 1
        for ward_dict in subject.get_ward_vitals():
            # subject_id chart_time item_name  value
            ward_name = ward_dict['item_name']
            wards[ward_name] += 1
    # print(f"LABS: {labs}")
    # print(f"MEDS: {meds}")
    # print(f"VITS: {vitals}")
    # print(f"WARD: {wards}")
    save_counter_to_json(labs, "counter_labs.json")
    save_counter_to_json(meds, "counter_medications.json")
    save_counter_to_json(vitals, "counter_vitals.json")
    save_counter_to_json(wards, "counter_ward_vitals.json")

    print(f"LABS: {len(labs)}")
    print(f"MEDS: {len(meds)}")
    print(f"VITS: {len(vitals)}")
    print(f"WARD: {len(wards)}")

    """
    labs = ['glucose', 'creatinine', 'hct', 'potassium', 'sodium', 'hb']
    meds = ['ambroxol', 'famotidine', 'tramadol', 'magnesium oxide']
    vitals = ['rr', 'hr', 'spo2', 'etco2', 'fio2']
    ward_vitals = ['hr', 'nibp_sbp', 'nibp_dbp', 'rr']

    LABS: 38 (pre, peri and post operative)
    MEDS: 1143
    VITS: 74 (peri operative)
    WARD: 16 (pre and post operative)
    """



def read_subjects(parent_dir='../inspire_subjects'):
    """
    Reads all json files in given parent_dir, converts to subjects, and returns dict
    mapping subject_id to subject.
    :param parent_dir:
    :return: dict mapping subject_id to subject
    """
    subjects = dict()
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith('.json'):
                json_filepath = os.path.join(root, file)
                subject = Subject()
                subject.fromJSON(json_filepath)
                subject_id = subject.get_subject_id()
                subjects[subject_id] = subject
    return subjects








"""
hfrs_weights
{
    ...
    "G81": {
        "ICD Description": "Hemiplegia",
        "Number with code in devt cohort": 332,
        "devt percentage": 1.5,
        "Number with code in frail cohort": 240,
        "frail percentage": 4.9,
        "Points awarded": 4.4,
    }
}



Writing subject that died (12 operations) to file ../inspire_subjects/147283461.json 30 day death? False
Writing subject that died (10 operations) to file ../inspire_subjects/170236654.json 30 day death? False
Writing subject that died (10 operations) to file ../inspire_subjects/126716921.json 30 day death? False
Writing subject that died (9 operations) to file ../inspire_subjects/157240153.json 30 day death? False
Writing subject that died (8 operations) to file ../inspire_subjects/128054712.json 30 day death? False
Writing subject that died (7 operations) to file ../inspire_subjects/180662611.json 30 day death? False
Writing subject that died (7 operations) to file ../inspire_subjects/119372314.json 30 day death? False
Writing subject that died (7 operations) to file ../inspire_subjects/181462224.json 30 day death? False
Writing subject that died (7 operations) to file ../inspire_subjects/146639724.json 30 day death? False
Writing subject that died (7 operations) to file ../inspire_subjects/139997521.json 30 day death? False
Writing subject that died (6 operations) to file ../inspire_subjects/121861214.json 30 day death? False
Writing subject that died (6 operations) to file ../inspire_subjects/137233090.json 30 day death? False
Writing subject that died (6 operations) to file ../inspire_subjects/129082821.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/154148980.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/138721930.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/197368441.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/131283541.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/184024093.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/174363754.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/142942391.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/184263904.json 30 day death? True
Writing subject that died (5 operations) to file ../inspire_subjects/188639463.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/156429262.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/166480854.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/103375800.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/112264134.json 30 day death? False
Writing subject that died (5 operations) to file ../inspire_subjects/186777610.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/174838434.json 30 day death? True
Writing subject that died (4 operations) to file ../inspire_subjects/174006091.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/100407302.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/188748684.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/124751341.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/179803680.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/151484442.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/123437631.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/126590050.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/164719124.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/183017100.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/197166820.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/195951291.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/128651374.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/108010923.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/148259660.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/198410962.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/107126152.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/119619414.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/114704224.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/140653774.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/151595952.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/180785233.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/191862702.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/158624643.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/118309453.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/198462013.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/142074411.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/198516024.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/195071111.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/182630143.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/122736581.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/135909130.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/132033764.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/154413694.json 30 day death? True
Writing subject that died (4 operations) to file ../inspire_subjects/100241853.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/195522551.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/118082100.json 30 day death? False
Writing subject that died (4 operations) to file ../inspire_subjects/135313864.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/195788311.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/130543753.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/135169614.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/187620284.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/157233731.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/138603594.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/193105404.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/162363861.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/139768910.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/143645342.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/175630903.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/179029520.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/108010173.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/173038842.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/131460383.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/171897072.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/117744960.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/158995752.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/174570851.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/100833884.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/142456320.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/168066671.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/125803363.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/198139004.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/175089051.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/154678562.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/192384851.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/161763390.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/146887202.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/142340863.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/139144302.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/178563320.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/131735091.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/180449802.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/158483092.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/106850193.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/130815391.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/178239503.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/182485442.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/151337541.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/154709362.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/134503784.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/151875594.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/104525162.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/133030712.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/129054330.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/198956381.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/127897061.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/155773964.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/193578673.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/138535324.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/161924664.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/156675284.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/147833713.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/152571072.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/102005764.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/161826782.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/102124711.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/112159741.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/113177201.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/127062362.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/110055084.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/180835224.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/184737891.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/121166833.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/129496940.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/136359100.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/128673840.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/196651933.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/152301993.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/171493690.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/121973832.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/173485852.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/184773941.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/133859210.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/162205813.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/139134334.json 30 day death? True
Writing subject that died (3 operations) to file ../inspire_subjects/127354791.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/185885103.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/119729283.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/136833612.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/183015073.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/199422774.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/109120933.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/127923452.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/124767854.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/110641643.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/189409342.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/121015201.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/189413412.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/154572360.json 30 day death? False
Writing subject that died (3 operations) to file ../inspire_subjects/155667223.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/126736540.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/100221250.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/157206391.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/103722581.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/182029000.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/183481893.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/142623154.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/191166533.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/115539784.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/112204443.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/195830681.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/184686510.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/165709394.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/114166470.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/139111100.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/185970493.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/173360424.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/102096263.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/103982082.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/141023343.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/130169561.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/194593941.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/183645944.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/177334484.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/138953691.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/153654431.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/104745112.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/161138504.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/125667832.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/169472943.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/101406342.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/143582003.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/170978103.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/161607861.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/100882784.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/192214321.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/156450753.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/128084773.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/164314083.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/179403184.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/110127394.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/119536181.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/147742743.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/121834363.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/191662204.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/169343841.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/141498850.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/194459203.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/167539902.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/180827350.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/170488890.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/184196284.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/116616074.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/185110952.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/180587364.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/119796054.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/185561114.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/177232891.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/141061171.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/186632342.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/115831223.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/163557500.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/100316372.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/183170184.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/180239672.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/195939720.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/178888301.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/106953503.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/175280800.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/121993860.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/124566243.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/150468101.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/129573541.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/193769543.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/158571702.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/138084724.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/138093823.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/103363991.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/149827632.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/115564652.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/132339014.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/111568893.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/118428891.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/194259634.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/169468011.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/121373611.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/143636431.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/193393963.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/184427791.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/161409462.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/150706771.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/124498721.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/165294454.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/179045763.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/126187461.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/145336552.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/180580464.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/146704640.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/192018344.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/121228143.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/181124311.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/103224333.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/199723140.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/175353044.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/175944993.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/197722933.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/135568271.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/162427900.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/105036353.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/158467801.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/188599794.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/159848950.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/119827674.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/179479044.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/170004994.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/198361073.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/193148823.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/191201210.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/153951842.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/135905692.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/118480291.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/143288730.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/169840802.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/180391840.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/152207393.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/154875200.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/175817512.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/192149873.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/149514272.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/173455490.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/159333113.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/172650904.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/101862780.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/130263730.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/191870832.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/110544311.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/183000301.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/114439334.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/149047291.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/141752032.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/143013854.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/170331533.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/150143024.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/150049403.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/104916164.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/147986980.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/140336061.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/184893960.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/111306462.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/117070282.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/135327530.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/134050671.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/104495240.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/153951733.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/123765383.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/128951822.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/117952743.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/186767962.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/100751234.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/177090052.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/180977032.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/114058770.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/101130643.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/117894154.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/124017903.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/127340122.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/174457311.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/110917131.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/153160913.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/195567462.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/174948621.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/148337600.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/106388272.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/145010813.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/114204994.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/102555984.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/199138581.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/180396111.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/167741923.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/121835024.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/194112061.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/187947611.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/144832031.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/114929180.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/158221514.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/106913003.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/101355512.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/109408683.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/193437012.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/185151622.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/176118703.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/130516981.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/191746421.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/121142293.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/146520901.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/102353443.json 30 day death? False
Writing subject that died (2 operations) to file ../inspire_subjects/130425921.json 30 day death? True
Writing subject that died (2 operations) to file ../inspire_subjects/152981350.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/148192562.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/157260862.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/140696052.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/160969360.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/125284952.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/132724110.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/116450812.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/184731263.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109840233.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/137411141.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/102010872.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/171169472.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/193607230.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/197582802.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/189637001.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/106166111.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/174157273.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114906004.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/191791400.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/151244123.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/179875494.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/101139210.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/156559300.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/106645651.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/100033460.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/130465433.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/176782180.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/149022921.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/105964160.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/123915112.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/131108640.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/159859750.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/188913783.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/188118164.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/154312590.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/186185863.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/177922874.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/164390820.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/199475851.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/166838522.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/136159821.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/149958842.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/123116424.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/137634074.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/155580654.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/148708611.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/183524970.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/138539222.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/103782822.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/123309681.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/162172983.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/181532301.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/194138171.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/103984531.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/154064690.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/177251631.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/149173432.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/169721574.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/159992424.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/106614853.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/150341522.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/141957674.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/165194910.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/184671031.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/123749674.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/146866270.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/154213294.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/139217913.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/193251433.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/152725573.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/168030040.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109666910.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/143769274.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/173185513.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/136257534.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/125440581.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114782353.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/171729982.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/133776461.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/111186563.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/173233461.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/180057361.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/159299630.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/109567852.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/154699923.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/101331133.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/125583120.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/197948524.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/145876941.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/135932274.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/177451021.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/130457311.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/147164753.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/148157981.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/122108412.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/119101831.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/100573111.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114789163.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/122397362.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/166967783.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/112426824.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/152384790.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/115157880.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/111855722.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/173975144.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/152543172.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/160337051.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/149564794.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/180030171.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165484400.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/186564652.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/103737712.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/160148170.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/149993021.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/128018183.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/189195121.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/194148912.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/156167282.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/177129030.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/109275210.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/103648601.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/163976533.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/183421603.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/122861862.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/117794082.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/151825511.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/113823093.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109843892.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/124039502.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/118143733.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/149625052.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/111200410.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/136986702.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/147802321.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/121169681.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/101499142.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/159601874.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/130263000.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/101265912.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/138538862.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/190321643.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114339230.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/177467454.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/116130480.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/135568561.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/129975464.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/113294044.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/173871792.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114651690.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/172406810.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/124324671.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/178231151.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/189702022.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/189952354.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/123739924.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114410312.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/135230292.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/133439550.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/108421744.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/149066902.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109028210.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/114675602.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/166537550.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165528944.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/197996011.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/155576900.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/133947983.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/123710330.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/153741890.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/169482231.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/166899974.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/139667653.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/169748220.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/159898430.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/177322093.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/168223484.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/135097151.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/179939433.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/116901212.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/133185483.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/124360921.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/193303131.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/137766242.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/140438624.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/189458114.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/134657452.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/166147584.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/190143952.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/132605490.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/155464523.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/189302043.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/162476131.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/145996664.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/177493610.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/185780962.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/135081733.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/125627403.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/119401934.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/195875161.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/100301573.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/100516694.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/178989553.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/192963772.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/105656884.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/121172544.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/163402362.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/127196210.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/119282711.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/143948431.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/186918382.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/187747423.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/125712841.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/112047260.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/172093031.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/161303760.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/152401113.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/130181784.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/192757102.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114300073.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/104059484.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/181208121.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/146225093.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/116535054.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/138204913.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/176727101.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/154339810.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/153909940.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/148315094.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/115168760.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/116680464.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/172321191.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/159005134.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/120640203.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/127717702.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/176876594.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/104023053.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/194129164.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/145114881.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/197443970.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/112396860.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/123638264.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/141142470.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/130297691.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/150636270.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/186875624.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/160837504.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/115790932.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/121618654.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/197658720.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/164997641.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/159300733.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/113930881.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/130978792.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/124739800.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/189608271.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/168600920.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109385940.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/132824982.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/191670881.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/131699924.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/193217662.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/137453022.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/133984980.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/187709471.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/183197771.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/115567311.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/164313364.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/117551171.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/192610231.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/181885353.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/193794171.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/156338724.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/193314854.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/129058882.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/150392182.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/153307063.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/139543801.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/146460682.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/106569512.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/121574794.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/199503464.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/110070681.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/138257363.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/145239503.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/115262392.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/127765273.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/152603393.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/154752614.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/147876641.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/143826572.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/153529824.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/133369322.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/189768083.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/123134214.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/181929463.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/140074093.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/123260724.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/143558531.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/193144970.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/171040791.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/153372504.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/187914262.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/150220680.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/147889850.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/192616771.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/199490004.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/169989763.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/161501392.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/150499953.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/148382232.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/159937472.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/176392252.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/185531921.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/183544012.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/181372413.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/116166152.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/165758704.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/111282224.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/178060343.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/115131390.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/187501154.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/156288544.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/105582611.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/133221674.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/105922792.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/153098861.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/190398044.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/178146914.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/144886644.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/191521871.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/189968754.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/110181713.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/134448770.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/185361530.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/185903271.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/174773781.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165887534.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/124379720.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/150119241.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/103535981.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/169736540.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/195373010.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/154945490.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/160001612.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/194863512.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/149460872.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/148138721.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/124807434.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/157562100.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/120782344.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/169666464.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/126222314.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/128573521.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/172779471.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/160124973.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/142399852.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/175016423.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/141555230.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/137212163.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/109453583.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/115775962.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/135963053.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/163772282.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/146771482.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/176465664.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/149650122.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/110439082.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/111789014.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/160028353.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114455110.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/161936670.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/123684223.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/128175513.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/164321822.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/171447920.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/129953313.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/119846891.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/138258501.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/185311201.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/152582672.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/144661952.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/174050582.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/192881321.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/122385523.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/151497601.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/151446263.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/140180611.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/178146223.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/192742400.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/140972702.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/125252951.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/146417052.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/102509492.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/185489234.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/198033184.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/133850600.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/166823860.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/172870983.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/102470232.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/196608222.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/115495272.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/116967580.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/156420820.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/108476524.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/149247872.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/164449160.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/113517673.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/140591662.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/191394294.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/150470464.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/125980243.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/107146004.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/156648992.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/154109814.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/156225880.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/135445484.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/148346660.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/181813472.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/176392760.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/101505643.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/141912432.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/113765081.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/128528441.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/107328754.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/136158732.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/156568172.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/100528073.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/142544603.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/169534220.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/171918424.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/185901203.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109844304.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/144819641.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/119064330.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/150334533.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/170434604.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/126124080.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/127952870.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/184278042.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/114852294.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/128874171.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/146647553.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/107009653.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/154545464.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/144971651.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/110051661.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/149016811.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/142106420.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/178156861.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/159901424.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/105305690.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165909622.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/180816174.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165199001.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/183775083.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/171208962.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/120122183.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109153483.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/121599654.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/161012983.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/161232822.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/147557821.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/194168421.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/105801541.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/133624853.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/108187162.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/152899671.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/157018291.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/167514601.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/182279204.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/106907581.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/128048230.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/178963454.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/120826082.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/143386441.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/198840131.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/142378672.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/103613800.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165295972.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/109347970.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/164841104.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/153474464.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/150920191.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/188454401.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/145712542.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/162270433.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/138981233.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/148044730.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/122412123.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/140742430.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/116829170.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/166268923.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/181431310.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/169655584.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/169550314.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/117593153.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/188945333.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/188140603.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/199254512.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/193917553.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/182639372.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/124694314.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/119841912.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/173066201.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/127556043.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/135159634.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/145810762.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/182666864.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/164849181.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/168809351.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/117093960.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/183381534.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165837880.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/192491562.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/105456170.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/159133110.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/101645034.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/175698821.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/135497683.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/187781911.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/139198180.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/190381392.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/113302853.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/166579711.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/171668914.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/198301111.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/141799982.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/125733131.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/151058124.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/137597511.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/142640380.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/116782502.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/112270830.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/176249322.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/192105520.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/132467773.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/182739030.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/150940900.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/153865400.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/148576104.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/122242783.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/175844880.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/190274801.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/143737553.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/177971952.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/154024570.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/173902890.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/138400444.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/179610550.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/167104370.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/114801860.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/199216860.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/181792792.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/150449192.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/168905030.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/123029290.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/119620921.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/111893250.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/165996362.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/127639812.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/115292494.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/175657974.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/148273930.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/133338024.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/140106882.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/175034821.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/105663650.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/197897562.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/187148553.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/128136901.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/144169520.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/191114290.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/136464830.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/107422553.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/199180794.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/110294784.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/153994294.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/136647810.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/100403813.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/150796331.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/168713154.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/112950494.json 30 day death? True
Writing subject that died (1 operations) to file ../inspire_subjects/147468780.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/137690480.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/111041300.json 30 day death? False
Writing subject that died (1 operations) to file ../inspire_subjects/165031932.json 30 day death? False
"""
