"""
Utility module to compute basics statistics
(mean, standard deviation, median, modified medican absolute deviation)
for subjects.  Saves results and intended to be used later to standardise
vitals values.
"""


import numpy as np
import pandas as pd

import random
import math
from stopwatch import Stopwatch
import subject as subject_module
import json

def compute_median(data_list):
    """
    Gets median from lst of numbers (could also use statistics or numpy).
    :param data_list: list of values (should be sortable)
    :return: float median
    """
    sorted_lst = sorted(data_list)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 1:
        # convert to float so return type is deterministic
        return float( sorted_lst[mid] )
    else:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2

def compute_mean_absolute_deviation(data_list, median=None):
    if median is None:
        median = compute_median(data_list)
    deviations = list() # list of absolute deviations
    for value in data_list:
        deviation = abs( value - median )
        deviations.append(deviation)
    mad = compute_median(deviations) # median absolute deviation
    return mad


def compute_modified_z_score(data_list, median=None, mad=None):
    """
    Computes modified z-score using median and mad (and multiplying by 0.6745 scale factor).
    These z-scores are more robust to outliers.
    :param data_list: list of values (must be numbers)
    :param median: float, if None, will be computed from data_list
    :param mad: float, median absolute deviation, if None, will be computed from data_list and median
    :return: z-score_list (same length, each value corresponding to data_list)
    """
    if median is None:
        median = compute_median(data_list)
    if mad is None:
        mad = compute_mean_absolute_deviation(data_list, median=median)
    modified_z_scores = list()

    """
    The constant 0.6745 makes the scale comparable to the standard deviation 
        under a normal distribution.
    Common threshold: M > 3.5 indicates a potential outlier.
    Robust because both median and MAD have a breakdown point of 50%.

    The scaling constant 0.6745 is chosen because it approximates the 75th percentile 
    (or third quartile) of the standard normal distribution, ensuring that for 
    outlier-free normally distributed data, the MAD is roughly equal to 0.6745 times 
    the standard deviation—making modified z-scores behave similarly to standard z-scores 
    in ideal cases.
    """
    scale = 0.6745
    for value in data_list:
        z_score = (value - median) / mad
        # Scale to be comparable with normal distributions
        modified_z_score = scale * z_score
        modified_z_scores.append(modified_z_score)

    return modified_z_scores


def compute_mean(data_list):
    return np.mean(data_list)

def compute_stdev(data_list):
    return np.std(data_list, ddof=1)  # sample standard deviation


def library_approach(data_list):
    # Example data with outliers
    data = np.array(data_list)

    # ---- Classical z-score ----
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # sample standard deviation
    z_scores = (data - mean) / std

    # ---- Robust z-score (Modified Z based on Median & MAD) ----
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad

    print(f"mean={mean} type={type(mean)}")
    print(f"std={std} type={type(std)}")
    print(f"median={median} type={type(median)}")
    print(f"mad={mad} type={type(mad)}")

    # Put results in a table
    df = pd.DataFrame({
        "Value": data,
        "Classical_Z": z_scores,
        "Modified_Z": modified_z_scores
    })

    print(f"z_scores mean {np.mean(z_scores):.4f}")
    print(f"z_scores std {np.std(z_scores, ddof=1):.4f}")

    print(f"Modified_z_scores mean {np.mean(modified_z_scores):.4f}")
    print(f"Modified_z_scores std {np.std(modified_z_scores, ddof=1):.4f}")

    print(df)

def main_test():

    data_list = [10, 12, 11, 13, 12, 11, 100]
    # data_list = [10, 12, 11, 13, 12, 11]

    # count = 10  # Number of integers to generate
    # numbers = random.sample(range(-500, 501), count)

    library_approach(data_list)

    median = compute_median(data_list)
    mad    = compute_mean_absolute_deviation(data_list)

    print(f"MY median={median} type={type(median)}")
    print(f"MY mad={mad} type={type(mad)}")

    modified_z_scores = compute_modified_z_score(data_list)
    for value, modified_z_score in zip(data_list, modified_z_scores):
        print(f"{value} -> {modified_z_score:.4f}")

def read_vital_stats(filepath='vital_stats_all.json'):
    """
    Reads vital stats and returns dict of dict.
    :param filepath:
    :return: vital_name -> {'mean':value, 'stdev':value, 'median':value, 'mad':value, 'length':value}
    """
    with open(filepath, 'r') as file:
        vital_stats = json.load(file)
    # "hr": {
    #     "mean": 75.6669105300075,
    #     "stdev": 16.314533970124334,
    #     "median": 74.0,
    #     "mad": 12.0,
    #     "length": 11536516}
    return vital_stats

def save_vital_stats(vital_stats, filepath='vital_stats_all.json'):
    """
    Saves the dict of dicts to specified json file.
    :param vital_stats:
    :param filepath:
    :return:
    """
    with open(filepath, 'w') as file:
        json.dump(vital_stats, file, indent=4)

def read_subjects_save_vital_stats(inspire_root_dir='../inspire_subjects'):
    # inspire_root_dir = "../inspire_subjects_small/"
    # inspire_root_dir = "../inspire_subjects"
    stopwatch = Stopwatch()
    subjects = subject_module.read_subjects(inspire_root_dir)
    print(f"Reading took {stopwatch.elapsedTime() / 60:.2f} minutes")
    # subject_module.subjects_statistics( subjects )
    # # LABS: 38
    # # MEDS: 1143
    # # VITS: 74     (operation vitals)
    # # WARD: 16     (ward vitals)
    # plot_chart_time_range_histogram( subjects )
    # if True:
    #     return

    # Collect the names of all vital signs
    item_names = set()
    for subject_id in subjects.keys():
        subject = subjects[subject_id]

        # Get the operation and ward vitals, these are in list of list of str format
        operation_vitals = subject.get_vitals()
        ward_vitals = subject.get_ward_vitals()
        labs = subject.get_labs()

        # Reformat so in dict of dict (chart_time -> value)
        operation_series = subject_module.convert_vitals_to_dictionary(operation_vitals)
        ward_series = subject_module.convert_vitals_to_dictionary(ward_vitals)
        lab_series = subject_module.convert_vitals_to_dictionary(labs)

        for item_name in operation_series.keys():
            item_names.add(item_name)
        for item_name in ward_series.keys():
            item_names.add(item_name)
        for item_name in lab_series.keys():
            item_names.add(item_name)

    # Now, for each vital name, get the values and compute the mean, standard deviation, median and mad.
    vitals_values = dict()
    for item_name in item_names:
        vitals_values[item_name] = list()

    for subject_id in subjects.keys():
        subject = subjects[subject_id]

        operation_vitals = subject.get_vitals()
        ward_vitals = subject.get_ward_vitals()
        labs = subject.get_labs()

        # Convert to series, this is actually dict mapping chart_time to each value
        operation_series = subject_module.convert_vitals_to_dictionary(operation_vitals)
        ward_series      = subject_module.convert_vitals_to_dictionary(ward_vitals)
        lab_series       = subject_module.convert_vitals_to_dictionary(labs)

        for item_name in item_names:
            if item_name in operation_series:
                subjects_vital_series = operation_series[item_name]
                # Convert so these are only the values, for this task we do not need the times
                subjects_vital_values = list(subjects_vital_series.values())
                # print(f"SERIES {item_name} {subjects_vital_series}")
                # print(f"VALUES {item_name} {subjects_vital_values}")
                # element = subjects_vital_values[0]
                # print(f"TYPE   {item_name} {type(element)}")
                # if not isinstance(element, float):
                #     raise TypeError(f"Expected float but got {type(element)}")

                vitals_values[item_name].extend(subjects_vital_values)
            if item_name in ward_series:
                subjects_vital_series = ward_series[item_name]
                # Convert so these are only the values, for this task we do not need the times
                subjects_vital_values = list(subjects_vital_series.values())
                # print(f"SERIES {item_name} {subjects_vital_series}")
                # print(f"VALUES {item_name} {subjects_vital_values}")
                # element = subjects_vital_values[0]
                # print(f"TYPE   {item_name} {type(element)}")
                # if not isinstance(element, float):
                #     raise TypeError(f"Expected float but got {type(element)}")

                vitals_values[item_name].extend(subjects_vital_values)

            if item_name in lab_series:
                subjects_vital_series = lab_series[item_name]
                subjects_vital_values = list(subjects_vital_series.values())
                vitals_values[item_name].extend(subjects_vital_values)


    # Now compute the stats for all subjects considered
    vital_stats = dict()
    for item_name in sorted(item_names):
        vitals_values_list = vitals_values[item_name]

        n = len(vitals_values_list)
        if n < 2:
            print(f"Skipping {item_name}, too few values {vitals_values_list}")
            continue

        # if item_name == 'air':
        #     print(f"DEBUG {item_name}, values {vitals_values_list}")
        # if item_name == 'svi':
        #     print(f"DEBUG {item_name}, values {vitals_values_list}")

        # print(f"Vital {item_name} list={vitals_values_list}")
        mean  = compute_mean(vitals_values_list)
        stdev = compute_stdev(vitals_values_list)
        median = compute_median(vitals_values_list)
        mad    = compute_mean_absolute_deviation(vitals_values_list, median=median)

        # When MAD=0 (indicating that over 50% of the values are identical,
        # often at the median of 0.  This happens quite often
        # This will cause issue for computing modified z-score :(
        # "vent": {
        # "mean": 0.6096265056150529,
        # "stdev": 0.4878344782157534,
        # "median": 1.0,
        # "mad": 0.0,
        # "length": 530182}


        # Sanity checks, things that result in invalid z-scores
        # if math.isnan(stdev):
        #     raise ValueError(f"stdev is nan: {item_name} -> {vitals_values_list}")
        # if stdev == 0.0:
        #     raise ValueError(f"stdev is 0.0: {item_name} -> {vitals_values_list}")
        # if math.isnan(mad):
        #     raise ValueError(f"mad is nan: {item_name} -> {vitals_values_list}")
        # if mad == 0.0:
        #     raise ValueError(f"mad is 0.0: {item_name} -> {vitals_values_list}")

        # # We assume the median/mad might be invalid while mean/stdev might still be valid
        # # So, if the median/mad is invlida, fall back to the mean/stdev, otherwise complain
        # use_z_score = False
        # if math.isnan(mad) or mad == 0.0:
        #     use_z_score = True
        #     if math.isnan(stdev) or stdev == 0.0:
        #         raise ValueError(f"stdev {stdev} is invalid: {item_name} -> {vitals_values_list}")


        # print(f"\nVital {item_name}")
        # print(f"      mean {mean:.2f}")
        # print(f"     stdev {stdev:.2f}")
        # print(f"    median {median:.2f}")
        # print(f"       mad {mad:.2f}")

        vital_stats[item_name] = {'mean':mean, 'stdev':stdev, 'median':median, 'mad':mad, 'length':n}

    # Save dictionary to JSON file
    save_vital_stats(vital_stats)

def main():
    # Read all subjects to compute and save vital stats, this will take 2-3 minutes
    # read_subjects_save_vital_stats()

    # Read in saved vitals (very fast) and print out example stats
    vital_stats = read_vital_stats()
    print(f"hr -> {vital_stats['hr']}")



if __name__ == "__main__":
    main()