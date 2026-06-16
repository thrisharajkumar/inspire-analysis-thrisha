import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import NearestNDInterpolator

import matplotlib.pyplot as plt

import random
from stopwatch import Stopwatch
import subject as subject_module
import subject_analysis
import vital_stats as vital_stats_module


def smooth_fade_to_mean_interpolator(x, y, kind='cubic', decay=1.0):
    """
    Creates a function that interpolates within data and smoothly decays to mean outside.

    Parameters:
        x, y  : arrays of data points
        kind  : interp1d interpolation type (default cubic)
        decay : length scale over which to fade to mean outside range
    """
    mean_val = np.mean(y)
    interp_func = interp1d(x, y, kind=kind, bounds_error=False, fill_value="extrapolate")

    x_min, x_max = np.min(x), np.max(x)

    def faded_func(x_query):
        x_query = np.asarray(x_query)
        y_interp = interp_func(x_query)

        # Distances outside domain
        dist_left = x_min - x_query
        dist_right = x_query - x_max

        # Left side fade mask
        mask_left = x_query < x_min
        weight_left = np.exp(-np.abs(dist_left[mask_left]) / decay)
        y_interp[mask_left] = weight_left * y_interp[mask_left] + (1 - weight_left) * mean_val

        # Right side fade mask
        mask_right = x_query > x_max
        weight_right = np.exp(-np.abs(dist_right[mask_right]) / decay)
        y_interp[mask_right] = weight_right * y_interp[mask_right] + (1 - weight_right) * mean_val

        return y_interp

    return faded_func


# Step 1: Align time series to a common time grid for a single subject
def align_time_series( time_series, full_length, vital_stats, subject_id ):
    """
    Takes time series possibly at different sample rates and interpolate so that
    all values are aligned (either observed value or interpolated).  A mask
    column is created for each time series to indicate whether the value was
    observed or interpolated.
    :param time_series: dict[feature_names] of dict[chart_time, value]
    :param full_length: length of longest vitals (other will be interpolated/padded)
    :param vital_stats: dict lookup to find statistics (mean) for feature, used to fill in missing values
    :return: dataframe with 2*len(time_series)
    """

    # Find the common time grid (min to max chart_time across all series)
    all_times = []
    # for series in time_series.values():
    #     all_times.extend(series[time_column].values)
    for feature_name in time_series.keys():
        chart_time2values = time_series[feature_name]
        all_times.extend( list(chart_time2values.keys()) )
    min_time = min(all_times)
    max_time = max(all_times)

    # Create a time grid up to full_length, starting from min_time
    # common_time = range(min_time, max_time)
    common_time = range(min_time, min(min_time + full_length, max_time + 1))
    if len(common_time) < full_length:
        # Pad with additional time points
        common_time = range(min_time, min_time + full_length)
    # print(f"len(common_time)={common_time}  min_time={min_time}  max_time={max_time}")

    # if len(common_time) < full_length:
    #     # Pad with additional time points
    #     common_time = range(min_time, min_time + full_length)

    # Create an empty DataFrame with the common time grid
    df = pd.DataFrame(index=common_time)

    # Interpolate each series
    for feature_name, chart_time2values in time_series.items():

        # Depending on how many observed values exists
        # create observed/interpolated values and masks
        values      = None # must be a dict mapping chart_time to value
        mask_values = None # must be a dict mapping chart_time to mask value
        if len(chart_time2values) == 0:
            # "hr": {
            #     "mean": 75.6669105300075,
            #     "stdev": 16.314533970124334,
            #     "median": 74.0,
            #     "mad": 12.0,
            #     "length": 11536516}
            stats = vital_stats[feature_name]
            mean = stats['mean']
            median = stats['median']
            print(f"ALIGN {subject_id} No feature data {feature_name}, using mean {mean} or median {median}")
            values = dict()
            mask_values = dict()
            for chart_time in common_time:
                values[chart_time] = median
                mask_values[chart_time] = 0.0
            print(f"    len values {len(values)}" )
        else:
            # Nice to include if/else block but treated as interpolated for now
            if len(chart_time2values) == 1:
                # If only one value (interpolate produces nans)
                # We add another value a next minute, same value
                # Though artificial, believe this is very minor
                # print(f"ALIGN BEFORE={series_df}")
                first_key, first_value = next(iter(chart_time2values.items()))
                chart_time2values[first_key + 1] = first_value

            #-------------------------------------------------#
            # Interpolate the series
            # -------------------------------------------------#
            # Convert the chart_time2values to list of keys and values
            x_data = list( chart_time2values.keys() )
            y_data = list( chart_time2values.values() )
            # 'quadratic' is also suitable,nearest ok, cubic results in massive deviations
            kind = 'linear'
            interp_func_smooth = smooth_fade_to_mean_interpolator(x_data, y_data, kind=kind, decay=1.0)
            # Convert range to list for interp1d
            values = interp_func_smooth(list(common_time))

            # Create the mask
            mask_values = dict()
            for chart_time in common_time:
                if chart_time in chart_time2values:
                    mask_values[chart_time] = 1.0
                else:
                    mask_values[chart_time] = 0.0

        # Fit spline with smoothing factor s=0 (through all points)
        # spline_func = UnivariateSpline(x_data, y_data, s=0)

        # For each feature, add interpolated and mask for all common_time keys
        # Mask (1.0 for observed, 0.0 for interpolated or padded)
        df[feature_name] = values
        df[f'{feature_name}_mask'] = mask_values

        # Sanity check, can happen when we only have one sample, interpolate does not handle
        if df[feature_name].isna().any():
            # print(f"x_data={x_data}")
            # print(f"y_data={y_data}")
            raise ValueError(f"Series {feature_name} has nan values!")

    return df

#
# # Step 2: Normalize the data
# def normalize_data(df, feature_columns, scaler=None):
#     if scaler is None:
#         scaler = StandardScaler()
#         df[feature_columns] = scaler.fit_transform(df[feature_columns])
#     else:
#         df[feature_columns] = scaler.transform(df[feature_columns])
#     return df, scaler




# Plot time series (unchanged)
def plot_time_series(df, features, subject):
    """
    Plots all feature names separating observed from interpolated (using the mask).
    Assumes dataframe has associated mask (name suffixed with _mask), 0 = interpolated, 1 = observed.
    :param df:
    :param features:
    :param subject:
    :return:
    """
    subject_id = subject.get_subject_id()
    operations = subject.get_operations()
    print(f"plot_time_series: {subject_id} num operations {len(operations)}")

    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(features, 1):
        plt.subplot(len(features), 1, i)

        # Sometime (due to filtering time range) we do not have requested features in df
        if feature in df.columns:
            interpolated = df[df[f'{feature}_mask'] == 0.0]
            plt.scatter(interpolated.index, interpolated[feature], label=f'{feature} (interpolated)',
                        color='blue', s=50, marker='o', alpha=0.5)
            observed = df[df[f'{feature}_mask'] == 1.0]
            plt.scatter(observed.index, observed[feature], label=f'{feature} (observed)',
                        color='red', s=50, marker='o')
        plt.title(f'{subject_id} {feature} Time Series')
        plt.xlabel('Time')
        plt.ylabel(feature)

        # Write indicator when operation in/out of room and operation starts/stops
        # for operation in operations:
        operation = subject.get_last_operation()
        orin_time = int(operation['orin_time'].strip())
        orout_time = int(operation['orout_time'].strip())
        opstart_time = int(operation['opstart_time'].strip())
        opend_time = int(operation['opend_time'].strip())

        plt.axvline(x=orin_time, color='orange')
        plt.axvline(x=orout_time, color='orange')
        plt.axvline(x=opstart_time, color='cyan')
        plt.axvline(x=opend_time, color='cyan')

        # also show operation times on medication plot
        plt.axvline(x=orin_time, color='orange')
        plt.axvline(x=orout_time, color='orange')
        plt.axvline(x=opstart_time, color='cyan')
        plt.axvline(x=opend_time, color='cyan')

        # also show operation times on labs plot
        plt.axvline(x=orin_time, color='orange')
        plt.axvline(x=orout_time, color='orange')
        plt.axvline(x=opstart_time, color='cyan')
        plt.axvline(x=opend_time, color='cyan')

        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()



def generate_time_series(subject_id, feature_name, observed_samples, min_chart_time, max_chart_time, label):
    """
    Generate a pandas DataFrame with a specified sequence length, name, and random increasing chart_time.
    For subjects that died, we correlate glucose (slight linear increase in time).
    Parameters:
    observed_samples (int): Number of observed samples to generate, in [min_chart_time,max_chart_time)
    feature_name (str): Name of the column for values (e.g., 'glucose')
    min_chart_time (int): Minimum value for chart_time (inclusive)
    max_chart_time (int): Maximum value for chart_time (non-inclusive)
    label (bool): 1 the subject died, 0 they survived
    Returns:
    pd.DataFrame: DataFrame with increasing 'chart_time' and named column with sample values
    """
    # Generate random increasing chart_time values
    chart_time = np.sort(np.random.randint(min_chart_time, max_chart_time, size=observed_samples))
    # chart_time = sorted( random.sample(range(min_chart_time, max_chart_time), observed_samples) )

    # Remove duplicates, unfortunately, may no longer be sample_length
    chart_time = np.unique(chart_time)

    # Generate sample values (random integers between 100 and 150 to mimic glucose levels)
    values = np.random.randint(50, 150, size=len(chart_time))

    #-----------------------------------------------------#
    # USEFUL DEBUGGING CODE, ADDS CORRELATION WITH DIED
    #-----------------------------------------------------#
    if ( feature_name == 'glucose' or feature_name == 'sodium') and label == 1:
        # Add a correlation based on survived/died
        # print(f"Adding correlation {df_name} died={died}")
        slope = 25 * random.random() / len(values)
        for time_step in range(len(values)):
            value = values[time_step]
            value = value + (slope * time_step)
            values[time_step] = value
    # -----------------------------------------------------#


    # Create DataFrame
    df = pd.DataFrame({
        'chart_time': chart_time,
        feature_name: values
    })

    return df


def generate_dataset(num_subjects, feature_columns, proportion_died):
    # Make a function
    observed_samples = 200  # ideal, may generate duplicates
    base_min_ct = 0  # (inclusive)
    base_max_ct = 1000  # (non-inclusive)

    # TESTING fake/random data
    # print(f"Number of items {len(subjects_data)}")
    subjects_data = dict()

    for i in range(num_subjects):
        subject_id = f'subject{i}'
        # label = random.choice([0, 1])  # 0=survived, 1=died
        label = np.random.choice([0, 1], p=[1.0-proportion_died, proportion_died])

        time_series = dict()
        for feature_name in feature_columns:
            time_series[feature_name] = generate_time_series(subject_id,
                                                             feature_name,
                                                             observed_samples,
                                                             base_min_ct,
                                                             base_max_ct,
                                                             label)
        subject = {
            'timeseries': time_series,
            'label': label
        }
        subjects_data[f'subject{i}'] = subject


    # Determine the chart_time range for only this subject's time series
    subject_ct_length = base_max_ct - base_min_ct
    # Derive seq_length from the subject_ct_length, critical hyper-parameter
    seq_length = int(0.025 * subject_ct_length)
    print(f"USING seq_length={seq_length}")

    return subjects_data, seq_length


# def generate_dataset(num_subjects, feature_columns, proportion_died):
#     observed_proportion = 0.3 # once time range determined, proportion observed sample
#     variance_proportion = 0.0 # for start and stop time range
#     vital_min_ct = -50  # (inclusive)
#     vital_max_ct = 50  # 100 # (non-inclusive)
#     variance = int((vital_max_ct - vital_min_ct) * variance_proportion)
#
#     # seed = 46
#     seed = 123456
#     np.random.seed(seed)
#
#     subjects_data = dict()
#     for i in range(num_subjects):
#         subject_id = f'subject{i}'
#         # label = random.choice([0, 1])  # 0=survived, 1=died
#         label = np.random.choice([0, 1], p=[1.0 - proportion_died, proportion_died])
#
#         time_series = dict()
#         for feature_name in feature_columns:
#             # We want random start/stop chart times, and random number of observed times/values
#             # sign = 1 if random.random() < 0.5 else -1
#             offset = random.randint(0, variance)
#             sign = random.choice([1, -1])
#             start_ct = vital_min_ct + (sign * offset)
#
#             offset = random.randint(0, variance)
#             sign = random.choice([1, -1])
#             stop_ct = vital_max_ct + (sign * offset)
#
#             time_range = stop_ct - start_ct
#             observed_samples = int(observed_proportion * time_range)
#             observed_samples = random.choice(range(observed_samples, observed_samples + variance))
#             print(f"Subject ID {i}: label? {label} range={time_range} observed_samples={observed_samples} vital={feature_name}")
#             time_series[feature_name] = generate_time_series(subject_id,
#                                                              feature_name,
#                                                              observed_samples,
#                                                              start_ct,
#                                                              stop_ct, label)
#             # time_series[feature_name] = generate_time_series(subject_id, feature_name, observed_samples, min_ct, max_ct, died)
#
#         subject = {
#             'timeseries': time_series,
#             'label': label
#         }
#         subjects_data[subject_id] = subject
#     # Ignore any randomness and just set to proportion of estimated range
#     seq_length = int(0.025 * (vital_max_ct - vital_min_ct))
#     return subjects_data, seq_length

def convert_vitals_to_dictionary(vitals, min_chart_time=None, max_chart_time=None):
    vitals_dict = dict()
    for subject_vital in vitals:
        # op_id      = subject_vital['op_id']
        # subject_id = subject_vital['subject_id']
        chart_time = int(subject_vital['chart_time'])

        min_constraint_met = (min_chart_time is None) or (chart_time >= min_chart_time)
        max_constraint_met = (max_chart_time is None) or (chart_time <= max_chart_time)
        if min_constraint_met and max_constraint_met:
            item_name = subject_vital['item_name']
            value = float(subject_vital['value'])

            if item_name not in vitals_dict:
                vitals_dict[item_name] = dict()
            series = vitals_dict[item_name]
            series[chart_time] = value
    return vitals_dict # dict of dicts

def extract_frames( subject, minutes_before_operation ):
    """
    Extracts all the subjects ward vitals, operations vitals, labs and medications as
    dataframes.  Each feature is represented by its own dataframe with one chart_time
    and feature column.
    :param minutes_before_operation:
    :param subject:
    :return: dict of dataframes [feature_name] -> dataframe{'chart_time','feature_name'}
    """

    """
    The time range for some subjects is massive.

    Note that chart_time is in minutes, but median taken over 5 minute intervals
    (will never have more than one value over five minute interval).
    
    In the INSPIRE dataset, the 'chart_time' represents time in 5-minute intervals, 
    relative to a reference point (time zero), which is defined as the first admission 
    time for operation during the study period. This applies to variables such as vital signs, 
    laboratory results, and other time-series data in tables like 'vitals', 
    'ward_vitals', 'labs', and 'medications'. The time is anonymized by 
    converting all timestamps to relative times (in minutes) from time zero, 
    and for variables with measurements shorter than 5 minutes, the data are 
    aggregated to the median value within these 5-minute intervals.
    """

    # operations = subject.get_operations_by_orin_time()
    # print(f"READ inspire_dataset: {subject.get_subject_id()} num operations {len(operations)}  min {min(operations.keys())}")
    # for orin_time in sorted( operations.keys() ):
    #     operation = operations[orin_time]
    #     print(f"  {operation['subject_id']} {orin_time} -> {operation['orin_time']}")

    # subject_analysis.plot_vitals(subject)

    # Filter to only include vitals within days_before_operation of last operation
    last_operation = subject.get_last_operation()
    last_operation_orin_time = int(last_operation['orin_time'].strip())
    last_operation_orout_time = int(last_operation['orout_time'].strip())

    # Max time include start of operation
    # If predicting entirely before the operation, this is necessary
    # Interesting to consider during the operation but different inference problem
    #   e.g. when to stop because complex interaction between ward and operation vitals
    min_chart_time = last_operation_orin_time - minutes_before_operation
    max_chart_time = last_operation_orin_time - 1 # omit any equal, no operation vitals

    # Create the dict that will hold all the pre-processed vital series
    timeseries = dict()

    #-----------------------------------------------------#
    # Get an ward vitals that match feature name
    # -----------------------------------------------------#
    ward_vitals_list = subject.get_ward_vitals() # returns list
    ward_vitals_dict = convert_vitals_to_dictionary(ward_vitals_list, min_chart_time, max_chart_time)
    # Copy to consolidated dictionary
    for feature_name in ward_vitals_dict.keys():
        chart_time2values = ward_vitals_dict[feature_name]
        timeseries[feature_name] = chart_time2values

    # -----------------------------------------------------#
    # Get operation vitals, some may collide with ward vitals?
    # -----------------------------------------------------#
    labs_list = subject.get_labs()
    labs_dict = convert_vitals_to_dictionary(labs_list, min_chart_time, max_chart_time)
    for feature_name in labs_dict.keys():
        chart_time2values = labs_dict[feature_name]
        if feature_name in timeseries:
            # I believe nomenclature for labs is different than ward vitals, no collisions?
            raise ValueError(f"Conflict, have both ward vital and lab for {feature_name}")
        timeseries[feature_name] = chart_time2values

    # # -----------------------------------------------------#
    # # Get operation vitals, some may collide with ward vitals?
    # # -----------------------------------------------------#
    # vitals_list = subject.get_vitals()
    # vitals_dict = convert_vitals_to_dictionary(vitals_list, min_chart_time, max_chart_time)
    # for feature_name in vitals_dict.keys():
    #     chart_time2values = vitals_dict[feature_name]
    #     if feature_name in timeseries:
    #         # Alternatively append to existing dataframe?
    #         raise ValueError(f"Conflict, have both ward and operation vitals for {feature_name}")
    #     timeseries[feature_name] = chart_time2values

    # -----------------------------------------------------#
    # Need to also add medications
    # -----------------------------------------------------#


    return timeseries


def plot_chart_time_range_histogram(subjects):
    """
    Plots a histogram of the chart_time range (in hours) for a list of subjects.

    Args:
        subjects (list): List of Subject objects, each with get_min_chart_time()
                        and get_max_chart_time() methods returning chart_time
                        in minutes.
    """
    # Calculate chart_time range for each subject (in 5-minute intervals)
    # chart_time_ranges = [
    #     subject.get_max_chart_time() - subject.get_min_chart_time()
    #     for subject in subjects
    # ]

    # One chart time is 5 minutes
    chart_time_day = (24 * 60)  # 1440

    chart_time_ranges = list()
    for subject_id in subjects.keys():
        subject = subjects[subject_id]
        min_chart_time, max_chart_time = subject.get_chart_time_range()
        chart_time_range = max_chart_time - min_chart_time
        chart_time_ranges.append(chart_time_range)

    # Convert ranges from 5-minute intervals to hours (1 interval = 5 minutes = 5/60 hours)
    # chart_time_ranges_hours = [range_val * 5 / 60 for range_val in chart_time_ranges]
    # chart_time_ranges_days = [range_val * 5 / 60 / 24 for range_val in chart_time_ranges]
    chart_time_ranges_days = list()
    for chart_time_range in chart_time_ranges:
        range_in_days = chart_time_range / chart_time_day # ct / ct/day
        # print(f"{range_in_days}")
        chart_time_ranges_days.append(range_in_days)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(chart_time_ranges_days, bins=1000, edgecolor='black')
    plt.xlabel('Chart Time Range (days)')
    plt.ylabel('Number of Subjects')
    # plt.title('Histogram of Chart Time Range per Subject')
    plt.grid(True, alpha=0.3)
    plt.savefig('plot_chart_time_histogram.png', dpi=300)

    # Show plot
    plt.show()


def get_chart_time_range(subject_id, subject_data):
    subject_min_ct = None
    subject_max_ct = None
    timeseries = subject_data['timeseries']
    for feature_name in timeseries.keys():
        # dataframe = feature_frames[feature_name]
        # feature_min_ct = dataframe['chart_time'].min()
        # feature_max_ct = dataframe['chart_time'].max()

        chart_time2values = timeseries[feature_name]
        if len(chart_time2values) == 0:
            continue
        feature_min_ct = min(chart_time2values.keys())
        feature_max_ct = max(chart_time2values.keys())

        # print(f"{subject_id}: {feature_name} min={feature_min_ct} max ={feature_max_ct}")
        if subject_min_ct is None or feature_min_ct < subject_min_ct:
            subject_min_ct = feature_min_ct
        if subject_max_ct is None or feature_max_ct > subject_max_ct:
            subject_max_ct = feature_max_ct

    if subject_max_ct is None or subject_min_ct is None:
        print(f"No chart_times for {subject_id}, time series {timeseries}")
        return None
    # chart_time is in minutes, relative to 0 when subject was admitted for first operation
    time_range_minutes = subject_max_ct - subject_min_ct
    time_range_hours = time_range_minutes / 60
    time_range_days = time_range_hours / 24
    time_range_weeks = time_range_days / 7
    time_range_years = time_range_weeks / 52

    print(f"Subject ID {subject_id} died? {subject_data['label']}  [{subject_min_ct},{subject_max_ct}] time_range_minutes={time_range_minutes} days={time_range_days:.2f}  years={time_range_years:.3f}")
    return time_range_minutes


# Modified main function
def main():
    inspire_root_dir = "../inspire_subjects_small/"
    # inspire_root_dir = "../inspire_subjects"
    stopwatch = Stopwatch()
    subjects = subject_module.read_subjects(inspire_root_dir)
    print(f"Reading took {stopwatch.elapsedTime() / 60:.2f} minutes")
    # subject_module.subjects_statistics( subjects )
    # # LABS: 38
    # # MEDS: 1143
    # # VITS: 74
    # # WARD: 16
    # plot_chart_time_range_histogram( subjects )
    # if True:
    #     return

    # ---------------------------------------------------------------#
    # This code is analysing how 30 dat mortality is defined
    # Should consider moving to a dedicated main
    # ---------------------------------------------------------------#
    for subject_id in subjects.keys():
        subject = subjects[subject_id]
        operations = subject.get_operations_by_orin_time()
        number_operations = len(operations)
        op_i = 1
        for orin_time in sorted( operations.keys() ):
            operation = operations[orin_time]
            inhosp_death_time   = operation['inhosp_death_time'].strip()
            allcause_death_time = operation['allcause_death_time'].strip()
            orout_time          = operation['orout_time'].strip()
            death_30day = subject.inhosp_death_30day()
            if len(inhosp_death_time) > 0:
                print(f"{subject_id} 30 days? {death_30day} operations {number_operations}, died during operation {op_i} at orout_time={orout_time}  inhosp_death_time={inhosp_death_time}  allcause_death_time={allcause_death_time}")
            op_i += 1
    # ---------------------------------------------------------------#



    #---------------------------------------------------------------#
    # Sample data for multiple subjects (replace with your actual data)
    # These parameters are only for generating data
    # They are not molde hyper-parameters
    # ---------------------------------------------------------------#
    # num_subjects = 10
    # If we have a large number of subjects, sample to get smaller set, do before aligning!!!
    # subjects_subset = dict(random.sample(list(subjects.items()), num_subjects))

    feature_names = ['glucose', 'spo2']
    # feature_names = ['glucose', 'potassium', 'sodium', 'creatinine']
    # feature_names = ['hb', 'platelet', 'wbc', 'aptt', 'ptinr', 'glucose', 'spo2', 'hr', ]

    # INPUT_VARS = ['age', 'sex', 'emop', 'bmi', 'andur',
    #              'preop_hb', 'preop_platelet', 'preop_wbc',
    #              'preop_aptt', 'preop_ptinr', 'preop_glucose',
    #              'preop_bun', 'preop_albumin', 'preop_ast',
    #              'preop_alt', 'preop_creatinine', 'preop_sodium',
    #              'preop_potassium']

    # TESTING fake/random data
    # print(f"Number of items {len(multi_data)}")
    # proportion_died = 0.10

    # subjects_data = extract_features(feature_columns, subjects)
    # ---------------------------------------------------------------#
    # Convert the subject's labs, ward vitals, medications, vitals to
    # time series dataframes.  Each dataframe has its own
    # chart_time column and values column.  Align will them combine
    # into one consistent dataframe with one chart_time and all features_names.
    # ---------------------------------------------------------------#
    days_before_operation = 5
    minutes_before_operation = days_before_operation * 24 * 60

    subjects_data = dict()
    for subject_id in subjects.keys():
        subject = subjects[subject_id]

        timeseries = extract_frames( subject, minutes_before_operation )
        # Filter out features we are not interested in, to get subset
        # Thought about filtering in extract_features but simpler this way
        timeseries_subset = dict()
        for feature_name in feature_names:
            if feature_name in timeseries:
                chart_time2values = timeseries[feature_name]
            else:
                # subject did not have any feature data for time interval
                chart_time2values = dict()
            timeseries_subset[feature_name] = chart_time2values

        subject_data = {
            'timeseries': timeseries_subset,
            'label': subject.inhosp_death_30day()
        }
        subjects_data[subject_id] = subject_data
    print(f"Extracted data from {len(subjects_data)} subjects")

    # ---------------------------------------------------------------#
    # Irritating issue, the requested feature may not exist for subject
    # However, align will add "interpolated" value as the median for the feature.
    # We derive the median from all subjects in the INSPIRE dataset
    # ---------------------------------------------------------------#
    vital_stats = vital_stats_module.read_vital_stats()
    for subject_id in subjects_data.keys():
        subject_data = subjects_data[subject_id]

        timeseries = subject_data['timeseries']
        for feature_name in feature_names:
            if feature_name not in list(timeseries.keys()):
                print(f"{subject_id} has no feature {feature_name}")



    #---------------------------------------------------------------#
    # Visualise each subject's aligned series.
    # ---------------------------------------------------------------#
    print(f"Number of subjects {len(subjects_data)}")
    for subject_id in subjects_data.keys():
        subject = subjects[subject_id]
        subject_data = subjects_data[subject_id]
        # print(subject)
        # print(f"ALIGNED {subject_id}")

        time_range_minutes = get_chart_time_range(subject_id, subject_data)
        # print(f"  time_range_minutes {time_range_minutes}")
        if time_range_minutes is not None and time_range_minutes > 0:
            df = align_time_series(subject_data['timeseries'], time_range_minutes, vital_stats, subject_id)
            # print(f"  num vitals {len(df.columns)}")
            # print("DATAFRAME")
            # print(df)
            plot_time_series(df, feature_names, subject)
        else:
            print(f"ISSUE: No time series for subject {subject_id}")

        # subject_analysis.plot_vitals( subject )


    # ---------------------------------------------------------------#
    # Prepare the dataset for the encoding / feature learning phase
    # ---------------------------------------------------------------#
    # seq_length = 3
    # auto_dataloader, scaler, model_dim = preprocess_for_autoencode(multi_data, seq_length, full_length )

    # # Split data into train and test sets
    # train_data, test_data = create_train_test_split(multi_data, train_size=0.7, not_survived_pct=not_survived_pct)
    #
    # # Preprocess for autoencoding using training data
    # seq_length = 5 # this is a hyper-parameter
    # auto_dataloader, scaler, model_dim = preprocess_for_autoencode(train_data, seq_length, full_length)
    #
    # Demo: Print batch shapes
    # for batch, pos_enc in auto_dataloader:
    #     # print(f"Autoencode Batch shape: {batch.shape}")
    #     # print(f"Positional encoding shape: {pos_enc.shape}")
    #     # break
    #     print(f"Autoencode Batch: {batch}")
    #     print(f"Positional encoding: {pos_enc}")



if __name__ == "__main__":
    main()