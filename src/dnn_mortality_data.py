"""
dnn_mortality_data.py

Two data sources, same output shape:

1. generate_dataset()   -- synthetic data for pipeline development/testing
2. load_real_subjects()  -- real INSPIRE subject JSON files (one .json per
                            patient, containing 'operations', 'labs',
                            'ward_vitals', 'vitals', 'diagnoses', 'medications')

Both return:
    subjects_data = {
        'subject_id': {
            'timeseries': {feature_name: DataFrame(chart_time, feature_name), ...},
            'label': 0 or 1
        },
        ...
    }
    seq_length = int

This is exactly the shape dnn_mortality_pipeline.main() expects, so the rest
of the pipeline (align_time_series, preprocess_for_autoencode, etc.) does not
need to change regardless of which loader you call.
"""
import pandas as pd
import numpy as np
import json
import os
import glob
import random


# ============================================================
# SYNTHETIC DATA (kept for quick pipeline smoke-testing)
# ============================================================
def generate_time_series(subject_id, feature_name, observed_samples, min_chart_time, max_chart_time, label):
    chart_time = np.sort(np.random.randint(min_chart_time, max_chart_time, size=observed_samples))
    chart_time = np.unique(chart_time)
    values = np.random.randint(50, 150, size=len(chart_time))

    if (feature_name == 'glucose' or feature_name == 'sodium') and label == 1:
        slope = 25 * random.random() / max(len(values), 1)
        for time_step in range(len(values)):
            values[time_step] = values[time_step] + (slope * time_step)

    df = pd.DataFrame({'chart_time': chart_time, feature_name: values})
    return df


def generate_dataset(num_subjects, feature_columns, proportion_died):
    observed_samples = 200
    base_min_ct = 0
    base_max_ct = 1000

    subjects_data = dict()
    for i in range(num_subjects):
        subject_id = f'subject{i}'
        label = np.random.choice([0, 1], p=[1.0 - proportion_died, proportion_died])

        time_series = dict()
        for feature_name in feature_columns:
            time_series[feature_name] = generate_time_series(
                subject_id, feature_name, observed_samples, base_min_ct, base_max_ct, label
            )
        subjects_data[subject_id] = {'timeseries': time_series, 'label': label}

    subject_ct_length = base_max_ct - base_min_ct
    seq_length = int(0.025 * subject_ct_length)
    print(f"USING seq_length={seq_length}")
    return subjects_data, seq_length


# ============================================================
# REAL INSPIRE JSON DATA LOADER
# ============================================================
def _parse_time(value):
    """Operation time fields come in as strings (sometimes with whitespace) or
    None/empty for missing values. Returns float or None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == '':
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _thirty_day_label(operation):
    """
    label = 1 if patient died within 30 days of leaving the OR (orout_time),
    else 0. Same definition used by James's subject.inhosp_death_30day()
    and the GBM pipeline.
    """
    death_time = _parse_time(operation.get('inhosp_death_time'))
    orout_time = _parse_time(operation.get('orout_time'))
    if death_time is None or orout_time is None:
        return 0
    thirty_days_minutes = 30 * 24 * 60
    return 1 if death_time < (orout_time + thirty_days_minutes) else 0


def _records_to_chart_time_dict(records, feature_columns, min_chart_time, max_chart_time):
    """
    Converts a flat list of {chart_time, item_name, value} dicts (as stored in
    each subject JSON's 'labs' / 'ward_vitals' lists) into:

        {feature_name: {chart_time: value, ...}, ...}

    filtered to [min_chart_time, max_chart_time] inclusive. This mirrors
    James's convert_vitals_to_dictionary() in his version of this file.
    """
    by_feature = {col: dict() for col in feature_columns}
    for r in records:
        item = r.get('item_name')
        if item not in by_feature:
            continue
        ct = _parse_time(r.get('chart_time'))
        val = _parse_time(r.get('value'))
        if ct is None or val is None:
            continue
        if min_chart_time <= ct <= max_chart_time:
            by_feature[item][ct] = val
    return by_feature


def extract_frames_from_json(subject_data, feature_columns, minutes_before_operation):
    """
    Real-data equivalent of James's extract_frames(), operating on a single
    already-loaded subject JSON dict instead of a Subject object.

    Pulls observations from BOTH labs and ward_vitals (whichever list has the
    matching item_name) for the window ending at the last operation's
    orin_time, going back `minutes_before_operation` minutes.

    :param subject_data: parsed JSON dict for one subject (has 'operations',
                          'labs', 'ward_vitals', ...)
    :param feature_columns: list of item_name strings to extract
    :param minutes_before_operation: size of the pre-op window in minutes
    :return: (timeseries dict, label) where
             timeseries = {feature_name: {chart_time: value, ...}, ...}
             label = 0 or 1
    """
    operations = subject_data.get('operations', [])
    if len(operations) == 0:
        return None, None

    # Use the last operation, consistent with James's get_last_operation()
    last_operation = operations[-1]
    orin_time = _parse_time(last_operation.get('orin_time'))
    if orin_time is None:
        return None, None

    min_chart_time = orin_time - minutes_before_operation
    max_chart_time = orin_time - 1  # omit the operation itself

    all_records = subject_data.get('labs', []) + subject_data.get('ward_vitals', [])
    timeseries = _records_to_chart_time_dict(all_records, feature_columns, min_chart_time, max_chart_time)

    label = _thirty_day_label(last_operation)
    return timeseries, label


def get_chart_time_range(subject_id, subject_data):
    """
    Same logic as James's get_chart_time_range(): finds the min/max chart_time
    actually observed across all of this subject's extracted feature series,
    used to determine how long the aligned dataframe needs to be.
    Returns minutes (int) or None if no observations at all.
    """
    timeseries = subject_data['timeseries']
    subject_min_ct = None
    subject_max_ct = None
    for feature_name in timeseries.keys():
        chart_time2values = timeseries[feature_name]
        if len(chart_time2values) == 0:
            continue
        feature_min_ct = min(chart_time2values.keys())
        feature_max_ct = max(chart_time2values.keys())
        if subject_min_ct is None or feature_min_ct < subject_min_ct:
            subject_min_ct = feature_min_ct
        if subject_max_ct is None or feature_max_ct > subject_max_ct:
            subject_max_ct = feature_max_ct

    if subject_min_ct is None or subject_max_ct is None:
        return None
    return subject_max_ct - subject_min_ct


def load_real_subjects(json_dir, feature_columns, days_before_operation=5, min_observations=2):
    """
    Loads every <subject_id>.json file in json_dir and returns subjects_data
    in the exact shape dnn_mortality_pipeline.main() expects from
    generate_dataset().

    :param json_dir: folder containing one .json file per patient
    :param feature_columns: list of lab/ward_vital item_name strings, e.g.
                             ['glucose', 'potassium', 'sodium', 'creatinine']
    :param days_before_operation: size of the pre-operative window, in days,
                                   ending at orin_time of the last operation
    :param min_observations: skip subjects with fewer than this many total
                              observations across all feature_columns combined
                              within the window (too sparse to interpolate
                              meaningfully)
    :return: (subjects_data, seq_length)
    """
    json_paths = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    if len(json_paths) == 0:
        raise FileNotFoundError(f"No .json files found in {json_dir}")

    minutes_before_operation = days_before_operation * 24 * 60

    subjects_data = dict()
    skipped_no_operation = 0
    skipped_sparse = 0

    for path in json_paths:
        with open(path) as f:
            raw = json.load(f)

        subject_id = raw.get('subject_id', os.path.splitext(os.path.basename(path))[0])

        timeseries, label = extract_frames_from_json(raw, feature_columns, minutes_before_operation)
        if timeseries is None:
            skipped_no_operation += 1
            continue

        total_observations = sum(len(v) for v in timeseries.values())
        if total_observations < min_observations:
            skipped_sparse += 1
            continue

        # Ensure every requested feature key exists (even if empty dict) so
        # downstream align_time_series() always sees a consistent set of
        # feature_columns -- matches James's timeseries_subset pattern.
        for feature_name in feature_columns:
            if feature_name not in timeseries:
                timeseries[feature_name] = dict()

        subjects_data[subject_id] = {'timeseries': timeseries, 'label': label}

    if len(subjects_data) == 0:
        raise ValueError(
            f"Loaded 0 usable subjects from {json_dir} "
            f"(skipped_no_operation={skipped_no_operation}, skipped_sparse={skipped_sparse}). "
            f"Check that feature_columns match real item_name values in labs/ward_vitals, "
            f"e.g. ['glucose','potassium','sodium','creatinine']."
        )

    num_died = sum(1 for s in subjects_data.values() if s['label'] == 1)
    num_survived = len(subjects_data) - num_died
    print(f"load_real_subjects: loaded {len(subjects_data)} subjects "
          f"({num_died} died, {num_survived} survived) "
          f"[skipped {skipped_no_operation} no-operation, {skipped_sparse} too-sparse]")

    # seq_length: 2.5% of the pre-op window length, same heuristic as
    # generate_dataset(), with a sane floor so short windows still work.
    seq_length = max(5, int(0.025 * minutes_before_operation))
    print(f"USING seq_length={seq_length}")

    return subjects_data, seq_length


if __name__ == "__main__":
    # Quick smoke test: point this at a folder of subject JSON files.
    feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']
    subjects_data, seq_length = load_real_subjects('./inspire_subjects_json', feature_columns)
    for sid, info in list(subjects_data.items())[:3]:
        print(f"{sid}: label={info['label']}")
        for feat, vals in info['timeseries'].items():
            print(f"  {feat}: {len(vals)} observations")
