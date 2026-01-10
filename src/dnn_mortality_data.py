"""
Module to load real INSPIRE dataset for mortality prediction.
Replaces synthetic data generation with actual data from CSV files.
"""

import pandas as pd
import os
from typing import Dict, Tuple, List, Any


def generate_dataset(
    dataset_dir: str,
    feature_columns: List[str],
    proportion_died: float = None,
    num_subjects: int = None
) -> Tuple[Dict[str, Any], int]:
    """
    Load real INSPIRE dataset from CSV files and format for mortality prediction.
    
    Parameters
    ----------
    dataset_dir : str
        Path to directory containing INSPIRE CSV files (operations.csv, vitals.csv, labs.csv, etc.)
        Can also be a path to the parent directory containing 'died' and 'survived' subdirectories
        with subject JSON files.
    feature_columns : List[str]
        List of feature names to extract (e.g., ['glucose', 'potassium', 'sodium', 'creatinine'])
    proportion_died : float, optional
        Desired proportion of deceased patients in the returned dataset.
        If None, returns all available data with actual proportions.
    num_subjects : int, optional
        Maximum number of subjects to load. If None, loads all available subjects.
        
    Returns
    -------
    Tuple[Dict[str, Any], int]
        subjects_data : Dict[str, Any]
            Dictionary mapping subject_id to subject data containing:
            - 'timeseries': Dict[str, pd.DataFrame] - timeseries data for each feature with 'chart_time'
            - 'label': int - 0 for survived, 1 for died in hospital
            - 'subject_id': str or int
        seq_length : int
            Maximum sequence length across all subjects (max_chart_time - min_chart_time)
    """
    
    # Try to load from INSPIRE dataset CSVs
    if os.path.exists(os.path.join(dataset_dir, 'operations.csv')):
        return _load_from_inspire_csvs(dataset_dir, feature_columns, proportion_died, num_subjects)
    # Try to load from saved JSON subjects
    elif os.path.isdir(os.path.join(dataset_dir, 'died')) or os.path.isdir(os.path.join(dataset_dir, 'survived')):
        return _load_from_subject_jsons(dataset_dir, feature_columns, proportion_died, num_subjects)
    else:
        raise ValueError(
            f"Dataset directory '{dataset_dir}' must contain either:\n"
            "  1. INSPIRE CSV files (operations.csv, vitals.csv, labs.csv, ward_vitals.csv)\n"
            "  2. Subdirectories 'died' and 'survived' with subject JSON files"
        )


def _load_from_inspire_csvs(
    dataset_dir: str,
    feature_columns: List[str],
    proportion_died: float = None,
    num_subjects: int = None
) -> Tuple[Dict[str, Any], int]:
    """
    Load INSPIRE dataset directly from CSV files.
    
    CSV files expected:
    - operations.csv: Contains subject_id, op_id, inhosp_death_time, age, etc.
    - vitals.csv: Contains subject_id, op_id, chart_time, item_name, value
    - labs.csv: Contains subject_id, chart_time, item_name, value
    - ward_vitals.csv: Contains subject_id, chart_time, item_name, value
    """
    print(f"Loading INSPIRE dataset from CSV files in: {dataset_dir}")
    
    # Load operations to determine mortality
    operations_path = os.path.join(dataset_dir, 'operations.csv')
    vitals_path = os.path.join(dataset_dir, 'vitals.csv')
    labs_path = os.path.join(dataset_dir, 'labs.csv')
    ward_vitals_path = os.path.join(dataset_dir, 'ward_vitals.csv')
    
    print("  Reading operations.csv...")
    operations_df = pd.read_csv(operations_path)
    
    print("  Reading vitals.csv...")
    vitals_df = pd.read_csv(vitals_path)
    
    print("  Reading labs.csv...")
    labs_df = pd.read_csv(labs_path)
    
    print("  Reading ward_vitals.csv...")
    ward_vitals_df = pd.read_csv(ward_vitals_path)
    
    # Get unique subjects and their mortality status
    subjects_data = {}
    
    # Group operations by subject_id
    unique_subjects = operations_df['subject_id'].unique()
    
    # Limit number of subjects if specified
    if num_subjects:
        unique_subjects = unique_subjects[:num_subjects]
    
    print(f"  Processing {len(unique_subjects)} subjects...")
    
    for subject_id in unique_subjects:
        # Check if subject died in hospital
        subject_ops = operations_df[operations_df['subject_id'] == subject_id]
        died_in_hospital = subject_ops['inhosp_death_time'].notna().any()
        label = 1 if died_in_hospital else 0
        
        # Extract timeseries data for this subject
        timeseries = _extract_timeseries_for_subject(
            subject_id, feature_columns, vitals_df, labs_df, ward_vitals_df
        )
        
        # Skip subjects with no timeseries data
        if not timeseries or len(timeseries) == 0:
            continue
        
        subjects_data[subject_id] = {
            'subject_id': subject_id,
            'label': label,
            'timeseries': timeseries
        }
    
    # Filter by proportion_died if specified
    if proportion_died is not None:
        subjects_data = _balance_mortality(subjects_data, proportion_died)
    
    # Calculate max sequence length
    seq_length = _calculate_max_sequence_length(subjects_data)
    
    print(f"  Loaded {len(subjects_data)} subjects with sequence length {seq_length}")
    died_count = sum(1 for s in subjects_data.values() if s['label'] == 1)
    survived_count = len(subjects_data) - died_count
    print(f"  Mortality: {died_count} died, {survived_count} survived")
    
    return subjects_data, seq_length


def _load_from_subject_jsons(
    base_dir: str,
    feature_columns: List[str],
    proportion_died: float = None,
    num_subjects: int = None
) -> Tuple[Dict[str, Any], int]:
    """
    Load INSPIRE dataset from pre-processed subject JSON files.
    Expected directory structure:
    - base_dir/died/*.json
    - base_dir/survived/*.json
    """
    print(f"Loading INSPIRE dataset from subject JSON files in: {base_dir}")
    
    from subject import Subject
    
    subjects_data = {}
    
    # Load from died directory
    died_dir = os.path.join(base_dir, 'died')
    survived_dir = os.path.join(base_dir, 'survived')
    
    total_loaded = 0
    for subject_dir, label in [(died_dir, 1), (survived_dir, 0)]:
        if not os.path.isdir(subject_dir):
            print(f"  Warning: Directory not found: {subject_dir}")
            continue
        
        label_name = "died" if label == 1 else "survived"
        print(f"  Reading {label_name} subjects from {subject_dir}...")
        
        for json_file in os.listdir(subject_dir):
            if json_file.endswith('.json'):
                json_path = os.path.join(subject_dir, json_file)
                
                try:
                    subject = Subject()
                    subject.fromJSON(json_path)
                    subject_id = subject.get_subject_id()
                    
                    # Extract timeseries from subject vitals
                    timeseries = _extract_timeseries_from_subject(
                        subject, feature_columns
                    )
                    
                    if timeseries and len(timeseries) > 0:
                        subjects_data[subject_id] = {
                            'subject_id': subject_id,
                            'label': label,
                            'timeseries': timeseries,
                            'subject_obj': subject  # Keep reference if needed
                        }
                        total_loaded += 1
                        
                        if num_subjects and total_loaded >= num_subjects:
                            break
                except Exception as e:
                    print(f"    Warning: Error loading {json_file}: {e}")
                    continue
        
        if num_subjects and total_loaded >= num_subjects:
            break
    
    # Filter by proportion_died if specified
    if proportion_died is not None:
        subjects_data = _balance_mortality(subjects_data, proportion_died)
    
    # Calculate max sequence length
    seq_length = _calculate_max_sequence_length(subjects_data)
    
    print(f"  Loaded {len(subjects_data)} subjects with sequence length {seq_length}")
    died_count = sum(1 for s in subjects_data.values() if s['label'] == 1)
    survived_count = len(subjects_data) - died_count
    print(f"  Mortality: {died_count} died, {survived_count} survived")
    
    return subjects_data, seq_length


def _extract_timeseries_for_subject(
    subject_id,
    feature_columns: List[str],
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    ward_vitals_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Extract timeseries data for a single subject from the DataFrames.
    Returns a dict mapping feature names to DataFrames with 'chart_time' and feature values.
    """
    timeseries = {}
    
    # Combine all vital/lab data for this subject
    all_data = []
    
    # Get vitals for this subject
    subject_vitals = vitals_df[vitals_df['subject_id'] == subject_id]
    if len(subject_vitals) > 0:
        all_data.append(subject_vitals[['chart_time', 'item_name', 'value']].copy())
    
    # Get labs for this subject
    subject_labs = labs_df[labs_df['subject_id'] == subject_id]
    if len(subject_labs) > 0:
        all_data.append(subject_labs[['chart_time', 'item_name', 'value']].copy())
    
    # Get ward vitals for this subject
    subject_ward_vitals = ward_vitals_df[ward_vitals_df['subject_id'] == subject_id]
    if len(subject_ward_vitals) > 0:
        all_data.append(subject_ward_vitals[['chart_time', 'item_name', 'value']].copy())
    
    if not all_data:
        return {}
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Extract each requested feature
    for feature in feature_columns:
        feature_data = combined_df[combined_df['item_name'] == feature].copy()
        
        if len(feature_data) > 0:
            # Rename columns for consistency
            feature_data = feature_data[['chart_time', 'value']].copy()
            feature_data.columns = ['chart_time', feature]
            feature_data['chart_time'] = pd.to_numeric(feature_data['chart_time'], errors='coerce')
            feature_data[feature] = pd.to_numeric(feature_data[feature], errors='coerce')
            # Remove NaN rows
            feature_data = feature_data.dropna()
            
            if len(feature_data) > 0:
                timeseries[feature] = feature_data.sort_values('chart_time').reset_index(drop=True)
    
    return timeseries


def _extract_timeseries_from_subject(subject, feature_columns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Extract timeseries data from a Subject object.
    """
    timeseries = {}
    
    # Get ward vitals from subject (these have chart_time and values)
    for feature in feature_columns:
        data_list = []
        
        # Try to extract from ward vitals
        for vital in subject.get_ward_vitals():
            # Assuming ward vital has structure with item_name and value
            if hasattr(vital, 'item_name') and vital.item_name == feature:
                if hasattr(vital, 'chart_time') and hasattr(vital, 'value'):
                    try:
                        chart_time = float(vital.chart_time)
                        value = float(vital.value)
                        data_list.append({'chart_time': chart_time, feature: value})
                    except (ValueError, TypeError):
                        continue
        
        if data_list:
            df = pd.DataFrame(data_list)
            df = df.sort_values('chart_time').reset_index(drop=True)
            timeseries[feature] = df
    
    return timeseries


def _balance_mortality(
    subjects_data: Dict[str, Any],
    target_proportion_died: float
) -> Dict[str, Any]:
    """
    Filter subjects to achieve target proportion of deceased patients.
    """
    died_subjects = {sid: s for sid, s in subjects_data.items() if s['label'] == 1}
    survived_subjects = {sid: s for sid, s in subjects_data.items() if s['label'] == 0}
    
    num_died = len(died_subjects)
    num_survived = len(survived_subjects)
    
    # Calculate target numbers based on proportion
    total = num_died + num_survived
    target_died = int(total * target_proportion_died)
    target_survived = total - target_died
    
    # Adjust if we don't have enough subjects in one category
    if target_died > num_died:
        target_died = num_died
        target_survived = total - target_died
    elif target_survived > num_survived:
        target_survived = num_survived
        target_died = total - target_survived
    
    # Randomly sample to achieve targets
    import random
    if target_died < num_died:
        died_ids = list(died_subjects.keys())
        kept_died_ids = set(random.sample(died_ids, target_died))
        died_subjects = {sid: s for sid, s in died_subjects.items() if sid in kept_died_ids}
    
    if target_survived < num_survived:
        survived_ids = list(survived_subjects.keys())
        kept_survived_ids = set(random.sample(survived_ids, target_survived))
        survived_subjects = {sid: s for sid, s in survived_subjects.items() if sid in kept_survived_ids}
    
    # Combine back
    balanced = {**died_subjects, **survived_subjects}
    return balanced


def _calculate_max_sequence_length(subjects_data: Dict[str, Any]) -> int:
    """
    Calculate the maximum sequence length across all subjects.
    Sequence length is defined as max(chart_time) - min(chart_time).
    """
    max_length = 0
    
    for subject_info in subjects_data.values():
        timeseries = subject_info.get('timeseries', {})
        
        all_times = []
        for feature_df in timeseries.values():
            if len(feature_df) > 0:
                all_times.extend(feature_df['chart_time'].values)
        
        if all_times:
            subject_length = int(max(all_times) - min(all_times))
            max_length = max(max_length, subject_length)
    
    return max_length if max_length > 0 else 1000  # Default minimum sequence length
