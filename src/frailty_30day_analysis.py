"""
30-Day Frailty Analysis Module

Analyzes 30-day survival outcomes comparing survivors vs non-survivors,
with focus on frailty metrics and Hospital Frailty Risk Score (HFRS).

This module provides:
- 30-day outcome calculation from INSPIRE dataset
- Frailty score computation using ICD-10 diagnosis codes
- Comparative analysis and visualization of survivor vs non-survivor frailty profiles
"""

import pandas as pd
import numpy as np
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt

# Import local modules for frailty analysis
from subject import Subject
import subject as subject_module
from stopwatch import Stopwatch


def get_dataset_dir():
    """
    Determine the dataset directory path.
    Priority:
    1. Environment variable INSPIRE_DATASET_DIR
    2. ../dataset relative to this script
    3. C:\\Users\\pc\\Desktop\\dataset (fallback for backward compatibility)
    """
    # Check environment variable
    if 'INSPIRE_DATASET_DIR' in os.environ:
        return os.environ['INSPIRE_DATASET_DIR']
    
    # Check relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir, '..', 'dataset')
    if os.path.exists(relative_path):
        return relative_path
    
    # Fallback to Desktop path (for backward compatibility)
    desktop_path = "C:\\Users\\pc\\Desktop\\dataset"
    if os.path.exists(desktop_path):
        return desktop_path
    
    raise ValueError("Dataset directory not found. Set INSPIRE_DATASET_DIR environment variable or place dataset in ../dataset")


def analyse_30day_frailty_outcomes(dataset_dir=None):
    """
    Analyze 30-day survival outcomes comparing survivors vs non-survivors,
    with focus on frailty metrics.
    
    Plots:
    1. Frailty score distribution for survivors vs non-survivors at 30 days
    2. Outcome comparison by frailty category (robust, pre-frail, frail)
    3. Mortality rate by frailty score bins
    4. Patient demographics and baseline characteristics
    
    Parameters:
        dataset_dir (str, optional): Path to INSPIRE dataset directory.
                                    If None, uses get_dataset_dir() to find it.
    
    Returns:
        pd.DataFrame: Operations dataframe with 30-day outcomes and frailty scores added
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    
    stopwatch = Stopwatch()
    print("\n" + "="*80)
    print("30-Day Frailty Analysis")
    print("="*80)
    
    # Load operations data
    operations_path = os.path.join(dataset_dir, "operations.csv")
    diagnosis_path = os.path.join(dataset_dir, "diagnosis.csv")
    
    print(f"\nLoading operations from: {operations_path}")
    operations_df = pd.read_csv(operations_path)
    
    print(f"Loading diagnosis from: {diagnosis_path}")
    diagnosis_df = pd.read_csv(diagnosis_path)
    
    # Define 30-day threshold (in minutes, assuming chart_time is in minutes)
    thirty_days_minutes = 30 * 24 * 60
    
    # Calculate outcomes at 30 days
    operations_df['allcause_death_time'] = pd.to_numeric(
        operations_df['allcause_death_time'], errors='coerce'
    )
    
    # Create binary outcome: died within 30 days
    operations_df['died_30day'] = (
        (operations_df['allcause_death_time'].notna()) & 
        (operations_df['allcause_death_time'] <= thirty_days_minutes)
    ).astype(int)
    
    # Calculate frailty scores for each patient using diagnosis data
    operations_df['subject_id'] = pd.to_numeric(
        operations_df['subject_id'], errors='coerce'
    )
    diagnosis_df['subject_id'] = pd.to_numeric(
        diagnosis_df['subject_id'], errors='coerce'
    )
    
    # Get ICD-10 codes for each subject
    subjects_diagnoses = diagnosis_df.groupby('subject_id')['icd10_cm'].apply(list).to_dict()
    
    print(f"\nCalculating frailty scores for {len(operations_df)} operations...")
    
    frailty_scores = []
    frailty_categories = []
    
    for idx, row in operations_df.iterrows():
        subject_id = row['subject_id']
        
        # Create Subject object with diagnoses
        subject = Subject()
        if subject_id in subjects_diagnoses:
            subject.diagnoses = subjects_diagnoses[subject_id]
        
        # Calculate HFRS
        frailty_score, frailty_category = subject_module.compute_hfrs(subject)
        frailty_scores.append(frailty_score)
        frailty_categories.append(frailty_category)
        
        if (idx + 1) % 20000 == 0:
            print(f"  Processed {idx + 1}/{len(operations_df)} operations...")
    
    operations_df['frailty_score'] = frailty_scores
    operations_df['frailty_category'] = frailty_categories
    
    # Remove rows with NaN frailty scores
    operations_df = operations_df.dropna(subset=['frailty_score'])
    
    # Separate survivors and non-survivors at 30 days
    survived_30 = operations_df[operations_df['died_30day'] == 0]
    died_30 = operations_df[operations_df['died_30day'] == 1]
    
    print(f"\n30-Day Outcomes:")
    print(f"  Total operations: {len(operations_df)}")
    print(f"  Survived 30 days: {len(survived_30)} ({100*len(survived_30)/len(operations_df):.1f}%)")
    print(f"  Died within 30 days: {len(died_30)} ({100*len(died_30)/len(operations_df):.1f}%)")
    print(f"\nFrailty Scores - Survived 30 days:")
    print(f"  Mean: {survived_30['frailty_score'].mean():.2f}")
    print(f"  Median: {survived_30['frailty_score'].median():.2f}")
    print(f"  Std: {survived_30['frailty_score'].std():.2f}")
    print(f"\nFrailty Scores - Died within 30 days:")
    print(f"  Mean: {died_30['frailty_score'].mean():.2f}")
    print(f"  Median: {died_30['frailty_score'].median():.2f}")
    print(f"  Std: {died_30['frailty_score'].std():.2f}")
    
    # --- Plot 1: Frailty Score Distribution ---
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(survived_30['frailty_score'], bins=50, alpha=0.6, label='Survived 30 days', color='green', edgecolor='black')
    plt.hist(died_30['frailty_score'], bins=50, alpha=0.6, label='Died within 30 days', color='red', edgecolor='black')
    plt.xlabel('Frailty Score')
    plt.ylabel('Frequency')
    plt.title('Frailty Score Distribution: 30-Day Outcomes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- Plot 2: Box plot comparison ---
    plt.subplot(2, 2, 2)
    data_to_plot = [survived_30['frailty_score'], died_30['frailty_score']]
    bp = plt.boxplot(data_to_plot, labels=['Survived 30 days', 'Died within 30 days'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Frailty Score')
    plt.title('Frailty Score Distribution by 30-Day Outcome')
    plt.grid(True, alpha=0.3)
    
    # --- Plot 3: Frailty Category Comparison ---
    plt.subplot(2, 2, 3)
    category_counts = pd.DataFrame({
        'Survived': survived_30['frailty_category'].value_counts(),
        'Died': died_30['frailty_category'].value_counts()
    }).fillna(0)
    category_counts.plot(kind='bar', ax=plt.gca(), color=['green', 'red'], alpha=0.7)
    plt.xlabel('Frailty Category')
    plt.ylabel('Number of Patients')
    plt.title('30-Day Outcomes by Frailty Category')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 4: Mortality Rate by Frailty Score Bins ---
    plt.subplot(2, 2, 4)
    # Create frailty score bins
    bins = [0, 2, 4, 6, 8, 10, 15]
    operations_df['frailty_bin'] = pd.cut(operations_df['frailty_score'], bins=bins)
    
    mortality_by_bin = operations_df.groupby('frailty_bin', observed=True).apply(
        lambda x: (x['died_30day'].sum() / len(x) * 100) if len(x) > 0 else 0
    )
    count_by_bin = operations_df.groupby('frailty_bin', observed=True).size()
    
    x_pos = np.arange(len(mortality_by_bin))
    bars = plt.bar(x_pos, mortality_by_bin.values, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, count_by_bin.values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'n={int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Frailty Score Range')
    plt.ylabel('30-Day Mortality Rate (%)')
    plt.title('30-Day Mortality Rate by Frailty Score')
    plt.xticks(x_pos, [f'{int(b.left)}-{int(b.right)}' for b in mortality_by_bin.index], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("fig_frailty_30day_outcomes.pdf", dpi=300)
    print(f"\n✓ Saved figure: fig_frailty_30day_outcomes.pdf")
    plt.show()
    
    # Print summary statistics by frailty category
    print(f"\n30-Day Mortality by Frailty Category:")
    for category in ['Robust', 'Pre-frail', 'Frail']:
        subset = operations_df[operations_df['frailty_category'] == category]
        if len(subset) > 0:
            mortality = subset['died_30day'].sum() / len(subset) * 100
            print(f"  {category}: {mortality:.1f}% ({subset['died_30day'].sum()}/{len(subset)})")
    
    print(f"\nAnalysis completed in {stopwatch.elapsedTime()/60:.2f} minutes")
    return operations_df


def main():
    """
    Main entry point for 30-day frailty analysis.
    """
    print("Starting 30-Day Frailty Analysis...")
    
    try:
        results_df = analyse_30day_frailty_outcomes()
        print("\n✓ Analysis completed successfully!")
        return results_df
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
