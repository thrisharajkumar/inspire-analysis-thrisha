#!/usr/bin/env python3
"""
Example script showing how to run the DNN mortality pipeline with REAL INSPIRE data.
This replaces synthetic data generation with actual patient data.
"""

import sys
import os

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dnn_mortality_pipeline import main


def main_real_data():
    """
    Run the DNN mortality prediction pipeline with REAL INSPIRE dataset.
    
    Update the dataset_dir path to point to your INSPIRE data:
    - Option 1: Path containing CSV files (operations.csv, vitals.csv, labs.csv, ward_vitals.csv)
    - Option 2: Path containing 'died' and 'survived' subdirectories with JSON files
    """
    
    # ===== UPDATE THIS PATH TO YOUR DATASET =====
    # Example paths:
    # dataset_dir = "C:\\Users\\pc\\Documents\\INSPIRE_Data"
    # dataset_dir = "D:\\datasets\\inspire_subjects"
    # dataset_dir = "../inspire_dataset"
    
    dataset_dir = None  # Use this to search default locations automatically
    
    # UNCOMMENT AND SET your actual path:
    # dataset_dir = r"C:\path\to\your\inspire\dataset"
    
    print("=" * 80)
    print("DNN Mortality Prediction Pipeline - REAL DATA VERSION")
    print("=" * 80)
    print()
    
    if dataset_dir is None:
        print("Dataset directory not set!")
        print()
        print("To use REAL data, edit this script and update the 'dataset_dir' variable:")
        print()
        print("  dataset_dir = r'C:\\path\\to\\inspire\\dataset'")
        print()
        print("Your dataset should contain either:")
        print("  1. CSV files: operations.csv, vitals.csv, labs.csv, ward_vitals.csv")
        print("  2. Subdirectories: 'died/' and 'survived/' with JSON files")
        print()
        print("See SWITCH_TO_REAL_DATA.md for detailed instructions.")
        print()
        return
    
    try:
        # Run the main pipeline with real data
        main(dataset_dir=dataset_dir)
        
        print()
        print("=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        
    except ValueError as e:
        print(f"ERROR: {e}")
        print()
        print("Common solutions:")
        print("1. Check that the dataset directory path is correct")
        print("2. Verify that required CSV files exist in the directory")
        print("3. Ensure file names are exactly: operations.csv, vitals.csv, labs.csv, ward_vitals.csv")
        print()
        print("See SWITCH_TO_REAL_DATA.md for more help.")
        
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_real_data()
