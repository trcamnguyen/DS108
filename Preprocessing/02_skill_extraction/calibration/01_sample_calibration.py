import pandas as pd
import numpy as np
import os

def generate_calibration_set(input_csv_path, output_csv_path, random_seed=42):
    """
    Generate a calibration dataset of 50 samples from the original dataset.
    Rules:
    - Jobs (standardized_title) with >= 10 samples in the dataset get 1 sample.
    - The rest (to reach 50 samples) will be randomly chosen from jobs with < 10 samples (1 sample each).
    """
    print(f"Reading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # Count occurrences of each standardized_title
    title_counts = df['standardized_title'].value_counts()
    
    # Filter titles with >= 10 rows
    major_titles = title_counts[title_counts >= 10].index
    print(f"Number of job groups with >= 10 samples: {len(major_titles)}")
    
    # Randomly select 1 sample for each of these major titles
    major_samples = df[df['standardized_title'].isin(major_titles)].groupby('standardized_title').sample(n=1, random_state=random_seed)
    
    # Calculate how many more samples we need to reach 50
    remaining_needed = 50 - len(major_samples)
    print(f"Number of additional samples needed to reach 50: {remaining_needed}")
    
    if remaining_needed > 0:
        # Get the remaining titles (under 10 rows)
        minor_titles = title_counts[title_counts < 10].index
        
        # Randomly choose these minor titles to sample from
        sample_size = min(remaining_needed, len(minor_titles))
        np.random.seed(random_seed)
        selected_minor_titles = np.random.choice(minor_titles, size=sample_size, replace=False)
        
        # Get 1 sample for each selected minor title
        minor_samples = df[df['standardized_title'].isin(selected_minor_titles)].groupby('standardized_title').sample(n=1, random_state=random_seed)
        
        # Combine the two datasets
        calibration_df = pd.concat([major_samples, minor_samples])
    else:
        # If the number of major_titles >= 50, we just take exactly 50 from the major set
        calibration_df = major_samples.sample(n=50, random_state=random_seed)
        
    # Save to CSV
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    calibration_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"\nDone! Calibration dataset with {len(calibration_df)} samples has been saved to: {output_csv_path}")
    return calibration_df

if __name__ == "__main__":
    # The script runs in the 02-skill_extraction directory, so we point to the data/interim dir
    input_path = '../../../data/interim/02-topcv_job_filtered.csv'
    output_path = 'calibration_dataset.csv'
    
    generate_calibration_set(input_path, output_path)
