import pandas as pd
import os

def main():
    input_file = "output/few_shot_parsed.csv"
    output_file = "output/few_shot_parsed_debug.csv"
    
    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'")
        return
        
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Columns to drop
    cols_to_drop = ["source_text", "min_years", "level", "mode", "row_id"]
    
    # Drop columns that exist in the dataframe to avoid KeyError
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    df_dropped = df.drop(columns=existing_cols_to_drop)
    
    # Save to new file
    df_dropped.to_csv(output_file, index=False)
    
    print(f"Successfully dropped columns: {', '.join(existing_cols_to_drop)}")
    print(f"Remaining columns: {', '.join(df_dropped.columns)}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
