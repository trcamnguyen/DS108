import pandas as pd
import os

def merge_csv_files():
    data_dir = r"C:\Users\tncn2\Documents\HK2-2026_2027\DS108\DS108\Crawl\data"
    
    file1_path = os.path.join(data_dir, "topcv_person1.csv")
    file2_path = os.path.join(data_dir, "topcv_person2.csv")
    out_path = r"C:\Users\tncn2\Documents\HK2-2026_2027\DS108\DS108\data\raw\00-topcv_raw.csv"

    print(f"Reading file: {file1_path}")
    df1 = pd.read_csv(file1_path)
    print(f"Number of records in file 1: {df1.shape[0]}")

    print(f"Reading file: {file2_path}")
    df2 = pd.read_csv(file2_path)
    print(f"Number of records in file 2: {df2.shape[0]}")

    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Total records after merging: {merged_df.shape[0]}")

    # Remove duplicated rows based on url
    merged_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    print(f"Total records after removing duplicates: {merged_df.shape[0]}")

    # Save to out_path
    merged_df.to_csv(out_path, index=False, encoding='utf-8-sig') 
    print(f"\n=> Successfully saved merged data to: {out_path}")

if __name__ == "__main__":
    merge_csv_files()
