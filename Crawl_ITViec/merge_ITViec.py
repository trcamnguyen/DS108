import pandas as pd
import os

def merge_itviec_files():
    data_dir = "data"  
    
    file1_path = os.path.join(data_dir, "itviec_person1.csv")
    file2_path = os.path.join(data_dir, "itviec_person2.csv")
    out_path = os.path.join(data_dir, "itviec_merged.csv")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(file1_path):
        print(f"Không tìm thấy file: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"Không tìm thấy file: {file2_path}")
        return

    print(f"Đang đọc file 1: {file1_path}")
    df1 = pd.read_csv(file1_path)
    print(f"   → Person1: {df1.shape[0]} records")

    print(f"Đang đọc file 2: {file2_path}")
    df2 = pd.read_csv(file2_path)
    print(f"   → Person2: {df2.shape[0]} records")

    # Merge hai file
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"   → Tổng sau khi gộp: {merged_df.shape[0]} records")

    # Xóa trùng lặp theo URL
    merged_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    print(f"   → Sau khi xóa trùng: {merged_df.shape[0]} records")

    # Sắp xếp lại theo thứ tự 
    merged_df = merged_df.sort_values(by='url').reset_index(drop=True)

    # Lưu file merged
    merged_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"\nMerge hoàn tất!")
    print(f"File tổng: {out_path}")
    print(f"Tổng số việc làm unique: {merged_df.shape[0]}")

if __name__ == "__main__":
    merge_itviec_files()