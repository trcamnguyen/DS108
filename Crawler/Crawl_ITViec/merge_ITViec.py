import pandas as pd
import os

def merge_itviec_files():
    data_dir = "data"

    file1_path = os.path.join(data_dir, "itviec_person1.csv")
    file2_path = os.path.join(data_dir, "itviec_person2.csv")
    out_dir = os.path.join("..", "..", "data", "raw")
    out_path = os.path.join(out_dir, "00-itviec_raw.csv")

    if not os.path.exists(file1_path):
        print(f"Không tìm thấy file: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"Không tìm thấy file: {file2_path}")
        return

    os.makedirs(out_dir, exist_ok=True)

    print(f"Đang đọc file 1: {file1_path}")
    df1 = pd.read_csv(file1_path)
    print(f"   → Person1: {df1.shape[0]} records")

    print(f"Đang đọc file 2: {file2_path}")
    df2 = pd.read_csv(file2_path)
    print(f"   → Person2: {df2.shape[0]} records")

    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"   → Tổng sau khi gộp: {merged_df.shape[0]} records")

    merged_df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    print(f"   → Sau khi xóa trùng URL: {merged_df.shape[0]} records")

    # Loại bỏ các dòng lỗi: tag/listing page bị crawl nhầm (job_title rỗng)
    before = merged_df.shape[0]
    merged_df = merged_df[merged_df["job_title"].notna() & (merged_df["job_title"].str.strip() != "")]
    dropped = before - merged_df.shape[0]
    if dropped:
        print(f"   → Loại bỏ {dropped} dòng lỗi (job_title rỗng — tag listing page bị crawl nhầm)")

    merged_df = merged_df.sort_values(by="url").reset_index(drop=True)

    merged_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nMerge hoàn tất!")
    print(f"Output: {out_path}")
    print(f"Tổng số việc làm hợp lệ: {merged_df.shape[0]}")

if __name__ == "__main__":
    merge_itviec_files()
