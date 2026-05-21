# DS108 — Data Crawlers

Thu thập tin tuyển dụng IT từ hai nguồn: **TopCV** và **ITViec**.

---

## Cấu trúc thư mục

```
Crawler/
├── Crawl_TopCV/
│   ├── job_links/
│   │   └── job_links_scraper.py     # Bước 1: Thu thập link
│   ├── crawler.py                   # Bước 2: Crawl chi tiết job
│   └── merge_topcv_data.py
│
└── Crawl_ITViec/
    ├── job_links_scraper.py         # Bước 1: Thu thập link theo tag
    ├── itviec_tag_generator.py      # Sinh danh sách tag IT
    ├── crawler.py                   # Bước 2: Crawl chi tiết job
    └── salary_recrawl.py            # Bước 3: Re-crawl lương bị ẩn
```

---

## TopCV

### Bước 1 — Thu thập link

```bash
cd Crawl_TopCV/job_links
python job_links_scraper.py
```

Duyệt qua 89 trang danh sách ngành CNTT (`/tim-viec-lam-cong-nghe-thong-tin-cr257`), trích xuất URL từ `h3.title a`. Kết quả được chia đôi thành `topcv_job_links_person1.csv` và `topcv_job_links_person2.csv` để 2 thành viên crawl song song.

### Bước 2 — Crawl chi tiết

```bash
cd Crawl_TopCV
python crawler.py person1
python crawler.py person2   # chạy song song trên máy khác
```

Đọc link từ CSV của từng người, gọi `requests` + `BeautifulSoup` để parse từng trang job. Các trường thu thập: `job_title`, `company`, `location`, `salary`, `experience`, `education`, `level`, `employment_type`, `industry`, `required_skills`, `preferred_skills`, `specialization`, `job_description`, `requirement`.

**Xử lý lỗi:** retry 3 lần, sleep ngẫu nhiên 1.5–3s giữa mỗi job, sleep 10–20s khi gặp 403. Checkpoint mỗi 10 job. Kết quả lưu vào `data/topcv_{person}.csv`.

---

## ITViec

### Điều kiện tiên quyết

ITViec yêu cầu đăng nhập để xem lương. Cập nhật `Cookie` trong `crawler.py` và `salary_recrawl.py` trước khi chạy.

### Bước 1 — Thu thập link theo tag

```bash
cd Crawl_ITViec
python job_links_scraper.py
```

Đọc danh sách tag IT từ `itviec_tags/itviec_tags.txt` (ví dụ: `python`, `react`, `nodejs`...). Với mỗi tag, duyệt qua tất cả các trang của `itviec.com/it-jobs/{tag}` và trích xuất URL job (nhận diện qua pattern kết thúc bằng `-NNNN`). Tự phát hiện số trang thực qua pagination. Link được lưu vào `job_links/itviec_job_links.csv` và chia đôi cho 2 người.

### Bước 2 — Crawl chi tiết

```bash
python crawler.py person1
python crawler.py person2
```

Parse từng trang job: header (`job_title`, `company`), `salary` (phát hiện nếu bị redirect về trang login), `location`, `required_skills` (tag kỹ năng), sidebar (`education`, `industry`), và nội dung từ section **"Your Skills and Experience"** → lưu vào cột `requirement`.

Sleep 4–8s giữa mỗi job (chậm hơn TopCV do anti-bot). Phát hiện cookie hết hạn qua `.sign-in-page` selector.

### Bước 3 — Re-crawl lương bị ẩn

```bash
python salary_recrawl.py all
```

Với các hàng còn "Thỏa thuận", crawl lại trang để kiểm tra lương có hiển thị sau khi đã đăng nhập không. Chỉ cập nhật các hàng thay đổi thực sự. Ghi log vào `salary_recrawl_{person}_changes.csv`.

---

## Output

| File | Nguồn | Mô tả |
|------|-------|-------|
| `Crawl_TopCV/data/topcv_person1.csv` | TopCV | Kết quả crawl person 1 |
| `Crawl_TopCV/data/topcv_person2.csv` | TopCV | Kết quả crawl person 2 |
| `Crawl_ITViec/data/itviec_person1.csv` | ITViec | Kết quả crawl person 1 |
| `Crawl_ITViec/data/itviec_person2.csv` | ITViec | Kết quả crawl person 2 |
| `Crawl_ITViec/data/itviec_merged.csv` | ITViec | Đã merge 2 file trên |

---

## Lưu ý

- **Không commit cookie** vào git — chỉ điền trực tiếp vào file trước khi chạy.
- Nếu bị dừng giữa chừng, chạy lại cùng lệnh — crawler tự đọc progress file và bỏ qua URL đã crawl.
- Sau khi cả 2 người crawl xong TopCV, chạy `merge_topcv_data.py` để gộp file.
