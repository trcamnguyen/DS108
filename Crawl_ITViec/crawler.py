import os
import sys
import json
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

# =============================
# CONFIG 
# =============================
DATA_DIR = "data"
person = sys.argv[1]

LINK_FILE = f"job_links/itviec_job_links_{person}.csv"
CSV_FILE = f"data/itviec_{person}.csv"
JSON_FILE = f"data/itviec_{person}.json"
PROGRESS_FILE = f"data/itviec_{person}_crawled_urls.txt"
FAILED_FILE = f"data/itviec_{person}_failed_urls.txt"

os.makedirs(DATA_DIR, exist_ok=True)

# =============================
# SESSION
# =============================
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer": "https://itviec.com/",
    "Connection": "keep-alive",
    "Cookie": """Enter_your_cookie_here"""
}
session = requests.Session()
session.headers.update(headers)

# =============================
# UTILS
# =============================
def clean_text(element):
    if not element: return ""
    text = element.get_text("\n", strip=True) if hasattr(element, "get_text") else str(element)
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())

def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=25)
            if response.status_code == 403:
                wait = random.uniform(60, 120)
                print(f"403 detected -> sleep {wait:.1f}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries - 1: raise e
            time.sleep(random.uniform(5, 10))
    return ""

# =============================
# KEYWORDS
# =============================
EXPERIENCE_KEYWORDS = [
    "experience", "experiences", "work experience", "working experience",
    "professional experience", "years of experience", "hands-on experience",
    "practical experience", "kinh nghiệm", "kinh nghiệm làm việc",
    "số năm kinh nghiệm",
]

REQUIREMENT_KEYWORDS = [
    "requirement", "requirements", "required", "must have", "must-have",
    "mandatory skills", "mandatory", "technical skills",
    "must have skills", "core skills", "qualification",
    "qualifications", "minimum qualifications", "yêu cầu",
    "yêu cầu công việc", "yêu cầu ứng viên", "kỹ năng cần có",
    "yêu cầu bắt buộc", "bắt buộc",
]

# =============================
# DETAIL PARSER
# =============================
def parse_job_detail(job_url):
    print(f"  -> Crawling: {job_url}")
    response = safe_get(job_url)
    if not response:
        return None
    soup = BeautifulSoup(response.text, "html.parser")

    # =============================
    # CHẶN TRANG TRUNG GIAN
    # =============================
    h1_el = soup.select_one("h1")
    h1_text = clean_text(h1_el).lower() if h1_el else ""

    if "jobs in" in h1_text or "việc làm" in h1_text:
        print(f"  [Skip] Intermediate page detected: {h1_text}")
        return None

    # =============================
    # HEADER
    # =============================
    header = soup.select_one(".job-header-info")
    job_title = clean_text(header.select_one("h1")) if header else ""
    company = clean_text(header.select_one(".employer-name")) if header else ""

    # =============================
    # SALARY
    # =============================
    # Bước 1: Thử lấy thẻ CHỨA SỐ trước (ips-2)
    salary_el = soup.select_one(".salary .ips-2") or soup.select_one(".ips-2")
    
    # Bước 2: Nếu KHÔNG THẤY thẻ số, mới lấy cái thẻ .salary bao quát
    if not salary_el:
        salary_el = soup.select_one(".salary")

    if salary_el:
        # Lấy nguyên văn
        salary_final = clean_text(salary_el).strip()
        
        if "sign in" in salary_final.lower() or "đăng nhập" in salary_final.lower():
            deep_check = salary_el.select_one(".ips-2")
            if deep_check:
                salary_final = clean_text(deep_check).strip()
    else:
        salary_final = "Not Found"

    # =============================
    # LOCATION
    # =============================
    location_list = []
    loc_spans = soup.select(".imb-3 .normal-text.text-rich-grey")
    exclude_list = ["posted", "at office", "hybrid", "remote"]

    for span in loc_spans:
        txt = clean_text(span)
        if txt and not any(ex in txt.lower() for ex in exclude_list):
            location_list.append(txt)
    location_final = " - ".join(location_list)

    # =============================
    # SKILLS 
    # =============================
    req_skills = []
    # specialization = []
    
    for label in soup.select(".fw-600"):
        label_text = clean_text(label).lower()
        if "skills" in label_text:
            parent_div = label.find_parent("div", class_="imb-4") or label.find_parent("div", class_="imb-3")
            if parent_div:
                tags = [clean_text(a) for a in parent_div.select("a.itag")]
                req_skills.extend(tags)
                break 

    # =============================
    # SIDEBAR INFO
    # =============================
    info = {}

    for row in soup.select(".job-show-employer-info .row"):
        lbl = clean_text(row.select_one(".col.text-dark-grey"))
        val = clean_text(row.select_one(".col.text-end"))
        if lbl and val:
            info[lbl.lower()] = val

    # =============================
    # CONTENT
    # =============================
    desc_content = ""
    req_content = ""
    exp_content = ""
    preferred_content = ""
    job_content = soup.select_one(".job-content")
    if job_content:

        # =============================
        # JOB DESCRIPTION
        # =============================
        for div in job_content.select(".paragraph"):
            h2 = div.find("h2")
            if not h2:
                continue
            h2_text = clean_text(h2).lower()
            if "job description" in h2_text:
                desc_content = "\n".join(
                    clean_text(child)
                    for child in div.find_all(recursive=False)
                    if child.name != "h2"
                )

        # =============================
        # YOUR SKILLS AND EXPERIENCE
        # =============================
        for div in job_content.select(".paragraph"):
            h2 = div.find("h2")
            if not h2:
                continue
            h2_text = clean_text(h2).lower()
            if "your skills and experience" in h2_text:
                # Cột requirement
                req_content = "\n".join(
                    clean_text(child)
                    for child in div.find_all(recursive=False)
                    if child.name != "h2"
                )
    return {
        "url": job_url,
        "job_title": job_title,
        "company": company,
        "location": location_final,
        "salary": salary_final,
        "experience": "",
        "education": info.get("education", ""),
        "level": "",
        "employment_type": "",
        "industry": info.get("company industry", ""),
        "required_skills": ", ".join(list(dict.fromkeys(req_skills))),
        "preferred_skills": "",
        "specialization": "",
        "job_description": desc_content.strip(),
        "requirement": req_content.strip()
    }

# =============================
# MAIN LOOP
# =============================
if __name__ == "__main__":
    try:
        links_df = pd.read_csv(LINK_FILE)
        all_links = links_df["url"].dropna().unique().tolist()
    except:
        print(f"Error: Link file {LINK_FILE} not found")
        sys.exit(1)

    crawled_urls = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            try: crawled_urls = set(json.load(f))
            except: crawled_urls = set()

    all_jobs = []
    if os.path.exists(CSV_FILE):
        try: all_jobs = pd.read_csv(CSV_FILE).to_dict(orient="records")
        except: all_jobs = []

    failed_urls = []
    remaining_links = [url for url in all_links if url not in crawled_urls]

    for idx, link in enumerate(remaining_links, start=1):
        try:
            exists = next((item for item in all_jobs if item["url"] == link), None)
            
            if exists:
                s_val = str(exists.get("salary", "")).lower()
                # CHỈ CÀO LẠI nếu lương chứa chữ "sign in" hoặc "đăng nhập"
                if "sign in" in s_val or "đăng nhập" in s_val:
                    print(f"[{idx}]  Re-crawling: {link} (Reason: {s_val})")
                    # Xóa dòng cũ bị lỗi lương 
                    all_jobs = [item for item in all_jobs if item["url"] != link]
                else:
                    # Nếu là con số hoặc "Thỏa thuận" thì bỏ qua
                    print(f"[{idx}] Skipping: {link}")
                    crawled_urls.add(link)
                    continue
            
            print(f"[{idx}/{len(remaining_links)}] Processing...")
            data = parse_job_detail(link)
            if data:
                all_jobs.append(data)
                crawled_urls.add(link)
            else:
                # NẾU LÀ LINK RÁC (Intermediate/Lỗi), cũng cho vào tiến độ để lần sau không cào lại
                crawled_urls.add(link)
                print(f"   [System] Marked invalid link as done to skip later.")
                time.sleep(0.5)
            if idx % 1 == 0:
                pd.DataFrame(all_jobs).to_csv(CSV_FILE, index=False, encoding="utf-8-sig",quoting=1)
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(crawled_urls), f)
                    f.flush()            # Ép dữ liệu ra khỏi bộ nhớ đệm Python
                    os.fsync(f.fileno()) # Ép dữ liệu xuống ổ cứng
                print(f"Checkpoint saved ({len(crawled_urls)} crawled)")
            time.sleep(random.uniform(4.0, 8.0))
        except Exception as e:
            print(f"Error with {link}: {e}")
            failed_urls.append({"url": link, "error": str(e)})

            # Nếu là lỗi 410 (Job hết hạn), đánh dấu để lần sau không cào lại nữa
            if "410" in str(e):
                crawled_urls.add(link)
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(crawled_urls), f)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"   [System] Marked Expired link (410) as done.")
                time.sleep(0.5)
    
    # Final Save
    if all_jobs:
        df = pd.DataFrame(all_jobs).drop_duplicates(subset=["url"])
        df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig", quoting=1)
        df.to_json(JSON_FILE, orient="records", force_ascii=False, indent=2)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(crawled_urls), f)
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(failed_urls, f)
    print("Done.")