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
    "Cookie": """NHẬP_COOKIE_CỦA_BẠN_TẠI_ĐÂY"""
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
    salary_el = soup.select_one(".salary .ips-2")
    salary_val = clean_text(salary_el) if salary_el else clean_text(soup.select_one(".salary"))

    if not salary_val or "sign in" in salary_val.lower() or "love it" in salary_val.lower():
        salary_final = "Thỏa thuận"
    else:
        salary_final = salary_val

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
    # SKILLS & SPECIALIZATION
    # =============================
    req_skills = []
    specialization = []
    
    for div in soup.select(".imb-4, .imb-3"):
        label_el = div.select_one(".fw-600")
        if not label_el:
            continue
        label_text = clean_text(label_el).lower()
        tags = [clean_text(a) for a in div.select(".itag")]
        if "skills" in label_text:
            req_skills.extend(tags)
        elif "expertise" in label_text:
            specialization.extend(tags)

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
            if "your skills and experience" not in h2_text:
                continue
            current_type = "preferred"
            for child in div.find_all(recursive=False):
                if child.name == "h2":
                    continue
                child_text = clean_text(child)
                if not child_text:
                    continue
                lower_text = child_text.lower()
                is_experience = any(
                    keyword in lower_text
                    for keyword in EXPERIENCE_KEYWORDS
                )
                is_requirement = any(
                    keyword in lower_text
                    for keyword in REQUIREMENT_KEYWORDS
                )
                if len(lower_text.split()) <= 12:
                    if is_experience:
                        current_type = "experience"
                        continue
                    elif is_requirement:
                        current_type = "requirement"
                        continue
                if current_type == "experience":
                    exp_content += child_text + "\n"
                elif current_type == "requirement":
                    req_content += child_text + "\n"
                else:
                    preferred_content += child_text + "\n"
    return {
        "url": job_url,
        "job_title": job_title,
        "company": company,
        "location": location_final,
        "salary": salary_final,
        "experience": exp_content.strip(),
        "education": info.get("education", ""),
        "level": "",
        "employment_type": "",
        "industry": info.get("company industry", ""),
        "required_skills": ", ".join(list(dict.fromkeys(req_skills))),
        "preferred_skills": preferred_content.strip(),
        "specialization": ", ".join(list(dict.fromkeys(specialization))),
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
            print(f"[{idx}/{len(remaining_links)}] Processing...")
            data = parse_job_detail(link)
            if data:
                all_jobs.append(data)
                crawled_urls.add(link)
            
            if idx % 10 == 0:
                pd.DataFrame(all_jobs).to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
                with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(crawled_urls), f)
                print(f"Checkpoint saved ({len(crawled_urls)} crawled)")
            time.sleep(random.uniform(4.0, 8.0))
        except Exception as e:
            print(f"Error with {link}: {e}")
            failed_urls.append({"url": link, "error": str(e)})

    # Final Save
    if all_jobs:
        df = pd.DataFrame(all_jobs).drop_duplicates(subset=["url"])
        df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        df.to_json(JSON_FILE, orient="records", force_ascii=False, indent=2)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(crawled_urls), f)
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(failed_urls, f)
    print("Done.")