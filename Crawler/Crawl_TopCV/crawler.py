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
# CSV_FILE = os.path.join(DATA_DIR, "topcv_person1.csv")
# JSON_FILE = os.path.join(DATA_DIR, "topcv_person1.json")
# PROGRESS_FILE = os.path.join(DATA_DIR, "topcv_person1_crawled_urls.json")
# FAILED_FILE = os.path.join(DATA_DIR, "topcv_person1_failed_urls.json")

person = sys.argv[1]

LINK_FILE = f"job_links/topcv_job_links_{person}.csv"
CSV_FILE = f"data/topcv_{person}.csv"
JSON_FILE = f"data/topcv_{person}.json"
PROGRESS_FILE = f"data/topcv_{person}_crawled_urls.txt"
FAILED_FILE = f"data/topcv_{person}_failed_urls.txt"

os.makedirs(DATA_DIR, exist_ok=True)

# =============================
# SESSION
# =============================
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer": "https://www.topcv.vn/",
    "Connection": "keep-alive"
}

session = requests.Session()
session.headers.update(headers)

# =============================
# UTILS
# =============================
def clean_text(element):
    if not element:
        return None

    if hasattr(element, "get_text"):
        text = element.get_text(" ", strip=True)
    else:
        text = str(element)

    return " ".join(text.split())


def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=20)

            if response.status_code == 403:
                wait = random.uniform(10, 20)
                print(f"403 detected -> sleep {wait:.1f}s")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except Exception as e:
            if attempt == retries - 1:
                raise e

            wait = random.uniform(5, 10)
            print(f"Retry after error: {e}")
            time.sleep(wait)


# =============================
# EXTRACTORS
# =============================
def extract_info_by_title(soup, target_titles):
    if isinstance(target_titles, str):
        target_titles = [target_titles]

    target_titles = [t.strip().lower() for t in target_titles]

    for block in soup.select(".job-detail__info--section-content"):
        title = clean_text(
            block.select_one(".job-detail__info--section-content-title")
        )

        if title and title.strip().lower() in target_titles:
            return clean_text(
                block.select_one(".job-detail__info--section-content-value")
            )

    return None


def extract_general_info_by_title(soup, target_title):
    for block in soup.select(".box-general-group-info"):
        title = clean_text(
            block.select_one(".box-general-group-info-title")
        )

        if title == target_title:
            return clean_text(
                block.select_one(".box-general-group-info-value")
            )

    return None


def extract_company_field(soup, target_title):
    for block in soup.select(".job-detail__company--information-item"):
        title = clean_text(block.select_one(".company-title"))

        if title and target_title in title:
            return clean_text(block.select_one(".company-value"))

    return None


def extract_skills_by_box_title(soup, target_title):
    for block in soup.select(".box-category"):
        title = clean_text(block.select_one(".box-title"))

        if title == target_title:
            skills = []

            for tag in block.select(".box-category-tag"):
                skill = clean_text(tag)

                if skill and skill not in skills:
                    skills.append(skill)

            return skills

    return []


def extract_specializations(soup):
    tags = []

    for tag in soup.select(".job-tags__group-list-tag-scroll a.item"):
        value = clean_text(tag)

        if value and value not in tags:
            tags.append(value)

    return tags


# =============================
# DETAIL PARSER
# =============================
def parse_job_detail(job_url):
    print(f"  -> Crawling: {job_url}")

    response = safe_get(job_url)
    soup = BeautifulSoup(response.text, "html.parser")

    return {
        "url": job_url,
        "job_title": clean_text(
            soup.select_one("h1.job-detail__info--title")
            or soup.select_one("h1")
        ),
        "company": clean_text(
            soup.select_one(".job-detail__info--company a")
            or soup.select_one(".company-name")
        ),
        "location": clean_text(
            soup.select_one('a[href*="tim-viec-lam"][title*="tại"]')
        ),
        "salary": extract_info_by_title(
            soup,
            ["Thu nhập", "Mức lương", "Lương"]
        ),
        "experience": extract_info_by_title(soup, "Kinh nghiệm"),
        "education": extract_general_info_by_title(soup, "Học vấn"),
        "level": extract_general_info_by_title(soup, "Cấp bậc"),
        "employment_type": extract_general_info_by_title(
            soup,
            "Hình thức làm việc"
        ),
        "industry": extract_company_field(soup, "Lĩnh vực"),
        "required_skills": extract_skills_by_box_title(
            soup,
            "Kỹ năng cần có"
        ),
        "preferred_skills": extract_skills_by_box_title(
            soup,
            "Kỹ năng nên có"
        ),
        "specialization": extract_specializations(soup),
        "job_description": clean_text(
            soup.select_one(
                ".job-description__item:not(.requirement) "
                ".job-description__item--content"
            )
        ),
        "requirement": clean_text(
            soup.select_one(
                ".job-description__item.requirement "
                ".job-description__item--content"
            )
        )
    }


# =============================
# LOAD INPUT LINKS
# =============================
links_df = pd.read_csv(LINK_FILE)
all_links = links_df["url"].dropna().tolist()

# =============================
# LOAD EXISTING PROGRESS
# =============================
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        crawled_urls = set(json.load(f))
else:
    crawled_urls = set()

# load failed urls nếu đã có
if os.path.exists(FAILED_FILE):
    with open(FAILED_FILE, "r", encoding="utf-8") as f:
        failed_urls = json.load(f)
else:
    failed_urls = []

# nếu csv đã tồn tại thì load dữ liệu cũ để append tiếp
if os.path.exists(CSV_FILE):
    existing_df = pd.read_csv(CSV_FILE)
    all_jobs = existing_df.to_dict(orient="records")
else:
    all_jobs = []

remaining_links = [
    url for url in all_links
    if url not in crawled_urls
]

print(f"Total links in file      : {len(all_links)}")
print(f"Already crawled          : {len(crawled_urls)}")
print(f"Remaining to crawl       : {len(remaining_links)}")

# =============================
# MAIN LOOP
# =============================
for idx, link in enumerate(remaining_links, start=1):
    try:
        print(f"[{idx}/{len(remaining_links)}] {link}")

        job_data = parse_job_detail(link)
        all_jobs.append(job_data)

        crawled_urls.add(link)

        # save every 10 jobs
        if idx % 10 == 0:
            df = pd.DataFrame(all_jobs)

            df.to_csv(
                CSV_FILE,
                index=False,
                encoding="utf-8-sig"
            )

            with open(JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    all_jobs,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    sorted(list(crawled_urls)),
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                f"Checkpoint saved "
                f"({len(crawled_urls)} crawled)"
            )

        time.sleep(random.uniform(1.5, 3.0))

    except Exception as e:
        print(f"Error with {link}: {e}")

        failed_urls.append({
            "url": link,
            "error": str(e)
        })

        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(
                failed_urls,
                f,
                ensure_ascii=False,
                indent=2
            )

# =============================
# FINAL SAVE
# =============================
df = pd.DataFrame(all_jobs)

# tránh duplicate nếu từng crawl dở rồi chạy lại
df = df.drop_duplicates(subset=["url"])

df.to_csv(
    CSV_FILE,
    index=False,
    encoding="utf-8-sig"
)

df.to_json(
    JSON_FILE,
    orient="records",
    force_ascii=False,
    indent=2
)

with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        sorted(list(crawled_urls)),
        f,
        ensure_ascii=False,
        indent=2
    )

with open(FAILED_FILE, "w", encoding="utf-8") as f:
    json.dump(
        failed_urls,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nDone.")
print(f"Saved CSV  : {CSV_FILE}")
print(f"Saved JSON : {JSON_FILE}")
print(f"Progress   : {PROGRESS_FILE}")
print(f"Failed     : {FAILED_FILE}")