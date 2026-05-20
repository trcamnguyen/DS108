import os
import json
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# =============================
# CONFIG
# =============================
TAG_INPUT_FILE = "itviec_tags/itviec_tags.txt"
JOB_LINK_DIR = "job_links"
JOB_LINKS_CSV = os.path.join(JOB_LINK_DIR, "itviec_job_links.csv")
JOB_LINKS_TXT = os.path.join(JOB_LINK_DIR, "itviec_job_links.txt")
PROGRESS_FILE = os.path.join(JOB_LINK_DIR, "itviec_tags_progress.json")

os.makedirs(JOB_LINK_DIR, exist_ok=True)

# =============================
# SESSION
# =============================
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer": "https://itviec.com/"
}

session = requests.Session()
session.headers.update(headers)

# =============================
# UTILS
# =============================
def safe_get(url, retries=5):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=25)
            if response.status_code == 403:
                wait = random.uniform(60, 120)
                print(f"403 Forbidden at {url} - sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries - 1:
                raise e
            wait = random.uniform(5, 10)
            print(f"Retry {attempt + 1} after error: {e}")
            time.sleep(wait)
    return None
# Hàm này tự động phát hiện số trang thực tế dựa trên Pagination
def get_actual_max_page(soup):
    try:
        pagination = soup.select(".ipagination .page:not(.next):not(.gap)")
        if pagination:
            pages = [int(p.get_text(strip=True)) for p in pagination if p.get_text(strip=True).isdigit()]
            return max(pages) if pages else 1
    except:
        pass
    return 1

# =============================
# MAIN LOGIC
# =============================
def crawl_itviec_links():
    if not os.path.exists(TAG_INPUT_FILE):
        print(f"Error: {TAG_INPUT_FILE} not found")
        return

    with open(TAG_INPUT_FILE, "r", encoding="utf-8") as f:
        tags = [line.strip() for line in f if line.strip()]

    crawled_tags = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            crawled_tags = json.load(f)

    all_links = set()
    if os.path.exists(JOB_LINKS_CSV):
        old_df = pd.read_csv(JOB_LINKS_CSV)
        all_links = set(old_df['url'].tolist())

    print("Starting ITviec Link Scraper...")

    for tag in tags:
        if tag in crawled_tags:
            continue
        
        print(f"Processing Tag: {tag}")
        base_url = f"https://itviec.com/it-jobs/{tag}"
        
        try:
            response = safe_get(base_url)
            if not response: continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            max_page = get_actual_max_page(soup)
            
            for page in range(1, max_page + 1):
                page_url = f"{base_url}?page={page}"
                p_resp = safe_get(page_url)
                if not p_resp: continue
                
                p_soup = BeautifulSoup(p_resp.text, "html.parser")
                page_links = []
                
                for h3 in p_soup.select("h3[data-url]"):
                    raw_url = h3.get("data-url")
                    if raw_url:
                        page_links.append(raw_url.split("?")[0])
                
                for a in p_soup.select("a.stretched-link[href*='/it-jobs/']"):
                    full_url = urljoin("https://itviec.com", a.get("href"))
                    page_links.append(full_url.split("?")[0])
                # Cập nhật vào Set để tự động loại bỏ trùng lặp
                all_links.update(page_links)
                print(f"  Page {page}/{max_page}: Collected links")
                time.sleep(random.uniform(2, 4))

            crawled_tags.append(tag)
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(crawled_tags, f)
            # Lưu file CSV sau mỗi tag để tránh mất dữ liệu khi có lỗi
            pd.DataFrame({"url": list(all_links)}).to_csv(JOB_LINKS_CSV, index=False, encoding="utf-8-sig")
            
        except Exception as e:
            print(f"Error at tag {tag}: {e}")
        
        time.sleep(random.uniform(10, 15))
        
    # =============================
    # EXPORT & SPLIT
    # =============================
    final_links = sorted(list(all_links))
    
    with open(JOB_LINKS_TXT, "w", encoding="utf-8") as f:
        for link in final_links:
            f.write(link + "\n")

    mid = len(final_links) // 2
    person1_links = final_links[:mid]
    person2_links = final_links[mid:]

    pd.DataFrame({"url": person1_links}).to_csv(os.path.join(JOB_LINK_DIR, "itviec_job_links_person1.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"url": person2_links}).to_csv(os.path.join(JOB_LINK_DIR, "itviec_job_links_person2.csv"), index=False, encoding="utf-8-sig")
    
    print("-" * 30)
    print(f"Total unique job links: {len(final_links)}")
    print(f"Person 1: {len(person1_links)} links")
    print(f"Person 2: {len(person2_links)} links")
    print("Finished. Data split and saved.")

if __name__ == "__main__":
    crawl_itviec_links()