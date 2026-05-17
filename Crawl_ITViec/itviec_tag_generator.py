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
TAG_DIR = "itviec_tags"
TAG_TXT_FILE = os.path.join(TAG_DIR, "itviec_tags.txt")
TAG_JSON_FILE = os.path.join(TAG_DIR, "itviec_structured_tags.json")

os.makedirs(TAG_DIR, exist_ok=True)

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
                wait = random.uniform(30, 60)
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

# =============================
# MAIN TAG GENERATOR
# =============================
def generate_itviec_tags():
    base_url = "https://itviec.com"
    indices = {
        "skill": "/jobs-skill-index",
        "expertise": "/jobs-expertise-index",
        "title": "/jobs-title-index",
        "company": "/jobs-company-index"
    }

    structured_tags = {k: [] for k in indices.keys()}
    all_unique_slugs = set()

    print("Starting ITviec Tag Generation...")

    for category, path in indices.items():
        url = base_url + path
        print(f"Accessing {category.upper()} Index: {url}")
        try:
            response = safe_get(url)
            if not response: continue
            soup = BeautifulSoup(response.text, "html.parser")
            
            # ITviec phân loại link công ty khác với link job
            prefix = "/companies/" if category == "company" else "/it-jobs/"
            
            # Tìm tất cả link có chứa prefix tương ứng
            found_links = soup.select(f"a[href*='{prefix}']")
            
            category_list = []
            for a in found_links:
                href = a.get("href", "")
                slug = href.split("?")[0].replace(prefix, "").strip("/")
                if slug and "/" not in slug and len(slug) < 50:
                    if slug not in ["vi", "en", "it-jobs", "companies"]:
                        category_list.append(slug)
                        all_unique_slugs.add(slug)
            
            structured_tags[category] = sorted(list(set(category_list)))
            print(f"  Found {len(structured_tags[category])} unique tags for {category}")
            time.sleep(random.uniform(3, 6))

        except Exception as e:
            print(f"Error at {category}: {e}")

    # =============================
    # EXPORT FILES
    # =============================
    # 1. Lưu file JSON 
    with open(TAG_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(structured_tags, f, ensure_ascii=False, indent=2)
    
    # 2. Lưu file TXT  
    sorted_all_tags = sorted(list(all_unique_slugs))
    with open(TAG_TXT_FILE, "w", encoding="utf-8") as f:
        for tag in sorted_all_tags:
            f.write(tag + "\n")
            
    print("-" * 30)
    print(f"Finished generating tags.")
    print(f"Total unique slugs: {len(sorted_all_tags)}")
    print(f"Output files: {TAG_TXT_FILE}, {TAG_JSON_FILE}")

if __name__ == "__main__":
    generate_itviec_tags()