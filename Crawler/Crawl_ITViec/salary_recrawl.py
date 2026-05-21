"""
salary_recrawl.py — Re-crawl salary for rows that currently show "Thỏa thuận".
Chỉ fetch salary element, không re-parse toàn bộ job.

Usage:
    python salary_recrawl.py person1
    python salary_recrawl.py person2
    python salary_recrawl.py all
"""

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

NEGOTIABLE_VALUES = {
    "thỏa thuận", "thoa thuan", "negotiable", "competitive",
    "attractive", "let's discuss", "lets discuss", "best in the market",
}

CHECKPOINT_EVERY = 20

# =============================
# SESSION  — cập nhật cookie mới nếu cần
# =============================
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer": "https://itviec.com/",
    "Connection": "keep-alive",
    "Cookie": """Enter_your_cookies_here"""
}
session = requests.Session()
session.headers.update(headers)

# =============================
# UTILS
# =============================
def clean_text(element):
    if not element:
        return ""
    text = element.get_text("\n", strip=True) if hasattr(element, "get_text") else str(element)
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=25)
            if response.status_code == 403:
                wait = random.uniform(60, 120)
                print(f"  403 detected -> sleep {wait:.1f}s")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(random.uniform(5, 10))
    return None


def is_negotiable(salary_str):
    if not isinstance(salary_str, str) or salary_str.strip() == "":
        return True
    return salary_str.strip().lower() in NEGOTIABLE_VALUES


def fetch_salary(url):
    """Fetch only salary from a job detail page. Returns new salary string or None on error."""
    response = safe_get(url)
    if not response:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Detect cookie expired
    page_title = soup.title.get_text().lower() if soup.title else ""
    if soup.select_one(".sign-in-page") or "đăng nhập" in page_title or "sign in" in page_title:
        print(f"  [WARN] Cookie expired!")
        return None

    # Skip tag/listing pages
    if not soup.select_one(".job-header-info"):
        print(f"  [Skip] Not a job detail page")
        return None

    salary_el = soup.select_one(".salary .ips-2")
    salary_val = clean_text(salary_el) if salary_el else clean_text(soup.select_one(".salary"))

    if not salary_val or "sign in" in salary_val.lower() or "love it" in salary_val.lower():
        return "Thỏa thuận"

    return salary_val


# =============================
# MAIN
# =============================
def recrawl_salary(person):
    csv_path = os.path.join(DATA_DIR, f"itviec_{person}.csv")
    progress_path = os.path.join(DATA_DIR, f"salary_recrawl_{person}_done.json")
    log_path = os.path.join(DATA_DIR, f"salary_recrawl_{person}_changes.csv")

    df = pd.read_csv(csv_path)

    done_urls = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            done_urls = set(json.load(f))

    target_idx = df.index[df["salary"].apply(is_negotiable) & ~df["url"].isin(done_urls)]
    total = len(target_idx)
    print(f"\n[{person}] {total} rows to re-crawl (skipping {len(done_urls)} already done)\n")

    changes = []
    cookie_expired = False

    for count, idx in enumerate(target_idx, start=1):
        url = df.at[idx, "url"]
        old_salary = df.at[idx, "salary"]
        print(f"  [{count}/{total}] {url}")

        try:
            new_salary = fetch_salary(url)
        except Exception as e:
            print(f"  [Error] {e}")
            new_salary = None

        if new_salary is None:
            # Cookie expired or network error — stop to avoid wasting requests
            if not cookie_expired:
                print("\n  [STOP] Possible cookie expiry. Save progress and exit.")
                cookie_expired = True
            break

        if new_salary != old_salary and new_salary != "Thỏa thuận":
            print(f"  [UPDATE] '{old_salary}' -> '{new_salary}'")
            changes.append({"url": url, "old_salary": old_salary, "new_salary": new_salary})
            df.at[idx, "salary"] = new_salary

        done_urls.add(url)

        if count % CHECKPOINT_EVERY == 0:
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(list(done_urls), f)
            print(f"  [Checkpoint] Saved ({count}/{total})")

        time.sleep(random.uniform(3.0, 6.0))

    # Final save
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(list(done_urls), f)

    if changes:
        pd.DataFrame(changes).to_csv(log_path, index=False, encoding="utf-8-sig")

    updated = len(changes)
    print(f"\n[{person}] Done. Updated {updated}/{count} rows with specific salary.")
    if cookie_expired:
        print(f"  -> Stopped early due to cookie expiry. Run again after updating cookie.")
    print(f"  -> Changes log: {log_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("person1", "person2", "all"):
        print("Usage: python salary_recrawl.py person1 | person2 | all")
        sys.exit(1)

    target = sys.argv[1]
    if target == "all":
        recrawl_salary("person1")
        recrawl_salary("person2")
    else:
        recrawl_salary(target)
