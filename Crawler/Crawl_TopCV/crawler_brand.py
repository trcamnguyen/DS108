import os
import json
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup


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


def first_text(soup, selectors):
    """Try a list of CSS selectors, return text of the first one that hits."""
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            text = clean_text(node)
            if text:
                return text
    return None


# =============================
# EXTRACTORS (BRAND PAGE)
# =============================
# Cấu trúc đã confirm từ DevTools:
#
# <div class="box-job-info">
#   <div class="box-info">
#     <h2 class="title">Thông tin</h2>
#     <div class="box-main">
#       <div class="box-item">
#         <div class="box-img">...</div>
#         <div>
#           <strong>Mức lương</strong><br>
#           <span>...</span>
#         </div>
#       </div>
#       <div class="box-item">...</div>  <!-- Kinh nghiệm -->
#       <div class="box-item">...</div>  <!-- Cấp bậc, Học vấn, ... -->
#     </div>
#   </div>
# </div>
#
# <div class="box-info">                             <- job description
# <div class="box-info job-detail-section requirement">  <- requirement
# <div class="box-address">
#   <div style="margin-bottom: 10px">DIA CHI</div>   <- location
#   ...
# </div>

def extract_overview_by_label(soup, target_labels):
    """
    Quét các .box-item trong .box-info đầu tiên của .box-job-info, match theo
    <strong> label rồi lấy <span> NẰM CÙNG CHA với <strong> (tránh span trong
    .box-img cùng .box-item).
    """
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    target_labels = [t.strip().lower() for t in target_labels]

    # chỉ lấy .box-info đầu tiên trong .box-job-info (block overview)
    overview_box = soup.select_one(".box-job-info .box-info")
    if not overview_box:
        return None

    items = overview_box.select(".box-item")
    if not items:
        items = overview_box.select(".box-main .box-item")

    for item in items:
        label_node = item.select_one("strong")
        if not label_node:
            continue

        label_text = clean_text(label_node)
        if not label_text:
            continue

        if not any(t in label_text.lower() for t in target_labels):
            continue

        # value <span> nằm CÙNG CHA với <strong>, không phải bất kỳ span nào
        # trong .box-item (vì .box-img có thể chứa span icon).
        parent = label_node.parent
        if parent is None:
            continue

        value_node = parent.find("span", recursive=False)
        if value_node:
            val = clean_text(value_node)
            if val:
                return val

        # fallback: lấy text của parent, bỏ phần label
        full = clean_text(parent)
        if full and label_text in full:
            rest = full.replace(label_text, "", 1).strip(" :-")
            if rest:
                return rest

    return None


def extract_job_description(soup):
    """
    Job description nằm trong <div class="box-info"> (KHÔNG có class
    'requirement' và 'job-detail-section', và KHÔNG phải overview box
    chứa h2.title='Thông tin' + .box-main).
    """
    for box in soup.select("div.box-info"):
        classes = box.get("class", [])

        # bỏ box requirement
        if "requirement" in classes or "job-detail-section" in classes:
            continue

        # bỏ overview box ("Thông tin" + .box-main)
        if box.select_one(".box-main"):
            continue

        text = clean_text(box)
        if not text:
            continue

        # bỏ heading nếu có (vd 'Mô tả công việc')
        heading = box.select_one("h2, h3, .title")
        if heading:
            head_text = clean_text(heading)
            if head_text and text.startswith(head_text):
                text = text[len(head_text):].strip(" :-")

        return text

    return None


def extract_requirement(soup):
    """Requirement: <div class='box-info job-detail-section requirement'>."""
    box = soup.select_one("div.box-info.job-detail-section.requirement")
    if not box:
        box = soup.select_one("div.box-info.requirement")
    if not box:
        return None

    text = clean_text(box)
    if not text:
        return None

    heading = box.select_one("h2, h3, .title")
    if heading:
        head_text = clean_text(heading)
        if head_text and text.startswith(head_text):
            text = text[len(head_text):].strip(" :-")

    return text


def extract_location(soup):
    """
    Location: trong .box-address, lấy phần tử có inline style
    'margin-bottom: 10px'.
    """
    container = soup.select_one(".box-address")
    if not container:
        return None

    # selector attribute substring để bắt cả 2 dạng spacing
    node = (
        container.select_one('[style*="margin-bottom: 10px"]')
        or container.select_one('[style*="margin-bottom:10px"]')
    )

    if node:
        return clean_text(node)
    return None


def collect_tags(soup, selectors):
    tags = []
    for sel in selectors:
        for node in soup.select(sel):
            value = clean_text(node)
            if value and value not in tags:
                tags.append(value)
    return tags


# =============================
# DETAIL PARSER (BRAND)
# =============================
def parse_brand_job_detail(job_url):
    print(f"  -> Crawling brand: {job_url}")

    response = safe_get(job_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # ----- title -----
    title_node = (
        soup.select_one(".block-left .box-header h2.title")
        or soup.select_one(".box-header h2.title")
    )

    if title_node:
        # remove span icon (icon-verified-employer level-five, etc.)
        for icon in title_node.select("span[class*='icon']"):
            icon.decompose()
        job_title = clean_text(title_node)
    else:
        # fallback cho các template brand khác (nếu có)
        job_title = first_text(soup, [
            "h1.job-desc__title",
            ".job-desc-info h1",
            ".job-title h1",
        ])
    
    # # DEBUG
    # title_check = soup.select(".block-left .box-header h2.title")
    # print(f"DEBUG: found {len(title_check)} h2.title in raw HTML")
    # h2_all = soup.select("h2.title")
    # print(f"DEBUG: found {len(h2_all)} h2.title total")
    # if h2_all:
    #     print(f"DEBUG: first h2.title text = {clean_text(h2_all[0])[:100]}")

    # ----- company -----
    company = first_text(soup, [
        ".company-name a",
        ".company-name",
        ".job-desc__company a",
        ".box-company-info .name",
    ])

    # ----- location -----
    location = extract_location(soup)

    # ----- overview fields (từ .box-job-info .box-item) -----
    salary = extract_overview_by_label(soup, ["mức lương", "thu nhập", "lương"])
    experience = extract_overview_by_label(soup, ["kinh nghiệm"])
    education = extract_overview_by_label(soup, ["học vấn", "trình độ"])
    level = extract_overview_by_label(soup, ["cấp bậc"])
    employment_type = extract_overview_by_label(
        soup, ["hình thức làm việc", "hình thức"]
    )
    industry = extract_overview_by_label(soup, ["lĩnh vực", "ngành nghề"])

    # ----- description / requirement -----
    job_description = extract_job_description(soup)
    requirement = extract_requirement(soup)

    # ----- specialization (best-effort) -----
    specialization = collect_tags(soup, [
        ".job-tags__group-list-tag-scroll a.item",
        ".tags-job a",
        ".job-tag a",
    ])

    return {
        "url": job_url,
        "job_title": job_title,
        "company": company,
        "location": location,
        "salary": salary,
        "experience": experience,
        "education": education,
        "level": level,
        "employment_type": employment_type,
        "industry": industry,
        "required_skills": [],
        "preferred_skills": [],
        "specialization": specialization,
        "job_description": job_description,
        "requirement": requirement,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Crawl lại toàn bộ brand URL, bỏ qua progress cũ.",
    )
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")

    CSV_FILE      = os.path.join(PROJECT_ROOT, "data", "raw", "00-topcv_raw.csv")
    PROGRESS_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "00-topcv_brand_progress.json")
    FAILED_FILE   = os.path.join(PROJECT_ROOT, "data", "raw", "00-topcv_brand_failed.json")

    # =============================
    # LOAD DATASET
    # =============================
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    if "brand_recrawled" not in df.columns:
        df["brand_recrawled"] = False
    print(f"Loaded dataset           : {len(df)} rows, {len(df.columns)} cols")

    # =============================
    # LOAD PROGRESS
    # =============================
    if args.force:
        crawled_urls  = set()
        failed_entries = []
        print("--force: bỏ qua progress cũ, crawl lại toàn bộ.")
    else:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                try:
                    crawled_urls = set(json.load(f))
                except json.JSONDecodeError:
                    crawled_urls = set()
        else:
            crawled_urls = set()

        if os.path.exists(FAILED_FILE):
            with open(FAILED_FILE, "r", encoding="utf-8") as f:
                try:
                    failed_entries = json.load(f)
                except json.JSONDecodeError:
                    failed_entries = []
        else:
            failed_entries = []

    # index để xóa nhanh entry đã resolve
    failed_by_url = {e["url"]: i for i, e in enumerate(failed_entries)}

    # =============================
    # TÌM BRAND ROWS CẦN CRAWL
    # =============================
    def needs_recrawl(row):
        key_fields = ["job_title", "job_description", "requirement", "salary"]
        empty = 0
        for k in key_fields:
            v = row.get(k)
            if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "[]"):
                empty += 1
        return empty >= 2

    brand_mask = df["url"].str.contains("/brand/", na=False)

    if args.force:
        # crawl lại tất cả brand rows, không lọc needs_recrawl
        to_crawl = [(i, df.at[i, "url"]) for i in df[brand_mask].index]
    else:
        to_crawl = [
            (i, df.at[i, "url"])
            for i in df[brand_mask].index
            if df.at[i, "url"] not in crawled_urls and needs_recrawl(df.loc[i].to_dict())
        ]

    print(f"Brand rows in dataset    : {brand_mask.sum()}")
    print(f"Already crawled          : {len(crawled_urls)}")
    print(f"To crawl                 : {len(to_crawl)}")

    # =============================
    # SAVE HELPERS
    # =============================
    def save_checkpoint():
        df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(crawled_urls), f, ensure_ascii=False, indent=2)
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            json.dump(failed_entries, f, ensure_ascii=False, indent=2)

    # =============================
    # MAIN LOOP
    # =============================
    UPDATABLE_COLS = [c for c in df.columns if c != "url"]

    for n, (row_idx, url) in enumerate(to_crawl, start=1):
        try:
            print(f"[{n}/{len(to_crawl)}] row {row_idx} | {url}")

            job_data = parse_brand_job_detail(url)

            # cập nhật đúng ô, không đụng các dòng khác
            for col in UPDATABLE_COLS:
                if col in job_data:
                    val = job_data[col]
                    # list phải convert sang str để df.at không bị lỗi broadcast
                    if isinstance(val, list):
                        val = str(val)
                    df.at[row_idx, col] = val

            df.at[row_idx, "brand_recrawled"] = True
            crawled_urls.add(url)
            # xóa khỏi failed list nếu trước đó đã fail
            if url in failed_by_url:
                del failed_entries[failed_by_url.pop(url)]
                failed_by_url = {e["url"]: i for i, e in enumerate(failed_entries)}
            print(f"    -> updated row {row_idx}")

            if n % 10 == 0:
                save_checkpoint()
                print(f"    Checkpoint saved ({len(crawled_urls)} crawled total)")

            time.sleep(random.uniform(1.5, 3.0))

        except Exception as e:
            print(f"    ERROR: {e}")
            failed_entries.append({"url": url, "row": row_idx, "error": str(e)})
            with open(FAILED_FILE, "w", encoding="utf-8") as f:
                json.dump(failed_entries, f, ensure_ascii=False, indent=2)

    # =============================
    # FINAL SAVE
    # =============================
    save_checkpoint()
    print(f"\nDone. {len(crawled_urls)} crawled, {len(failed_entries)} failed.")
    print(f"Saved : {CSV_FILE}")