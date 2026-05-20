import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd

# =============================
# CONFIG
# =============================
LINK_FILE = "job_links/topcv_job_links_person1.csv"
OUTPUT_FILE = "test.csv"
TEST_LIMIT = 10

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.topcv.vn/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

session = requests.Session()
session.headers.update(headers)


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
                print(f"403 detected. Sleep {wait:.1f}s")
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


def parse_job_detail(job_url):
    print(f"  -> Crawling detail: {job_url}")

    # giữ nguyên logic cũ, chỉ sửa đúng biến job_url
    response = safe_get(job_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    data = {
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

    return data


# =============================
# MAIN
# =============================
links_df = pd.read_csv(LINK_FILE)

# giả sử file có cột tên là "url"
job_links = links_df["url"].dropna().tolist()

# chỉ test 10 link đầu
job_links = job_links[:TEST_LIMIT]

print(f"Testing {len(job_links)} links from {LINK_FILE}")

all_jobs = []

for i, link in enumerate(job_links, start=1):
    try:
        print(f"[{i}/{len(job_links)}] {link}")

        job_data = parse_job_detail(link)
        all_jobs.append(job_data)

        time.sleep(random.uniform(1.5, 3.0))

    except Exception as e:
        print(f"Error with {link}: {e}")

df = pd.DataFrame(all_jobs)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

print("\n===== Preview =====")
print(
    df[
        [
            "job_title",
            "company",
            "location",
            "salary",
            "experience",
            "education",
            "level",
            "employment_type",
            "industry",
            "required_skills",
            "preferred_skills",
            "specialization"
        ]
    ]
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved to {OUTPUT_FILE}")