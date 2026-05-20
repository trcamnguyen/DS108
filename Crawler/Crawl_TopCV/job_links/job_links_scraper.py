import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer": "https://www.topcv.vn/"
}

session = requests.Session()
session.headers.update(headers)


def safe_get(url, retries=5):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=20)

            if response.status_code == 403:
                wait = random.uniform(10, 20)
                print(f"403 at {url} -> sleep {wait:.1f}s")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed: {url}")
                raise e

            wait = random.uniform(3, 8)
            print(f"Retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(wait)

    return None


def crawl_job_links(base_url, start_page=1, end_page=89):
    all_links = []

    for page in range(start_page, end_page + 1):
        page_url = f"{base_url}?page={page}"
        print(f"[Page {page}] {page_url}")

        try:
            response = safe_get(page_url)
            soup = BeautifulSoup(response.text, "html.parser")

            page_links = []

            for a in soup.select("h3.title a[href]"):
                link = a["href"].split("?")[0]

                if not link.startswith("http"):
                    link = "https://www.topcv.vn" + link

                page_links.append(link)

            # bỏ trùng trong riêng trang đó
            page_links = list(dict.fromkeys(page_links))

            print(f"  Found {len(page_links)} jobs")

            all_links.extend(page_links)

            # lưu tạm sau mỗi trang để tránh mất dữ liệu
            tmp_df = pd.DataFrame({
                "url": list(dict.fromkeys(all_links))
            })

            tmp_df.to_csv(
                "topcv_job_links_partial.csv",
                index=False,
                encoding="utf-8-sig"
            )

            time.sleep(random.uniform(2, 5))

        except Exception as e:
            print(f"Error at page {page}: {e}")

    # bỏ trùng toàn bộ
    all_links = list(dict.fromkeys(all_links))

    return all_links


if __name__ == "__main__":
    job_links = crawl_job_links(
        BASE_URL,
        start_page=1,
        end_page=89
    )

    print(f"\nTotal unique job links: {len(job_links)}")

    df = pd.DataFrame({
        "url": job_links
    })

    df.to_csv(
        "topcv_job_links.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # file txt để dễ chia cho 2 người
    with open("topcv_job_links.txt", "w", encoding="utf-8") as f:
        for link in job_links:
            f.write(link + "\n")

    print("Saved:")
    print(" - topcv_job_links.csv")
    print(" - topcv_job_links.txt")

    # chia đôi cho 2 người
    mid = len(job_links) // 2

    person1 = job_links[:mid]
    person2 = job_links[mid:]

    pd.DataFrame({"url": person1}).to_csv(
        "topcv_job_links_person1.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"url": person2}).to_csv(
        "topcv_job_links_person2.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Person 1: {len(person1)} links")
    print(f"Person 2: {len(person2)} links")