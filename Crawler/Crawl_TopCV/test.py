"""
Test extractors trên 1 file HTML đã lưu, không cần request.

Cách dùng:
    1. Mở 1 trang brand TopCV trên browser, Ctrl+S -> lưu HTML thành brand.html
    2. python test_brand_extractor.py brand.html
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from bs4 import BeautifulSoup

from crawler_brand import (
    extract_overview_by_label,
    extract_job_description,
    extract_requirement,
    extract_location,
    first_text,
    collect_tags,
)

html_path = sys.argv[1]

with open(html_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# DEBUG
title_check = soup.select(".block-left .box-header h2.title")
print(f"DEBUG: found {len(title_check)} h2.title in .block-left .box-header")
h2_all = soup.select("h2.title")
print(f"DEBUG: found {len(h2_all)} h2.title total")
if h2_all:
    print(f"DEBUG: first h2.title text = {h2_all[0].get_text(strip=True)[:100]}")
print("=" * 70)
print("TITLE:", first_text(soup, [
    ".block-left .box-header h2.title",
    ".box-header h2.title",
    "h2.title",
    "h1.job-desc__title", ".job-desc-info h1", ".job-title h1", "h1",
]))
print("COMPANY:", first_text(soup, [
    ".company-name a", ".company-name", ".job-desc__company a",
    ".box-company-info .name"
]))
print("LOCATION:", extract_location(soup))
print("-" * 70)
print("SALARY:", extract_overview_by_label(soup, ["mức lương", "thu nhập", "lương"]))
print("EXPERIENCE:", extract_overview_by_label(soup, ["kinh nghiệm"]))
print("EDUCATION:", extract_overview_by_label(soup, ["học vấn", "trình độ"]))
print("LEVEL:", extract_overview_by_label(soup, ["cấp bậc"]))
print("EMPLOYMENT TYPE:", extract_overview_by_label(soup, ["hình thức làm việc", "hình thức"]))
print("INDUSTRY:", extract_overview_by_label(soup, ["lĩnh vực", "ngành nghề"]))
print("-" * 70)

desc = extract_job_description(soup)
print("JOB DESCRIPTION:")
print((desc or "")[:500], "..." if desc and len(desc) > 500 else "")
print("-" * 70)

req = extract_requirement(soup)
print("REQUIREMENT:")
print((req or "")[:500], "..." if req and len(req) > 500 else "")
print("=" * 70)