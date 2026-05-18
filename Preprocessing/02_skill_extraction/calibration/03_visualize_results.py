import pandas as pd
import json
import re
from pathlib import Path

def clean_json_text(raw_text):
    # Loại bỏ markdown
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    return cleaned

def generate_html_report(csv_path, jsonl_path, output_html):
    # Đọc dataset gốc
    df = pd.read_csv(csv_path)
    
    # Đọc kết quả từ LLM
    results = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            row_id = data.get("row_id")
            
            if data.get("error") is None:
                try:
                    cleaned = clean_json_text(data["raw_text"])
                    parsed = json.loads(cleaned)
                    results[row_id] = parsed.get("skills", [])
                except Exception as e:
                    results[row_id] = f"Error parsing JSON: {e}"
            else:
                results[row_id] = f"LLM Error: {data['error']}"

    # Tạo HTML
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Skill Extraction Results</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }",
        ".container { max-width: 1200px; margin: 0 auto; }",
        ".card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".req-box { background: #e3f2fd; padding: 15px; border-left: 4px solid #1976d2; margin-bottom: 15px; font-size: 15px; line-height: 1.5; }",
        "table { width: 100%; border-collapse: collapse; margin-top: 10px; }",
        "th, td { border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 14px; }",
        "th { background-color: #f5f5f5; font-weight: bold; }",
        ".required { color: #d32f2f; font-weight: bold; }",
        ".preferred { color: #388e3c; font-weight: bold; }",
        ".error-box { background: #ffebee; color: #c62828; padding: 15px; border-radius: 4px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
        "<h1>Kết quả trích xuất kỹ năng (Skill Extraction)</h1>"
    ]
    
    success_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        html_content.append("<div class='card'>")
        html_content.append(f"<h3>Row ID: {idx}</h3>")
        html_content.append(f"<div class='req-box'><strong>Requirement:</strong><br>{row.get('requirement', '')}</div>")
        
        extracted = results.get(idx)
        
        if extracted is None:
            html_content.append("<div class='error-box'>Chưa được xử lý (Not found in JSONL)</div>")
        elif isinstance(extracted, str):
            error_count += 1
            html_content.append(f"<div class='error-box'><strong>Lỗi:</strong> {extracted}</div>")
        else:
            success_count += 1
            if not extracted:
                html_content.append("<p>Không tìm thấy skill nào.</p>")
            else:
                html_content.append("<table>")
                html_content.append("<tr><th>Skill Name</th><th>Label</th><th>Category</th><th>Min Years</th><th>Level</th><th>Language Level (Raw/Mapped)</th><th>Source Text</th></tr>")
                
                for skill in extracted:
                    label_class = "required" if skill.get("label") == "required_skill" else "preferred"
                    label_text = "Required" if skill.get("label") == "required_skill" else "Preferred"
                    
                    lang_raw = skill.get('language_level_raw', '') or ''
                    lang_mapped = skill.get('language_level_mapped', '') or ''
                    lang_str = f"{lang_raw} / {lang_mapped}" if lang_raw or lang_mapped else "-"
                    
                    html_content.append("<tr>")
                    html_content.append(f"<td><strong>{skill.get('skill_name', '')}</strong></td>")
                    html_content.append(f"<td class='{label_class}'>{label_text}</td>")
                    html_content.append(f"<td>{skill.get('category', '')}</td>")
                    html_content.append(f"<td>{skill.get('min_years') or '-'}</td>")
                    html_content.append(f"<td>{skill.get('level') or '-'}</td>")
                    html_content.append(f"<td>{lang_str}</td>")
                    html_content.append(f"<td><em>{skill.get('source_text', '')}</em></td>")
                    html_content.append("</tr>")
                
                html_content.append("</table>")
        
        html_content.append("</div>")
    
    html_content.insert(21, f"<p>Tổng số mẫu: {len(df)} | Thành công: {success_count} | Lỗi: {error_count}</p>")
    
    html_content.extend([
        "</div>",
        "</body>",
        "</html>"
    ])
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))
        
    print(f"Đã tạo file HTML tại: {output_html}")
    print(f"Thành công: {success_count}, Lỗi: {error_count}")

if __name__ == "__main__":
    csv_path = "calibration_dataset.csv"
    jsonl_path = "output/few_shot_raw.jsonl"
    output_html = "output/visualization.html"
    
    Path("output").mkdir(exist_ok=True)
    generate_html_report(csv_path, jsonl_path, output_html)
