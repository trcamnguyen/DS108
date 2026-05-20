# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import json
import os

# Cấu hình trang Streamlit
st.set_page_config(layout="wide", page_title="Skill Annotation App")

# Khởi tạo các biến trong session_state để lưu trữ trạng thái
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
    # Tự động tải dữ liệu cũ nếu file mặc định tồn tại để làm tiếp tục
    if os.path.exists("annotated_skills.json"):
        try:
            with open("annotated_skills.json", "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                for item in saved_data:
                    st.session_state.annotations[str(item["id"])] = item
                if saved_data:
                    # Cập nhật index tới job cuối cùng đã làm
                    max_id = max([int(item["id"]) for item in saved_data])
                    st.session_state.current_index = max_id
        except Exception:
            pass

def load_data():
    """Đọc dữ liệu từ file calibration_dataset.csv"""
    file_path = "calibration_dataset.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def _invalidate_editor(job_key):
    """Xóa base DataFrame và widget state của data_editor để buộc rebuild."""
    for prefix in ("_base_", "editor_"):
        k = f"{prefix}{job_key}"
        if k in st.session_state:
            del st.session_state[k]

def _invalidate_all_editors():
    """Xóa tất cả editor state (dùng khi import dữ liệu mới)."""
    for k in list(st.session_state.keys()):
        if k.startswith("_base_") or k.startswith("editor_"):
            del st.session_state[k]

def main():
    st.title("🎯 Gán nhãn kỹ năng (Skill Annotation)")
    
    df = load_data()
    if df is None:
        st.error("Không tìm thấy file `calibration_dataset.csv` trong thư mục hiện tại. Vui lòng kiểm tra lại!")
        return
        
    total_jobs = len(df)
    
    if st.session_state.current_index >= total_jobs:
        st.session_state.current_index = total_jobs - 1
        
    idx = st.session_state.current_index
    row = df.iloc[idx]
    
    job_key = str(idx)
    # Khởi tạo bản ghi rỗng cho job nếu chưa có
    if job_key not in st.session_state.annotations:
        st.session_state.annotations[job_key] = {
            "id": int(idx),
            "job_title": str(row.get('job_title', '')),
            "requirement": str(row.get('requirement', '')),
            "skills": []
        }
        
    # --- Điều hướng (Navigation) ---
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅️ Previous Job") and st.session_state.current_index > 0:
            _invalidate_editor(job_key)
            st.session_state.current_index -= 1
            st.rerun()
    with col_nav2:
        st.markdown(f"<h3 style='text-align: center;'>Job {st.session_state.current_index + 1} / {total_jobs}</h3>", unsafe_allow_html=True)
    with col_nav3:
        if st.button("Next Job ➡️") and st.session_state.current_index < total_jobs - 1:
            _invalidate_editor(job_key)
            st.session_state.current_index += 1
            st.rerun()

    st.markdown("---")
    
    # --- Thông tin Job ---
    st.header("📄 Thông tin Requirement")
    st.write(f"**Job Title:** {row.get('job_title', 'N/A')} | **Company:** {row.get('company', 'N/A')}")
    
    # Hiển thị Requirement rõ ràng
    st.info(row.get('requirement', ''))
    
    st.markdown("---")
    
    # --- Form Thêm Kỹ năng (Annotation Form) ---
    st.header("✍️ Thêm Kỹ năng (Skill)")
    
    with st.form("add_skill_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            skill_name = st.text_input("Skill Name (*)")
        with col2:
            label = st.selectbox("Label", ["required_skill", "preferred_skill"])
        with col3:
            category = st.selectbox("Category", [
                "Programming Language", "Framework / Library", "Database", 
                "Infrastructure & DevOps", "AI / ML / Data", "Data Engineering & Analytics", 
                "Testing & QA", "Engineering Concepts & Methodologies", "Tool & Platform", "Soft Skill", 
                "Domain Knowledge", "IT Support & Hardware", "Embedded & Firmware", "Other"
            ])
            
        col4, col5 = st.columns(2)
        with col4:
            min_years_input = st.text_input("Min Years (để trống nếu không có)", value="")
        with col5:
            level = st.selectbox("Level", ["None", "basic", "intermediate", "expert"])
            
        submitted = st.form_submit_button("➕ Thêm Skill")
        
        if submitted:
            if skill_name.strip() == "":
                st.warning("Vui lòng nhập Skill Name!")
            else:
                # Xử lý min_years
                min_years = None
                if min_years_input.strip() != "":
                    try:
                        # Thử ép kiểu float/int
                        min_years = float(min_years_input)
                        if min_years.is_integer():
                            min_years = int(min_years)
                    except ValueError:
                        # Nếu người dùng nhập chữ (ví dụ "3+")
                        min_years = min_years_input.strip()
                        
                # Xử lý level
                final_level = None if level == "None" else level
                
                skill_obj = {
                    "skill_name": skill_name.strip(),
                    "label": label,
                    "category": category,
                    "min_years": min_years,
                    "level": final_level
                }
                
                st.session_state.annotations[job_key]["skills"].append(skill_obj)
                _invalidate_editor(job_key)
                st.success(f"Đã thêm thành công: {skill_name}")

    # --- Hiển thị danh sách kỹ năng đã gán nhãn ---
    st.markdown("### 📋 Các kỹ năng đã thêm cho Job này (Có thể chỉnh sửa trực tiếp)")

    @st.fragment
    def render_skill_editor():
        skills = st.session_state.annotations[job_key]["skills"]

        if not skills:
            st.write("Chưa có kỹ năng nào được thêm.")
            return

        # Stable base: chỉ rebuild khi bị invalidate (add/delete/import/navigation),
        # KHÔNG rebuild khi user chỉnh sửa cell → giữ nguyên widget state & scroll.
        base_key = f"_base_{job_key}"
        if base_key not in st.session_state:
            st.session_state[base_key] = pd.DataFrame(skills)

        edited_df = st.data_editor(
            st.session_state[base_key],
            key=f"editor_{job_key}",
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "label": st.column_config.SelectboxColumn(
                    "Label",
                    options=["required_skill", "preferred_skill"],
                    required=True
                ),
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=[
                        "Programming Language", "Framework / Library", "Database",
                        "Infrastructure & DevOps", "AI / ML / Data", "Data Engineering & Analytics",
                        "Testing & QA", "Engineering Concepts & Methodologies", "Tool & Platform", "Soft Skill",
                        "Domain Knowledge", "IT Support & Hardware", "Embedded & Firmware", "Other"
                    ],
                    required=True
                ),
                "level": st.column_config.SelectboxColumn(
                    "Level",
                    options=["None", "basic", "intermediate", "expert"]
                )
            }
        )

        # Đồng bộ kết quả edit vào annotations (để Save/Export đúng),
        # nhưng KHÔNG ghi lại vào base → base ổn định, widget không bị reset.
        st.session_state.annotations[job_key]["skills"] = (
            edited_df.where(pd.notnull(edited_df), None).to_dict("records")
        )

        # Nút xóa skill cuối cùng
        if st.button("🗑️ Xóa skill vừa thêm"):
            if st.session_state.annotations[job_key]["skills"]:
                st.session_state.annotations[job_key]["skills"].pop()
                _invalidate_editor(job_key)
                st.rerun()

    render_skill_editor()

    st.markdown("---")
    
    # --- Lưu Output ---
    st.header("💾 Quản lý Dữ liệu (Export/Import JSON)")
    output_filename = st.text_input("Tên file output/input:", value="annotated_skills.json")
    
    col_file1, col_file2 = st.columns(2)
    with col_file1:
        if st.button("Lưu toàn bộ kết quả ra file JSON"):
            # Chuyển đổi dictionary thành list để ra cấu trúc JSON chuẩn
            out_list = [val for k, val in st.session_state.annotations.items()]
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(out_list, f, ensure_ascii=False, indent=2)
                st.success(f"🎉 Đã lưu kết quả thành công vào file: {output_filename}!")
            except Exception as e:
                st.error(f"Lỗi khi lưu file: {e}")
                
    with col_file2:
        if st.button("Tải dữ liệu từ file này để làm tiếp"):
            if os.path.exists(output_filename):
                try:
                    with open(output_filename, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                        st.session_state.annotations = {}
                        for item in saved_data:
                            st.session_state.annotations[str(item["id"])] = item
                    _invalidate_all_editors()
                    st.success(f"Đã tải dữ liệu thành công từ file: {output_filename}! Bạn có thể chuyển job để thấy kết quả.")
                except Exception as e:
                    st.error(f"Lỗi khi đọc file: {e}")
            else:
                st.warning("File không tồn tại!")

if __name__ == "__main__":
    main()
