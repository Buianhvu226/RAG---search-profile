import pandas as pd
import re
from underthesea import word_tokenize, ner

# Đọc file CSV
file_path = "profiles_detailed_data_cleaned.csv"  # Thay bằng đường dẫn file CSV của bạn
df = pd.read_csv(file_path)

# Hàm để tách cột "Chi tiết" thành các phần nhỏ
def extract_details(text):
    # Tách từ và nhận dạng thực thể
    tokens = word_tokenize(text)
    entities = ner(text)
    
    # Khởi tạo các phần nhỏ
    details = {
        "Tên người thất lạc": "",
        "Tên người thân": [],
        "Địa danh": [],
        "Đặc điểm nhận dạng": "",
        "Ký ức": "",
        "Bối cảnh thất lạc": ""
    }
    
    # 1. Tên người thất lạc
    if "Họ và tên" in df.columns:
        details["Tên người thất lạc"] = df.loc[df["Chi tiết"] == text, "Họ và tên"].iloc[0]
    else:
        for entity in entities:
            if entity[2] == "B-PER" and not details["Tên người thất lạc"]:
                details["Tên người thất lạc"] = entity[0]
                break
    
    # 2. Tên người thân
    if "Tên cha" in df.columns and pd.notna(df.loc[df["Chi tiết"] == text, "Tên cha"]).iloc[0]:
        details["Tên người thân"].append(df.loc[df["Chi tiết"] == text, "Tên cha"].iloc[0])
    if "Tên mẹ" in df.columns and pd.notna(df.loc[df["Chi tiết"] == text, "Tên mẹ"]).iloc[0]:
        details["Tên người thân"].append(df.loc[df["Chi tiết"] == text, "Tên mẹ"].iloc[0])
    if "Tên anh-chị-em" in df.columns and pd.notna(df.loc[df["Chi tiết"] == text, "Tên anh-chị-em"]).iloc[0]:
        details["Tên người thân"].extend(df.loc[df["Chi tiết"] == text, "Tên anh-chị-em"].iloc[0].split(", "))
    for entity in entities:
        if entity[2] == "B-PER" and entity[0] not in details["Tên người thất lạc"] and entity[0] not in details["Tên người thân"]:
            details["Tên người thân"].append(entity[0])
    
    # 3. Địa danh
    for entity in entities:
        if entity[2] == "B-LOC":
            details["Địa danh"].append(entity[0])
    locations = re.findall(r"(Hà Nội|Tp\.HCM|Sài Gòn|Đà Nẵng|Thừa Thiên Huế|Thái Bình|Quảng Nam)", text)
    details["Quê quán"].extend(locations)
    
    # 4. Đặc điểm nhận dạng
    physical_keywords = ["cao", "thấp", "da", "vết sẹo", "nốt ruồi", "tóc", "mắt"]
    for sentence in text.split("."):
        if any(keyword in sentence.lower() for keyword in physical_keywords):
            details["Đặc điểm nhận dạng"] += sentence.strip() + " "
    
    # 5. Ký ức
    memory_keywords = ["làm việc", "bán", "ghé thăm", "nhớ", "ở", "chơi"]
    for sentence in text.split("."):
        if any(keyword in sentence.lower() for keyword in memory_keywords) and sentence not in details["Đặc điểm nhận dạng"]:
            details["Ký ức"] += sentence.strip() + " "
    
    # 6. Bối cảnh thất lạc
    context_keywords = ["mất liên lạc", "thất lạc", "bỏ đi", "đi đâu", "từ đó"]
    for sentence in text.split("."):
        if (re.search(r"\d{4}", sentence) or any(keyword in sentence.lower() for keyword in context_keywords)) and sentence not in details["Ký ức"]:
            details["Bối cảnh thất lạc"] += sentence.strip() + " "
    
    # 7. Người thất lạc đi tìm gia đình
    if "tìm gia đình" in text.lower() or "tìm mẹ" in text.lower() or "tìm cha" in text.lower():
        details["Người thất lạc đi tìm gia đình"] = "Người thất lạc tìm gia đình"
    else:
        details["Người thất lạc đi tìm gia đình"] = "Gia đình tìm người thất lạc"
    
    # Loại bỏ khoảng trắng thừa
    for key in details:
        if isinstance(details[key], str):
            details[key] = details[key].strip()
        elif isinstance(details[key], list):
            details[key] = [item.strip() for item in details[key] if item.strip()]
    
    return details

# Áp dụng hàm tách chi tiết
extracted_data = df["Chi tiết"].apply(extract_details)
new_columns = pd.DataFrame(extracted_data.tolist())
df = pd.concat([df, new_columns], axis=1)

# Lưu file CSV mới
output_file = "processed_missing_persons.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Đã xử lý và lưu file vào: {output_file}")

# Hiển thị một số dòng đầu để kiểm tra
print(df[["Chi tiết", "Tên người thất lạc", "Tên người thân", "Quê quán", "Đặc điểm nhận dạng", "Ký ức"]].head())