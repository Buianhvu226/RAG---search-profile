import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import google.generativeai as genai

# Thiết lập API key cho Gemini
GEMINI_API_KEY = "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"  # Thay bằng API key của bạn
genai.configure(api_key=GEMINI_API_KEY)

# Tải PhoBERT để mã hóa
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Hàm lấy embedding bằng PhoBERT
def get_embedding(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Tải dữ liệu gốc và embedding
df = pd.read_csv("processed_missing_persons.csv")
embedding_df = pd.read_pickle("embeddings.pkl")  # Tải embedding PhoBERT đã tạo trước

# Hàm dùng Gemini để tách truy vấn
def parse_query_with_gemini(query):
    try:
        # Gọi Gemini để phân tích truy vấn
        response = genai.generate_content(
            model="gemini-pro",  # Tên mô hình Gemini, có thể thay đổi theo tài liệu
            prompt=f"""
            Bạn là một trợ lý phân tích truy vấn tìm người mất tích. Tách truy vấn sau thành các phần:
            - Tên người thất lạc
            - Quê quán
            - Đặc điểm nhận dạng
            - Ký ức
            - Bối cảnh thất lạc
            Truy vấn: "{query}"
            Trả về kết quả dưới dạng:
            Tên người thất lạc: <tên>
            Quê quán: <địa điểm>
            Đặc điểm nhận dạng: <mô tả>
            Ký ức: <mô tả>
            Bối cảnh thất lạc: <mô tả>
            """
        )
        result = response.text  # Giả định Gemini trả về văn bản
        # Phân tích kết quả từ Gemini
        query_parts = {
            "Tên người thất lạc": "",
            "Quê quán": [],
            "Đặc điểm nhận dạng": "",
            "Ký ức": "",
            "Bối cảnh thất lạc": ""
        }
        for line in result.split("\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                if key == "Tên người thất lạc":
                    query_parts[key] = value.strip()
                elif key == "Quê quán":
                    query_parts[key] = [value.strip()] if value.strip() else []
                elif key in query_parts:
                    query_parts[key] = value.strip()
        return query_parts
    except Exception as e:
        print(f"Lỗi khi gọi Gemini: {e}")
        return {
            "Tên người thất lạc": "",
            "Quê quán": [],
            "Đặc điểm nhận dạng": "",
            "Ký ức": "",
            "Bối cảnh thất lạc": ""
        }

# Hàm tính điểm tương đồng tổng hợp
def compute_combined_similarity(query_parts, embedding_df):
    similarities = {}
    for key, value in query_parts.items():
        if isinstance(value, list):
            value = " ".join(value)
        if value.strip():
            query_embedding = get_embedding(value)  # Dùng PhoBERT để mã hóa
            data_embeddings = np.stack(embedding_df[key].values)
            similarities[key] = cosine_similarity([query_embedding], data_embeddings)[0]
    
    combined_similarity = np.zeros(len(df))
    weights = {
        "Tên người thất lạc": 0.4,
        "Quê quán": 0.2,
        "Đặc điểm nhận dạng": 0.15,
        "Ký ức": 0.15,
        "Bối cảnh thất lạc": 0.1
    }
    
    for key in similarities:
        combined_similarity += similarities[key] * weights.get(key, 0.1)
    
    return combined_similarity

# Ví dụ sử dụng
query = "Tôi tên Khánh, năm 75 tôi có vượt biên sang Úc để đi theo diện vượt biên. Mục đích là tôi qua trước để làm thủ tục nhận gia đình qua sau. Ba tôi tên Linh, mẹ tên Kính và 5 đứa em nhỏ. Tới năm 96 thì tôi mất liên lạc với gia đình."
query_parts = parse_query_with_gemini(query)
combined_similarity = compute_combined_similarity(query_parts, embedding_df)

# Lấy top 5 kết quả
top_indices = combined_similarity.argsort()[-5:][::-1]
top_matches = df.iloc[top_indices]
print(top_matches[["Chi tiết", "Tên người thất lạc", "Quê quán", "Đặc điểm nhận dạng", "Ký ức", "Bối cảnh thất lạc"]])