# Hệ thống hỗ trợ tìm kiếm người thân thất lạc bằng Vector Search

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import faiss

# 1. Đọc dữ liệu CSV
df = pd.read_csv("/mnt/data/processed_missing_persons.csv")

# 2. Chọn các cột cần thiết
columns = [
    "Chi tiết", "Năm thất lạc", "Người thất lạc đi tìm gia đình",
    "Tên người thất lạc", "Tên người thân", "Quê quán",
    "Đặc điểm nhận dạng", "Ký ức", "Bối cảnh thất lạc"
]

df = df[columns].fillna("")

# 3. Gộp nội dung các trường lại thành một văn bản thống nhất để mã hóa

def combine_text(row):
    return (
        f"Chi tiết: {row['Chi tiết']}. "
        f"Năm thất lạc: {row['Năm thất lạc']}. "
        f"Người tìm: {'Người bị thất lạc' if row['Người thất lạc đi tìm gia đình'] == 1 else 'Gia đình'}. "
        f"Tên người thất lạc: {row['Tên người thất lạc']}. "
        f"Người thân: {row['Tên người thân']}. "
        f"Quê quán: {row['Quê quán']}. "
        f"Đặc điểm: {row['Đặc điểm nhận dạng']}. "
        f"Ký ức: {row['Ký ức']}. "
        f"Bối cảnh: {row['Bối cảnh thất lạc']}"
    )

texts = df.apply(combine_text, axis=1).tolist()

# 4. Mã hóa văn bản bằng SentenceTransformer với mô hình tiếng Việt
model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

print("Đang mã hóa vector hồ sơ...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

# 5. Tạo FAISS index để tìm kiếm nhanh
index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings.cpu().numpy())
index.add(embeddings.cpu().numpy())

# 6. Hàm tìm kiếm hồ sơ gần nhất theo mô tả

def search_missing_profiles(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    faiss.normalize_L2(query_embedding.cpu().numpy())
    D, I = index.search(query_embedding.cpu().numpy().reshape(1, -1), top_k)
    results = df.iloc[I[0]]
    return results, D[0]

# 7. Ví dụ truy vấn
if __name__ == "__main__":
    print("\nVí dụ truy vấn:")
    query_text = "Tôi tên là Nam, thất lạc em gái khoảng năm 1980 ở Quảng Trị, chỉ nhớ em tên có chữ Tuyết, có một vết bớt bên má phải."
    results, scores = search_missing_profiles(query_text)

    for i, (index, row) in enumerate(results.iterrows()):
        print(f"\nKết quả {i+1} (Độ tương đồng: {scores[i]:.2f}):")
        print(row['Chi tiết'])
