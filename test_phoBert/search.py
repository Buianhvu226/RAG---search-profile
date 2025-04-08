import pandas as pd
import numpy as np
import faiss
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# Đọc dữ liệu từ file CSV
file_path = "dataset/processed_missing_persons.csv"
df = pd.read_csv(file_path)

# Tải PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Hàm mã hóa văn bản
def encode_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Các cột cần mã hóa
columns_to_encode = [
    "Chi tiết", "Năm thất lạc", "Người thất lạc đi tìm gia đình",
    "Tên người thất lạc", "Tên người thân", "Quê quán",
    "Đặc điểm nhận dạng", "Ký ức", "Bối cảnh thất lạc"
]

# Mã hóa từng cột
encoded_columns = {}
for column in columns_to_encode:
    print(f"Đang mã hóa cột: {column}")
    encoded_columns[column] = df[column].fillna("").apply(encode_text)

# Lưu embeddings vào file (nếu cần sử dụng lại)
with open("embeddings.pkl", "wb") as f:
    pickle.dump(encoded_columns, f)

# Tạo chỉ mục FAISS
dimension = len(encoded_columns["Chi tiết"][0])  # Kích thước vector
index = faiss.IndexFlatL2(dimension)

# Thêm dữ liệu vào chỉ mục
vectors = np.array(encoded_columns["Chi tiết"].tolist(), dtype="float32")
index.add(vectors)

# Hàm tìm kiếm
def search(query, top_k=5):
    query_vector = encode_text(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return distances, indices

# Ví dụ tìm kiếm
query = "Tìm người tên Sơn, quê Hà Nội, mất liên lạc năm 1995"
distances, indices = search(query)

# In kết quả tìm kiếm
print("Kết quả tìm kiếm:")
for idx in indices[0]:
    print(df.iloc[idx].to_dict())