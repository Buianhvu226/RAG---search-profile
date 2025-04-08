import pandas as pd
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

# Lưu kết quả mã hóa
import pickle
with open("embeddings.pkl", "wb") as f:
    pickle.dump(encoded_columns, f)
print("Đã lưu embeddings vào embeddings.pkl")