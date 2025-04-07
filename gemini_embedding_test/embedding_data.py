import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Đọc file CSV đã xử lý
file_path = "processed_missing_persons.csv"
df = pd.read_csv(file_path)

# Tải PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Hàm để tạo embedding cho một chuỗi văn bản
def get_embedding(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return np.zeros(768)  # Trả về vector rỗng nếu text không hợp lệ (PhoBERT output 768 chiều)
    
    # Tokenize và chuyển thành tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    # Lấy embedding từ lớp cuối cùng (mean pooling)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Các cột cần embedding
columns_to_embed = [
    "Người thất lạc đi tìm gia đình",
    "Tên người thất lạc",
    "Tên người thân",
    "Quê quán",
    "Đặc điểm nhận dạng",
    "Ký ức",
    "Bối cảnh thất lạc"
]

# Tạo embedding cho từng cột
embeddings = {}
for column in columns_to_embed:
    print(f"Đang embedding cột: {column}")
    # Xử lý trường hợp cột chứa danh sách (như "Tên người thân", "Quê quán")
    if df[column].dtype == object and df[column].apply(lambda x: isinstance(x, list)).any():
        embeddings[column] = df[column].apply(lambda x: get_embedding(" ".join(x)) if isinstance(x, list) else get_embedding(x))
    else:
        embeddings[column] = df[column].apply(get_embedding)

# Tạo DataFrame mới chứa embedding
embedding_df = pd.DataFrame(embeddings)

# Lưu embedding vào file (dùng pickle vì embedding là mảng numpy)
output_embedding_file = "embeddings.pkl"
embedding_df.to_pickle(output_embedding_file)
print(f"Đã lưu embedding vào: {output_embedding_file}")

# Lưu file CSV kết hợp (nếu muốn giữ thông tin gốc và embedding cùng nhau)
df_combined = pd.concat([df, embedding_df.add_prefix("embedding_")], axis=1)
output_combined_file = "processed_missing_persons_with_embeddings.csv"
df_combined.to_csv(output_combined_file, index=False, encoding="utf-8")
print(f"Đã lưu file kết hợp vào: {output_combined_file}")