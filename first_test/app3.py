import pandas as pd
from sentence_transformers import SentenceTransformer, util

def load_data(csv_file):
    """Đọc dữ liệu từ file CSV và trả về DataFrame"""
    df = pd.read_csv(csv_file)
    return df

def search_profiles(df, query, top_n=3):
    """Tìm kiếm các hồ sơ phù hợp nhất với mô tả đầu vào bằng Sentence Transformers"""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Kết hợp các thông tin quan trọng để tạo văn bản tìm kiếm
    df["combined_text"] = df[["Họ và tên", "Tên cha", "Tên mẹ", "Chi tiết"]].astype(str).agg(" ".join, axis=1)
    
    # Mã hóa mô tả đầu vào và hồ sơ trong dataset
    query_embedding = model.encode(query, convert_to_tensor=True)
    profile_embeddings = model.encode(df["combined_text"].tolist(), convert_to_tensor=True)
    
    # Tính toán độ tương đồng
    similarities = util.pytorch_cos_sim(query_embedding, profile_embeddings)[0]
    
    # Sắp xếp kết quả theo mức độ tương đồng cao nhất
    top_results = similarities.argsort(descending=True)[:top_n]
    
    return df.iloc[top_results.cpu().numpy()][["Họ và tên", "Năm sinh", "Năm thất lạc", "Tên cha", "Tên mẹ", "Chi tiết"]]

# Ví dụ sử dụng
if __name__ == "__main__":
    csv_file = "profiles_detailed_data_cleaned.csv"
    df = load_data(csv_file)
    query = "Tôi tên là Kim Hoàng Vũ (tên thật là Dũng), được nhận nuôi năm 81 bởi mẹ nuôi tên Nguyễn Thị Ba. Tôi được kể là khi còn nhỏ được mẹ nhận nuôi bởi lí do là ba mẹ tôi li dị, mỗi người mỗi hướng cộng thêm nhà quá khó khăn. Tôi được biết thêm bởi mẹ tôi (bà Ba) là nhà tôi còn 1 vài anh em nữa, và ba tôi tên Sên."
    
    results = search_profiles(df, query)
    print(results)
