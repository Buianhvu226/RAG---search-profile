import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time
from tqdm import tqdm
import google.generativeai as genai

# --- Cấu hình API key cho Google AI ---
# Thay thế bằng API key của bạn - hoặc lấy từ biến môi trường
GOOGLE_API_KEY = "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"  # API key trong app2.py
genai.configure(api_key=GOOGLE_API_KEY)

# Hàm lấy embedding từ Google API
def get_embedding(text, retry_count=3):
    """Lấy embedding từ Google API với cơ chế thử lại"""
    if not text or pd.isna(text) or not isinstance(text, str):
        # Trả về vector rỗng nếu text rỗng
        return np.zeros(768)  # Kích thước mặc định cho embedding
    
    # Giới hạn độ dài văn bản để tránh lỗi
    text = text[:8000]  # Gemini Embedding có giới hạn độ dài
    
    for attempt in range(retry_count):
        try:
            result = genai.embed_content(
                model="models/embedding-001",  # Model embedding của Google
                content=text,
                task_type="RETRIEVAL_QUERY"  # Loại nhiệm vụ: RETRIEVAL_DOCUMENT hoặc RETRIEVAL_QUERY
            )
            return result["embedding"]
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"Lỗi khi tạo embedding (lần {attempt+1}/{retry_count}): {e}")
                time.sleep(2)  # Đợi 2 giây trước khi thử lại
            else:
                print(f"Không thể tạo embedding sau {retry_count} lần: {e}")
                # Trả về vector zero nếu lỗi
                return np.zeros(768)  # Điều chỉnh kích thước phù hợp

# Hàm trích xuất thông tin chi tiết từ các cột
def combine_text(row, columns):
    """Kết hợp các cột thành một đoạn văn bản có cấu trúc"""
    parts = []
    
    # Thêm tiêu đề nếu có
    if 'Tiêu đề' in row and pd.notna(row['Tiêu đề']):
        parts.append(f"Tiêu đề: {row['Tiêu đề']}")
    
    # Thêm họ và tên nếu có
    if 'Họ và tên' in row and pd.notna(row['Họ và tên']):
        parts.append(f"Họ và tên: {row['Họ và tên']}")
    
    # Thêm các cột khác có cấu trúc
    for col in columns:
        if col in row and pd.notna(row[col]):
            # Xử lý đặc biệt cho các cột danh sách
            if isinstance(row[col], list):
                value = ", ".join([str(item) for item in row[col] if pd.notna(item)])
            else:
                value = str(row[col])
            
            if value.strip():  # Chỉ thêm nếu không rỗng
                parts.append(f"{col}: {value}")
    
    # Kết hợp thành một đoạn văn
    return " ".join(parts)

def main():
    print("Bắt đầu tạo embeddings với Google Gemini API...")
    
    # 1. Đọc dữ liệu
    try:
        # Thử đọc file đã xử lý, nếu không tìm thấy thì đọc file gốc
        try:
            df = pd.read_csv('processed_missing_persons.csv')
            print(f"Đã đọc dữ liệu từ processed_missing_persons.csv ({len(df)} hồ sơ)")
        except FileNotFoundError:
            df = pd.read_csv('profiles_detailed_data_cleaned.csv')
            print(f"Đã đọc dữ liệu từ profiles_detailed_data_cleaned.csv ({len(df)} hồ sơ)")
    except Exception as e:
        print(f"Lỗi khi đọc file dữ liệu: {e}")
        return
    
    # 2. Lựa chọn các cột để embedding
    columns_to_embed = [
        'Chi tiết', 'Năm thất lạc', 'Người thất lạc đi tìm gia đình',
        'Tên người thất lạc', 'Tên người thân', 'Quê quán',
        'Đặc điểm nhận dạng', 'Ký ức', 'Bối cảnh thất lạc'
    ]
    
    # Kiểm tra các cột có sẵn
    available_columns = [col for col in columns_to_embed if col in df.columns]
    print(f"Sử dụng các cột: {', '.join(available_columns)}")
    
    # 3. Tạo cột văn bản tổng hợp
    print("Tạo văn bản tổng hợp cho mỗi hồ sơ...")
    df['combined_text'] = df.apply(lambda row: combine_text(row, available_columns), axis=1)
    
    # 4. Kiểm tra xem file embedding đã tồn tại chưa
    embedding_file = 'gemini_embeddings.npy'
    if os.path.exists(embedding_file):
        print(f"Tìm thấy file embedding đã lưu ({embedding_file}), đang tải...")
        embeddings = np.load(embedding_file, allow_pickle=True)
        if len(embeddings) == len(df):
            df['embedding'] = list(embeddings)
            print("Đã tải embeddings từ file.")
        else:
            print(f"Kích thước embedding ({len(embeddings)}) không khớp với số lượng hồ sơ ({len(df)}). Tạo lại...")
            create_new = True
    else:
        create_new = True
    
    # 5. Tạo embeddings nếu cần
    if create_new:
        print("Bắt đầu tạo embeddings mới...")
        embeddings = []
        
        # Sử dụng tqdm để hiển thị tiến độ
        for i, text in enumerate(tqdm(df['combined_text'], desc="Đang tạo embedding")):
            try:
                emb = get_embedding(text)
                embeddings.append(emb)
                
                # Định kỳ lưu để tránh mất dữ liệu khi bị gián đoạn
                if (i+1) % 50 == 0:
                    np.save(embedding_file, np.array(embeddings + [np.zeros(768)] * (len(df) - i - 1), dtype=object))
                    print(f"\nĐã lưu tạm {i+1}/{len(df)} embeddings")
            except Exception as e:
                print(f"\nLỗi với hồ sơ {i}: {e}")
                embeddings.append(np.zeros(768))  # Thêm vector rỗng nếu có lỗi
                
            # Thêm độ trễ nhỏ để tránh đạt giới hạn API
            time.sleep(0.1)
        
        # 6. Lưu embeddings
        df['embedding'] = embeddings
        np.save(embedding_file, np.array(embeddings, dtype=object))
        print(f"Đã tạo và lưu {len(embeddings)} embeddings")
    
    # 7. Lưu DataFrame hoàn chỉnh
    df.to_pickle('profiles_with_gemini_embeddings.pkl')
    print("Đã lưu DataFrame với embeddings")
    
    # 8. Thử chức năng tìm kiếm
    test_search(df)

def test_search(df):
    print("\n--- Thử nghiệm tìm kiếm ---")
    
    # Chuyển đổi danh sách embeddings thành mảng NumPy để tính toán hiệu quả
    embeddings_list = df['embedding'].tolist()
    embedding_matrix = np.vstack(embeddings_list)
    
    # Vòng lặp để người dùng nhập nhiều truy vấn
    while True:
        user_query = input("\nNhập truy vấn tìm kiếm (hoặc 'q' để thoát): ")
        if user_query.lower() == 'q':
            break
            
        try:
            top_n = int(input("Số kết quả muốn hiển thị: ") or "5")
        except:
            top_n = 5
        
        print(f"\nTìm kiếm cho: '{user_query}'")
        try:
            query_embedding = get_embedding(user_query)
            query_embedding_np = np.array(query_embedding).reshape(1, -1)
            
            # Tính cosine similarity
            similarities = cosine_similarity(query_embedding_np, embedding_matrix)[0]
            
            # Lấy index của top_n hồ sơ tương đồng nhất
            top_n_indices = np.argsort(similarities)[-top_n:][::-1]
            
            # Hiển thị kết quả
            print("\n--- Kết quả tìm kiếm ---")
            for i, idx in enumerate(top_n_indices):
                profile = df.iloc[idx]
                print(f"{i+1}. Độ tương đồng: {similarities[idx]:.4f}")
                print(f"   Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                print(f"   Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                
                # Hiển thị chi tiết ngắn gọn
                if 'Chi tiết' in profile:
                    details = str(profile['Chi tiết'])
                    print(f"   Chi tiết: {details[:200]}..." if len(details) > 200 else f"   Chi tiết: {details}")
                
                # Hiển thị link nếu có
                if 'Link' in profile and pd.notna(profile['Link']):
                    print(f"   Link: {profile['Link']}")
                print("-" * 40)
        
        except Exception as e:
            print(f"Lỗi trong quá trình tìm kiếm: {e}")

if __name__ == "__main__":
    main()