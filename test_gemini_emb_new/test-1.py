import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os
import time # Để xử lý rate limit

# --- 1. Cấu hình và Chuẩn bị ---
# Thay YOUR_API_KEY bằng API Key của bạn
# os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY' # Cách khác để đặt key
genai.configure(api_key="AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64")

# Chọn model embedding
EMBEDDING_MODEL = 'text-embedding-004' # Hoặc model mới hơn nếu có

# Đường dẫn đến file CSV
# F:\missing_people(NCHCCCL)\data\profiles_detailed_data_cleaned.csv
CSV_FILE_PATH = 'F:\\missing_people(NCHCCCL)\\data\\profiles_detailed_data_cleaned.csv' # Đường dẫn đến file CSV
DETAIL_COLUMN = 'Chi tiết' # Tên cột chứa mô tả

# --- Hàm tạo Embedding ---
def get_embedding(text, model=EMBEDDING_MODEL):
    """Tạo embedding cho một đoạn văn bản sử dụng Gemini API."""
    try:
        # Xử lý trường hợp text là None hoặc không phải string
        if not isinstance(text, str):
            print(f"Warning: Input is not a string, returning None embedding. Input: {text}")
            # Trả về vector zero hoặc xử lý khác tùy yêu cầu
            # Ví dụ trả về vector zero có độ dài phù hợp (cần biết độ dài output của model)
            # Hoặc bỏ qua bản ghi này
            return None # Hoặc np.zeros(output_dimension)

        # Xử lý text rỗng
        if not text.strip():
             print("Warning: Input text is empty, returning None embedding.")
             return None # Hoặc np.zeros(output_dimension)

        result = genai.embed_content(
            model=f"models/{model}",
            content=text,
            task_type="RETRIEVAL_DOCUMENT" # Hoặc "SEMANTIC_SIMILARITY" tùy mục đích
        )
        return result['embedding']
    except Exception as e:
        print(f"Lỗi khi tạo embedding cho text: '{text[:50]}...' - Lỗi: {e}")
        # Có thể thêm logic retry hoặc chờ nếu gặp lỗi rate limit
        if "Resource has been exhausted" in str(e) or "429" in str(e):
             print("Gặp lỗi Rate Limit, chờ 60 giây...")
             time.sleep(60)
             return get_embedding(text, model) # Thử lại
        return None # Trả về None nếu có lỗi khác

# --- 2. Tạo và Lưu trữ Embeddings cho Dữ liệu CSV ---
print(f"Đang đọc dữ liệu từ {CSV_FILE_PATH}...")
try:
    # Use error_bad_lines=False (renamed to on_bad_lines='skip' in newer pandas)
    df = pd.read_csv(CSV_FILE_PATH, 
                     on_bad_lines='skip',  # Skip problematic rows
                     quoting=3,            # QUOTE_NONE - don't use quotes for fields
                     escapechar='\\',      # Use backslash as escape character
                     encoding='utf-8')     # Specify encoding explicitly
    
    # Reset index to ensure we have a simple integer index
    df = df.reset_index(drop=True)
    
    # Lấy danh sách các mô tả, xử lý giá trị NaN (nếu có) -> thay bằng chuỗi rỗng
    if DETAIL_COLUMN not in df.columns:
        print(f"Warning: Column '{DETAIL_COLUMN}' not found. Available columns: {df.columns.tolist()}")
        # Try to find a similar column name
        possible_columns = [col for col in df.columns if 'chi' in col.lower() or 'detail' in col.lower()]
        if possible_columns:
            DETAIL_COLUMN = possible_columns[0]
            print(f"Using '{DETAIL_COLUMN}' instead.")
        else:
            raise KeyError(f"Cannot find a suitable replacement for '{DETAIL_COLUMN}'")
            
    descriptions = df[DETAIL_COLUMN].fillna('').tolist()
    print(f"Đã đọc {len(descriptions)} mô tả.")
    print(f"Các cột có trong file: {df.columns.tolist()}")
    print(f"Loại index của DataFrame: {type(df.index)}")
    
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file CSV: {e}")
    print("Thử đọc file với cách khác...")
    
    # Alternative approach using a different delimiter
    try:
        df = pd.read_csv(CSV_FILE_PATH, 
                         sep='\t',         # Try tab as delimiter
                         on_bad_lines='skip',
                         encoding='utf-8')
        
        if DETAIL_COLUMN not in df.columns:
            print(f"Warning: Column '{DETAIL_COLUMN}' not found. Available columns: {df.columns.tolist()}")
            # Try to find a similar column name
            possible_columns = [col for col in df.columns if 'chi' in col.lower() or 'detail' in col.lower()]
            if possible_columns:
                DETAIL_COLUMN = possible_columns[0]
                print(f"Using '{DETAIL_COLUMN}' instead.")
            else:
                raise KeyError(f"Cannot find a suitable replacement for '{DETAIL_COLUMN}'")
                
        descriptions = df[DETAIL_COLUMN].fillna('').tolist()
        print(f"Đã đọc {len(descriptions)} mô tả với delimiter tab.")
    except Exception as e2:
        print(f"Lỗi khi đọc file CSV với cách thay thế: {e2}")
        exit()
except KeyError:
    print(f"Lỗi: Không tìm thấy cột '{DETAIL_COLUMN}' trong file CSV.")
    exit()

print("Bắt đầu tạo embeddings cho dữ liệu...")
embeddings_list = []
original_indices = [] # Lưu index gốc để map lại với DataFrame

for index, desc in enumerate(descriptions):
    # Bỏ qua các mô tả không hợp lệ (đã được xử lý trong get_embedding)
    embedding = get_embedding(desc)
    if embedding is not None:
        embeddings_list.append(embedding)
        original_indices.append(index) # Lưu index gốc của mô tả hợp lệ
    else:
        print(f"Bỏ qua mô tả ở index {index} do không tạo được embedding.")
    # Thêm delay nhỏ để tránh rate limit (nếu cần)
    # time.sleep(0.5) # Điều chỉnh nếu cần
    if (index + 1) % 50 == 0:
        print(f"Đã xử lý {index + 1}/{len(descriptions)} mô tả...")


if not embeddings_list:
    print("Không có embedding nào được tạo. Kết thúc chương trình.")
    exit()

# Chuyển danh sách embeddings thành mảng NumPy
embeddings_array = np.array(embeddings_list)
print(f"Đã tạo xong {len(embeddings_array)} embeddings.")

# Lưu embeddings (tùy chọn, hữu ích nếu không muốn tạo lại mỗi lần chạy)
# np.save('embeddings.npy', embeddings_array)
# df.loc[original_indices].to_pickle('original_data_with_indices.pkl') # Lưu data gốc tương ứng

# --- 3. Chức năng Tìm kiếm ---
def find_similar_profiles(query_description, data_embeddings, original_data_indices, df_original, top_n=5):
    """Tìm kiếm các hồ sơ tương đồng nhất."""
    print(f"\nĐang tìm kiếm cho mô tả: '{query_description[:100]}...'")

    # Tạo embedding cho query
    query_embedding = get_embedding(query_description, model=EMBEDDING_MODEL)
    if query_embedding is None:
        print("Không thể tạo embedding cho mô tả tìm kiếm.")
        return []

    query_embedding_np = np.array(query_embedding).reshape(1, -1) # Reshape thành 2D array

    # Tính Cosine Similarity
    # cosine_similarity trả về ma trận, ta lấy hàng đầu tiên
    similarities = cosine_similarity(query_embedding_np, data_embeddings)[0]

    # Lấy index của top N kết quả có độ tương đồng cao nhất
    # argsort trả về index từ thấp đến cao, nên ta lấy từ cuối lên
    top_n_indices_in_embeddings = np.argsort(similarities)[-top_n:][::-1]

    results = []
    for i in top_n_indices_in_embeddings:
        original_df_index = original_data_indices[i] # Lấy index gốc trong DataFrame
        similarity_score = similarities[i]
        profile_detail = df_original.loc[original_df_index, DETAIL_COLUMN]
        # Lấy thêm thông tin khác nếu muốn
        # other_info = df_original.loc[original_df_index, 'TenNguoiDang']

        results.append({
            'index_goc': original_df_index,
            'do_tuong_dong': similarity_score,
            'chi_tiet': profile_detail
            # 'thong_tin_khac': other_info
        })

    return results

# --- 4. Sử dụng chức năng tìm kiếm ---
# Ví dụ của bạn
input_description = "Tôi tên Sơn, bị thất lạc gia đình do chạy loạn chiến tranh tại Việt Nam năm 1976. Tôi nhớ tôi lúc đó khoảng 6 tuổi, mẹ tôi tên Hoa, ba tôi tên Hoàng."

# Nếu bạn đã lưu embeddings và data gốc:
# embeddings_array = np.load('embeddings.npy')
# original_data_with_indices = pd.read_pickle('original_data_with_indices.pkl')
# original_indices = original_data_with_indices.index.tolist()
# df = original_data_with_indices # Hoặc đọc lại df gốc nếu cần

# Thực hiện tìm kiếm
search_results = find_similar_profiles(input_description, embeddings_array, original_indices, df, top_n=5)

# Hiển thị kết quả
print("\n--- KẾT QUẢ TÌM KIẾM ---")
if search_results:
    for rank, result in enumerate(search_results, 1):
        print(f"\nHạng {rank}:")
        print(f"  - Index gốc trong CSV: {result['index_goc']}")
        print(f"  - Độ tương đồng (Cosine Similarity): {result['do_tuong_dong']:.4f}")
        print(f"  - Chi tiết hồ sơ gốc:\n    {result['chi_tiet']}")
else:
    print("Không tìm thấy hồ sơ nào phù hợp.")