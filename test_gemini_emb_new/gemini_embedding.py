import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time
from tqdm import tqdm
import google.generativeai as genai

# --- Cấu hình ---
# !!! CẢNH BÁO BẢO MẬT !!!
# KHÔNG NÊN để API Key trực tiếp trong code. Hãy sử dụng biến môi trường.
# Ví dụ: Chạy `export GOOGLE_API_KEY="YOUR_API_KEY"` trên Linux/macOS
# Hoặc `set GOOGLE_API_KEY=YOUR_API_KEY` trên Windows CMD
# Hoặc `$env:GOOGLE_API_KEY="YOUR_API_KEY"` trên PowerShell
# Sau đó đọc bằng: api_key = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64" # Thay thế bằng API key của bạn

# Đường dẫn file và tên cột
CSV_PATH = 'F:\\missing_people(NCHCCCL)\\data\\profiles_detailed_data_semantically_cleaned.csv'
DETAIL_COLUMN_NAME = 'Chi tiet_sach' # Cột chứa văn bản cần embedding
EMBEDDING_FILE = 'gemini_embeddings_details_only.npy' # File lưu embeddings
INDEX_FILE = 'original_indices_details_only.npy' # File lưu index gốc tương ứng

# --- Khởi tạo Google AI ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Đã cấu hình Google AI API Key thành công.")
except Exception as e:
    print(f"Lỗi cấu hình Google AI API Key: {e}")
    print("Vui lòng kiểm tra lại API Key hoặc kết nối mạng.")
    exit()

# --- Hàm lấy Embedding ---
def get_embedding(text, task_type, retry_count=3, model="text-embedding-004"):
    """Lấy embedding từ Google API với cơ chế thử lại.

    Args:
        text (str): Đoạn văn bản cần tạo embedding.
        task_type (str): Loại nhiệm vụ ('RETRIEVAL_DOCUMENT' hoặc 'RETRIEVAL_QUERY').
        retry_count (int): Số lần thử lại nếu gặp lỗi.
        model (str): Tên model embedding.

    Returns:
        list: Vector embedding hoặc None nếu thất bại.
    """
    if not isinstance(text, str) or not text.strip() or pd.isna(text):
        # print("Warning: Input text is empty or invalid, skipping embedding.")
        return None # Trả về None nếu text rỗng hoặc không hợp lệ

    # Giới hạn độ dài văn bản (khoảng 8192 tokens cho text-embedding-004)
    # Ước lượng đơn giản bằng ký tự, cần chặt chẽ hơn nếu muốn tối ưu
    text = text[:8000]

    for attempt in range(retry_count):
        try:
            result = genai.embed_content(
                model=f"models/{model}",
                content=text,
                task_type=task_type
            )
            return result["embedding"] # Trả về list embedding
        except Exception as e:
            print(f"Lỗi khi tạo embedding cho text '{text[:50]}...' (lần {attempt + 1}/{retry_count}): {e}")
            if "API key not valid" in str(e):
                 print("Lỗi API Key không hợp lệ. Vui lòng kiểm tra lại.")
                 return None # Dừng thử lại nếu key sai
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt # Thời gian chờ tăng dần (exponential backoff)
                print(f"Chờ {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"Không thể tạo embedding sau {retry_count} lần thử.")
                return None # Trả về None nếu vẫn lỗi sau khi thử lại
    return None # Trả về None nếu vòng lặp kết thúc mà không thành công

# --- Hàm Chính ---
def main():
    print("Bắt đầu quy trình tạo embeddings cho cột 'Chi tiết'...")

    # 1. Đọc dữ liệu CSV
    try:
        # Thêm các tham số để xử lý file CSV phức tạp nếu cần
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
        print(f"Đã đọc dữ liệu từ {CSV_PATH} ({len(df)} hồ sơ)")

        # Kiểm tra cột 'Chi tiết'
        if DETAIL_COLUMN_NAME not in df.columns:
            print(f"Lỗi: Không tìm thấy cột '{DETAIL_COLUMN_NAME}' trong file CSV.")
            print(f"Các cột hiện có: {df.columns.tolist()}")
            return

        # Quan trọng: Reset index để đảm bảo index là 0, 1, 2,...
        # Điều này giúp tránh lỗi KeyError khi dùng .loc sau này nếu index gốc bị lỗi
        df.reset_index(drop=True, inplace=True)
        print("Đã reset index của DataFrame.")
        print("Thông tin DataFrame:")
        df.info()
        print("\n5 dòng đầu:")
        print(df.head())

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}")
        return
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file CSV: {e}")
        return

    embeddings_matrix = None
    original_indices = None
    create_new = False

    # 2. Kiểm tra và Tải Embeddings/Indices đã lưu (nếu có)
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(INDEX_FILE):
        print(f"Tìm thấy file embedding ({EMBEDDING_FILE}) và index ({INDEX_FILE}) đã lưu.")
        try:
            embeddings_matrix = np.load(EMBEDDING_FILE, allow_pickle=True)
            original_indices = np.load(INDEX_FILE, allow_pickle=True)
            print(f"Đã tải {len(embeddings_matrix)} embeddings và {len(original_indices)} indices.")

            # Kiểm tra sơ bộ kích thước (có thể cần kiểm tra kỹ hơn)
            if len(embeddings_matrix) != len(original_indices):
                 print("Cảnh báo: Số lượng embeddings và indices không khớp! Sẽ tạo lại.")
                 create_new = True
            # Optional: Kiểm tra xem số indices có khớp với số dòng df không (nếu không có dòng nào bị skip)
            # elif len(original_indices) != len(df):
            #    print("Cảnh báo: Số lượng indices đã lưu không khớp với số dòng DataFrame hiện tại. Có thể cần tạo lại.")
            #    # create_new = True # Quyết định có tạo lại hay không

        except Exception as e:
            print(f"Lỗi khi tải file embedding hoặc index: {e}. Sẽ tạo lại.")
            create_new = True
    else:
        print("Không tìm thấy file embedding hoặc index đã lưu. Sẽ tạo mới.")
        create_new = True

    # 3. Tạo Embeddings Mới (nếu cần)
    if create_new:
        print("\nBắt đầu tạo embeddings mới cho cột 'Chi tiết'...")
        embeddings_list = []
        original_indices_list = []

        # Sử dụng tqdm để xem tiến độ
        # df[DETAIL_COLUMN_NAME].items() trả về (index, value)
        for index, text in tqdm(df[DETAIL_COLUMN_NAME].items(), total=len(df), desc="Đang tạo embedding"):
            emb = get_embedding(text, task_type="RETRIEVAL_DOCUMENT")

            if emb is not None:
                embeddings_list.append(emb)
                original_indices_list.append(index) # Lưu index gốc của dòng được tạo embedding

            # Thêm độ trễ nhỏ để tránh rate limit (có thể điều chỉnh)
            time.sleep(0.05) # 50ms delay

            # Lưu tạm thời (backup) - tùy chọn
            # if (len(original_indices_list) + 1) % 100 == 0:
            #     print(f"\nĐang lưu tạm {len(embeddings_list)} embeddings...")
            #     np.save(EMBEDDING_FILE + ".tmp", np.array(embeddings_list))
            #     np.save(INDEX_FILE + ".tmp", np.array(original_indices_list))

        if not embeddings_list:
            print("Không tạo được embedding nào. Kết thúc.")
            return

        embeddings_matrix = np.array(embeddings_list)
        original_indices = np.array(original_indices_list)

        print(f"\nĐã tạo xong {len(embeddings_matrix)} embeddings cho {len(original_indices)} hồ sơ hợp lệ.")

        # 4. Lưu Embeddings và Indices
        try:
            np.save(EMBEDDING_FILE, embeddings_matrix)
            np.save(INDEX_FILE, original_indices)
            print(f"Đã lưu embeddings vào {EMBEDDING_FILE}")
            print(f"Đã lưu indices gốc vào {INDEX_FILE}")
            # Xóa file tạm nếu có
            # if os.path.exists(EMBEDDING_FILE + ".tmp"): os.remove(EMBEDDING_FILE + ".tmp")
            # if os.path.exists(INDEX_FILE + ".tmp"): os.remove(INDEX_FILE + ".tmp")
        except Exception as e:
            print(f"Lỗi khi lưu file embedding hoặc index: {e}")

    # 5. Kiểm tra lại embeddings_matrix và original_indices trước khi tìm kiếm
    if embeddings_matrix is None or original_indices is None or len(embeddings_matrix) == 0:
         print("Không có embeddings để thực hiện tìm kiếm.")
         return
    if len(embeddings_matrix) != len(original_indices):
         print("Lỗi nghiêm trọng: Số lượng embeddings và indices không khớp sau khi tạo/tải.")
         return

    # 6. Thử nghiệm Tìm kiếm
    test_search(df, embeddings_matrix, original_indices)

# --- Hàm Tìm kiếm ---
def test_search(df_original, embeddings_matrix, original_indices, top_n_default=5):
    """Thực hiện tìm kiếm dựa trên embeddings."""
    print("\n--- Bắt đầu Thử nghiệm Tìm kiếm ---")
    print(f"Sẵn sàng tìm kiếm trên {len(embeddings_matrix)} hồ sơ đã được mã hóa.")

    while True:
        user_query = input(f"\nNhập mô tả tìm kiếm (hoặc 'q' để thoát): ")
        if user_query.lower() == 'q':
            break

        try:
            top_n_input = input(f"Số kết quả muốn hiển thị (mặc định {top_n_default}): ")
            top_n = int(top_n_input) if top_n_input else top_n_default
        except ValueError:
            print(f"Nhập không hợp lệ, sử dụng mặc định {top_n_default} kết quả.")
            top_n = top_n_default

        print(f"\nĐang tìm kiếm cho: '{user_query[:100]}...'")

        # Tạo embedding cho truy vấn
        query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")

        if query_embedding is None:
            print("Không thể tạo embedding cho truy vấn tìm kiếm. Vui lòng thử lại.")
            continue

        query_embedding_np = np.array(query_embedding).reshape(1, -1)

        try:
            # Tính cosine similarity
            similarities = cosine_similarity(query_embedding_np, embeddings_matrix)[0]

            # Lấy index (vị trí trong embeddings_matrix) của top_n kết quả
            # argsort trả về index từ thấp đến cao -> lấy từ cuối lên
            top_n_embedding_indices = np.argsort(similarities)[-top_n:][::-1]

            # Hiển thị kết quả
            print("\n--- Kết quả tìm kiếm ---")
            if not len(top_n_embedding_indices):
                 print("Không tìm thấy kết quả nào.")
                 continue

            for i, emb_idx in enumerate(top_n_embedding_indices):
                similarity_score = similarities[emb_idx]

                # Lấy index gốc trong DataFrame từ `original_indices`
                original_df_index = original_indices[emb_idx]

                # Truy xuất thông tin từ DataFrame gốc bằng index gốc
                try:
                    profile = df_original.loc[original_df_index]

                    print(f"\n{i + 1}. Hạng: {i+1} - Độ tương đồng: {similarity_score:.4f} (Index gốc: {original_df_index})")
                    # Sử dụng .get() để tránh lỗi nếu cột không tồn tại
                    print(f"   Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                    print(f"   Họ và tên: {profile.get('Họ và tên', 'N/A')}")

                    # Hiển thị cột 'Chi tiết' (nguồn embedding)
                    profile_detail = profile.get(DETAIL_COLUMN_NAME, 'N/A')
                    print(f"   Chi tiết: {str(profile_detail)[:300]}..." if len(str(profile_detail)) > 300 else f"   Chi tiết: {profile_detail}")

                    print(f"   Link: {profile.get('Link', 'N/A')}")
                    print("-" * 40)

                except KeyError:
                    print(f"\n{i + 1}. Lỗi: Không thể truy xuất hồ sơ với index gốc {original_df_index}. Có thể DataFrame đã thay đổi.")
                except Exception as e:
                     print(f"\n{i + 1}. Lỗi khi xử lý kết quả cho index gốc {original_df_index}: {e}")

        except Exception as e:
            print(f"Lỗi trong quá trình tính toán hoặc lấy kết quả tìm kiếm: {e}")

# --- Chạy chương trình ---
if __name__ == "__main__":
    main()
    print("\nChương trình đã kết thúc.")