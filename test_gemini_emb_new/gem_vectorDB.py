import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity # Không cần trực tiếp nữa
import numpy as np
import os
import time
from tqdm import tqdm
import google.generativeai as genai
import chromadb # Thư viện Vector DB
from chromadb.utils import embedding_functions # Để dùng embedding function của Gemini
import math # Để xử lý NaN
from concurrent.futures import ThreadPoolExecutor

# --- Cấu hình ---
# !!! CẢNH BÁO BẢO MẬT !!!
# Sử dụng biến môi trường cho API Key!
# Ví dụ Linux/macOS: export GOOGLE_API_KEY="YOUR_API_KEY"
# Ví dụ Windows CMD: set GOOGLE_API_KEY=YOUR_API_KEY
# Ví dụ PowerShell: $env:GOOGLE_API_KEY="YOUR_API_KEY"
# Sau đó đọc bằng: api_key = os.getenv("GOOGLE_API_KEY")

# Lấy API key từ biến môi trường, nếu không có thì dùng key trong code (KHÔNG KHUYẾN CÁO)
PRIMARY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM") # Thay thế nếu cần

if not PRIMARY_GOOGLE_API_KEY or PRIMARY_GOOGLE_API_KEY == "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM":
     print("CẢNH BÁO: API Key chính đang được đặt cứng trong code hoặc không tìm thấy trong biến môi trường. Vui lòng cấu hình biến môi trường GOOGLE_API_KEY.")
     # exit() # Có thể bỏ comment dòng này để bắt buộc dùng biến môi trường

# Đường dẫn file và tên cột
CSV_PATH = 'F:\\missing_people(NCHCCCL)\\data\\profiles_detailed_data_final_simplified.csv'
DETAIL_COLUMN_NAME = 'Chi tiet_merged' # Cột chứa văn bản cần embedding

# Cấu hình ChromaDB
CHROMA_PERSIST_PATH = "./chroma_db_store" # Thư mục lưu trữ dữ liệu ChromaDB
CHROMA_COLLECTION_NAME = "missing_people_profiles"
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Model dùng cho ChromaDB và get_embedding

# Cấu hình cho xác minh LLM (API Keys nên lấy từ biến môi trường hoặc file config an toàn)
GEMINI_API_KEYS = [
    PRIMARY_GOOGLE_API_KEY, # Sử dụng key chính làm key đầu tiên
    # Thêm các key khác nếu có, lý tưởng là từ biến môi trường
    os.getenv("GOOGLE_API_KEY_1", "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM"),
    os.getenv("GOOGLE_API_KEY_2", "AIzaSyDw2a1VhB3MXps3ldFUMyYvi65OTIMqFfM"),
    os.getenv("GOOGLE_API_KEY_3", "AIzaSyDats92Eac1yPpk4Z9soGf4nCCiBTh1P64"),
    os.getenv("GOOGLE_API_KEY_4", "AIzaSyBVEQzc89kQ1072ji4xR9wMPtBlzvqCIlY"),
    os.getenv("GOOGLE_API_KEY_5", "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"),
    os.getenv("GOOGLE_API_KEY_6", "AIzaSyDoT41uDC4u212LEnJPS0BPmKKjI4QyWZA")
]
# Lọc bỏ các key None hoặc trống nếu lấy từ biến môi trường không thành công
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key]
if not GEMINI_API_KEYS:
    print("Lỗi: Không có API Key nào hợp lệ cho Gemini LLM Verification. Vui lòng kiểm tra cấu hình.")
    exit()

# Cấu hình xử lý song song và retry LLM
BATCH_SIZE_LLM = 3
MAX_CONCURRENT_REQUESTS_LLM = len(GEMINI_API_KEYS) # Tận dụng tối đa số key
MAX_RETRIES_LLM = 5
INITIAL_RETRY_DELAY_LLM = 5
BATCH_GROUP_DELAY_LLM = 3

# --- Khởi tạo Google AI (cho get_embedding và extract_keywords) ---
try:
    genai.configure(api_key=PRIMARY_GOOGLE_API_KEY)
    print("Đã cấu hình Google AI API Key chính thành công.")
except Exception as e:
    print(f"Lỗi cấu hình Google AI API Key chính: {e}")
    print("Vui lòng kiểm tra lại API Key chính hoặc kết nối mạng.")
    exit()

# --- Khởi tạo ChromaDB Client và Collection ---
def initialize_vector_db():
    """Khởi tạo ChromaDB client và collection."""
    print(f"Khởi tạo ChromaDB client với đường dẫn lưu trữ: {CHROMA_PERSIST_PATH}")
    # Sử dụng PersistentClient để lưu trữ dữ liệu trên đĩa
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

    # Tạo embedding function sử dụng Google Generative AI
    # Lưu ý: Cần đảm bảo PRIMARY_GOOGLE_API_KEY hợp lệ
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=PRIMARY_GOOGLE_API_KEY, model_name=EMBEDDING_MODEL_NAME)
    print(f"Sử dụng embedding model: {EMBEDDING_MODEL_NAME}")

    # Lấy hoặc tạo collection
    # Sử dụng cosine distance để tính toán tương đồng
    try:
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=google_ef,
            metadata={"hnsw:space": "cosine"} # Chỉ định không gian đo lường là cosine
        )
        print(f"Đã kết nối/tạo collection '{CHROMA_COLLECTION_NAME}' thành công.")
        print(f"Số lượng hồ sơ hiện có trong collection: {collection.count()}")
        return collection
    except Exception as e:
        print(f"Lỗi khi kết nối/tạo collection ChromaDB: {e}")
        print("Kiểm tra lại đường dẫn lưu trữ và quyền truy cập.")
        exit()

# --- Hàm lấy Embedding (Giữ nguyên, nhưng đảm bảo model khớp với ChromaDB) ---
def get_embedding(text, task_type, retry_count=3, model=EMBEDDING_MODEL_NAME):
    """Lấy embedding từ Google API với cơ chế thử lại."""
    # Đảm bảo text là string và không rỗng/NaN
    if not isinstance(text, str) or not text.strip() or pd.isna(text):
        return None

    # Giới hạn độ dài văn bản để tránh lỗi API
    text = text[:8000]

    for attempt in range(retry_count):
        try:
            result = genai.embed_content(
                model=model, # Sử dụng model đã định nghĩa
                content=text,
                task_type=task_type
            )
            return result["embedding"]
        except Exception as e:
            print(f"Lỗi tạo embedding cho '{text[:50]}...' (lần {attempt + 1}/{retry_count}): {e}")
            if "API key not valid" in str(e):
                print("Lỗi API Key không hợp lệ. Vui lòng kiểm tra lại.")
                return None # Không thử lại nếu key sai
            # Xử lý lỗi quota (429) hoặc lỗi server (5xx)
            if "429" in str(e) or "500" in str(e) or "503" in str(e):
                 if attempt < retry_count - 1:
                    wait_time = 2 ** attempt # Exponential backoff
                    print(f"Chờ {wait_time} giây trước khi thử lại...")
                    time.sleep(wait_time)
                 else:
                    print(f"Không thể tạo embedding sau {retry_count} lần thử do lỗi API.")
                    return None
            else: # Lỗi khác không nên thử lại
                print("Lỗi không xác định, không thử lại.")
                return None
    return None

# --- Hàm Embed và Upsert dữ liệu vào ChromaDB ---
def embed_and_upsert_profiles(df, collection, batch_size_chroma=100):
    """Tạo embedding và upsert dữ liệu vào ChromaDB theo lô."""
    print(f"\nBắt đầu quá trình embedding và upsert vào collection '{collection.name}'...")
    profiles_to_upsert = []
    processed_count = 0

    total_profiles = len(df)
    progress_bar = tqdm(total=total_profiles, desc="Embedding & Upserting")

    for index, row in df.iterrows():
        text_to_embed = row.get(DETAIL_COLUMN_NAME)
        if pd.isna(text_to_embed) or not isinstance(text_to_embed, str) or not text_to_embed.strip():
            progress_bar.update(1)
            continue # Bỏ qua nếu dữ liệu không hợp lệ

        embedding = get_embedding(text_to_embed, task_type="RETRIEVAL_DOCUMENT")

        if embedding:
            # Chuẩn bị metadata - chỉ lưu các kiểu dữ liệu cơ bản
            metadata = {}
            for col in ['Tiêu đề', 'Họ và tên', 'Link', 'Năm sinh', 'Năm thất lạc']: # Thêm các cột metadata cần thiết
                if col in row:
                    value = row[col]
                    # Chuyển đổi kiểu dữ liệu không được hỗ trợ (ví dụ: NaN)
                    if pd.isna(value):
                        metadata[col] = "" # Hoặc None, nhưng string rỗng thường an toàn hơn
                    elif isinstance(value, (str, int, float, bool)):
                         metadata[col] = value
                    else:
                         metadata[col] = str(value) # Chuyển sang string nếu không chắc chắn
            # Đảm bảo không có giá trị NaN trong metadata
            metadata = {k: "" if pd.isna(v) else v for k, v in metadata.items()}


            profiles_to_upsert.append({
                "id": str(index), # ID phải là string
                "embedding": embedding,
                "metadata": metadata
                # "document": text_to_embed[:500] # Có thể lưu 1 phần document nếu muốn
            })

            # Upsert theo lô để tăng hiệu quả
            if len(profiles_to_upsert) >= batch_size_chroma:
                try:
                    collection.upsert(
                        ids=[p["id"] for p in profiles_to_upsert],
                        embeddings=[p["embedding"] for p in profiles_to_upsert],
                        metadatas=[p["metadata"] for p in profiles_to_upsert]
                        # documents=[p["document"] for p in profiles_to_upsert]
                    )
                    processed_count += len(profiles_to_upsert)
                    # print(f"Đã upsert {len(profiles_to_upsert)} hồ sơ. Tổng cộng: {processed_count}")
                    profiles_to_upsert = [] # Reset batch
                except Exception as e:
                    print(f"\nLỗi khi upsert batch vào ChromaDB: {e}")
                    # Có thể thêm xử lý lỗi chi tiết hơn ở đây
                # time.sleep(0.1) # Thêm delay nhỏ nếu gặp vấn đề về rate limit khi upsert

        progress_bar.update(1)
        # Thêm delay nhỏ giữa các lần gọi API embedding để tránh rate limit
        time.sleep(0.05) # 50ms

    # Upsert phần còn lại (nếu có)
    if profiles_to_upsert:
        try:
            collection.upsert(
                ids=[p["id"] for p in profiles_to_upsert],
                embeddings=[p["embedding"] for p in profiles_to_upsert],
                metadatas=[p["metadata"] for p in profiles_to_upsert]
                # documents=[p["document"] for p in profiles_to_upsert]
            )
            processed_count += len(profiles_to_upsert)
        except Exception as e:
             print(f"\nLỗi khi upsert batch cuối cùng vào ChromaDB: {e}")

    progress_bar.close()
    print(f"\nHoàn thành embedding và upsert. Tổng cộng {processed_count}/{total_profiles} hồ sơ hợp lệ đã được xử lý.")
    print(f"Số lượng hồ sơ cuối cùng trong collection: {collection.count()}")


# --- Hàm xác minh hồ sơ bằng LLM (Prompt đã được cải thiện ở lần trước) ---
def verify_profiles_with_llm(query, profiles_data, api_key):
    """Verify if profiles match the query using Gemini API."""
    # profiles_data bây giờ là list of dictionaries hoặc pandas Series
    profile_strings = []
    for profile in profiles_data:
        # Truy cập dữ liệu từ dict hoặc Series
        profile_id = profile.get('id') if isinstance(profile, dict) else profile.name
        title = profile.get('Tiêu đề', 'N/A')
        name = profile.get('Họ và tên', 'N/A')
        # Lấy chi tiết từ metadata nếu có, hoặc từ cột gốc nếu profile là Series
        detail = profile.get('metadata', {}).get(DETAIL_COLUMN_NAME, 'N/A') if isinstance(profile, dict) else profile.get(DETAIL_COLUMN_NAME, 'N/A')
        detail = str(detail).replace('\\', '/')[:1000] # Tăng giới hạn chi tiết

        profile_str = f"""
Index: {profile_id}
Tiêu đề: {title}
Họ tên: {name}
Chi tiết: {detail}
{"-"*40}"""
        profile_strings.append(profile_str)

    # Sử dụng prompt đã được cải thiện từ trước
    prompt = f"""Bạn là một chuyên gia phân tích hồ sơ tìm kiếm người thân thất lạc cực kỳ tỉ mỉ và chính xác. Nhiệm vụ của bạn là phân tích các hồ sơ dưới đây và chỉ xác định những hồ sơ nào mô tả **chính xác cùng một người** và **cùng một hoàn cảnh thất lạc** như được nêu trong Yêu cầu tìm kiếm.

Hãy so sánh **cực kỳ cẩn thận** các chi tiết nhận dạng cốt lõi:
- **Họ tên người thất lạc:** Phải khớp hoặc rất tương đồng.
- **Tên cha, mẹ, anh chị em (nếu có trong yêu cầu):** Phải khớp hoặc rất tương đồng.
- **Năm sinh:** Phải khớp hoặc gần đúng.
- **Quê quán/Địa chỉ liên quan:** Phải khớp hoặc có liên quan logic.
- **Hoàn cảnh thất lạc (thời gian, địa điểm, sự kiện chính):** Phải tương đồng đáng kể.

**Quy tắc loại trừ quan trọng:**
- Nếu **Họ tên người thất lạc** trong hồ sơ **khác biệt rõ ràng** so với yêu cầu, hãy **LOẠI BỎ** hồ sơ đó NGAY LẬP TỨC, bất kể các chi tiết khác có trùng khớp hay không.
- Nếu **tên cha mẹ hoặc anh chị em** (khi được cung cấp trong yêu cầu) trong hồ sơ **hoàn toàn khác biệt**, hồ sơ đó rất có thể **KHÔNG PHÙ HỢP** và cần được xem xét loại bỏ.
- Sự trùng khớp **chỉ** về địa danh hoặc năm sinh là **KHÔNG ĐỦ** để kết luận hồ sơ phù hợp nếu các tên riêng cốt lõi và hoàn cảnh thất lạc khác biệt.

Mỗi hồ sơ có Index gốc, Tiêu đề, Họ tên và Chi tiết mô tả.

Yêu cầu tìm kiếm:
{query}
------------------------------------

Các hồ sơ cần kiểm tra:
------------------------------------
{"".join(profile_strings)}
------------------------------------

Hãy trả về **chỉ các Index gốc** (là các chuỗi ID dạng số) của những hồ sơ mà bạn **rất chắc chắn** (high confidence) là phù hợp dựa trên **tất cả các tiêu chí cốt lõi** nêu trên. Mỗi index trên một dòng. Nếu không có hồ sơ nào thực sự phù hợp, trả về 'none'.
"""

    for attempt in range(MAX_RETRIES_LLM):
        try:
            # Không cần configure lại mỗi lần gọi nếu dùng cùng model
            # genai.configure(api_key=api_key) # Có thể bỏ nếu key không đổi
            model = genai.GenerativeModel('gemini-1.5-flash') # Thử nghiệm model mới hơn nếu muốn
            response = model.generate_content(prompt)

            if response.text:
                # Index trả về bây giờ là string ID, cần giữ nguyên là string
                matched_indices_str = [idx.strip() for idx in response.text.split('\n') if idx.strip().isdigit()]
                # print(f"API Key ending in ...{api_key[-4:]} returned: {matched_indices_str}") # Debug
                return matched_indices_str # Trả về list các ID dạng string
            else:
                # print(f"API Key ending in ...{api_key[-4:]} returned no text.") # Debug
                return [] # Trả về list rỗng nếu không có text
        except Exception as e:
            error_str = str(e)
            # print(f"Error with API Key ...{api_key[-4:]}: {error_str}") # Debug
            if "Resource has been exhausted" in error_str or "429" in error_str or "rate limit" in error_str.lower():
                wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                print(f"Quota/Rate Limit exhausted for API key. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{MAX_RETRIES_LLM})")
                time.sleep(wait_time)
            elif "API key not valid" in error_str:
                 print(f"Lỗi API Key không hợp lệ (Key: ...{api_key[-4:]}). Ngừng thử lại với key này.")
                 return [] # Trả về rỗng nếu key sai
            else:
                print(f"Lỗi không xác định khi gọi Gemini API: {e}")
                break # Thoát vòng lặp retry với các lỗi khác

    return [] # Trả về list rỗng nếu thất bại sau các lần thử lại

# --- Hàm xác minh song song (Cập nhật để xử lý profile_data) ---
def parallel_verify(query, ranked_profiles_data, max_profiles=300):
    """Perform parallel verification of profiles using multiple API keys."""
    # ranked_profiles_data là list các dict hoặc pandas Series
    max_profiles = min(max_profiles, len(ranked_profiles_data))
    profiles_to_verify = ranked_profiles_data[:max_profiles]
    print(f"Xử lý {max_profiles} hồ sơ có điểm số cao nhất để xác minh bằng LLM")

    if not profiles_to_verify:
        return []

    # Split profiles into batches
    batches = [profiles_to_verify[i:i + BATCH_SIZE_LLM]
               for i in range(0, len(profiles_to_verify), BATCH_SIZE_LLM)]

    print(f"Chia {len(profiles_to_verify)} hồ sơ thành {len(batches)} batch, mỗi batch tối đa {BATCH_SIZE_LLM} hồ sơ")

    verified_indices_str = set() # Dùng set để tránh trùng lặp ID

    num_api_keys = len(GEMINI_API_KEYS)
    print(f"Sử dụng {num_api_keys} API key để xử lý song song")

    # Process batches in smaller groups to avoid quota issues
    batch_groups = [batches[i:i + MAX_CONCURRENT_REQUESTS_LLM]
                    for i in range(0, len(batches), MAX_CONCURRENT_REQUESTS_LLM)]

    total_batches_processed = 0
    with tqdm(total=len(batches), desc="Verifying Batches (LLM)") as pbar_llm:
        for group_idx, batch_group in enumerate(batch_groups):
            # print(f"Đang xử lý nhóm batch {group_idx+1}/{len(batch_groups)}...")

            with ThreadPoolExecutor(max_workers=min(len(batch_group), MAX_CONCURRENT_REQUESTS_LLM)) as executor:
                futures = {}
                for i, batch in enumerate(batch_group):
                    if not batch: continue # Bỏ qua batch rỗng
                    # Luân phiên sử dụng API keys
                    api_key_index = (total_batches_processed + i) % num_api_keys
                    api_key = GEMINI_API_KEYS[api_key_index]
                    # print(f"Submitting batch {total_batches_processed + i + 1} with API key ...{api_key[-4:]}") # Debug
                    future = executor.submit(verify_profiles_with_llm, query, batch, api_key)
                    futures[future] = total_batches_processed + i + 1 # Lưu index batch để debug

                for future in futures:
                    batch_index = futures[future]
                    try:
                        result = future.result() # result là list các string ID
                        if result:
                            # print(f"Batch {batch_index} verification result: {result}") # Debug
                            verified_indices_str.update(result)
                        # else:
                            # print(f"Batch {batch_index} verification returned no matches.") # Debug
                    except Exception as e:
                        print(f"Lỗi khi xử lý kết quả của batch {batch_index}: {e}")

            total_batches_processed += len(batch_group)
            pbar_llm.update(len(batch_group))

            # Add delay between batch groups to avoid quota issues
            if group_idx < len(batch_groups) - 1:
                # print(f"Chờ {BATCH_GROUP_DELAY_LLM} giây trước khi xử lý nhóm batch tiếp theo...")
                time.sleep(BATCH_GROUP_DELAY_LLM)

    return list(verified_indices_str) # Trả về list các string ID đã xác minh


# --- Hàm trích xuất từ khóa từ truy vấn bằng Gemini (Giữ nguyên) ---
def extract_keywords_gemini(query, model="gemini-2.0-flash"): # Sử dụng model flash mới hơn
    """Trích xuất các từ khóa quan trọng từ truy vấn bằng Gemini (có ví dụ)."""
    # Prompt giữ nguyên như trước, đã được tối ưu
    prompt = f"""Phân tích các hồ sơ tìm kiếm người thân thất lạc sau và trích xuất các từ khóa quan trọng có thể dùng để tìm kiếm thông tin liên quan đến người mất tích. Trả về một danh sách các từ khóa và những từ có khả năng liên quan. Lưu ý tên riêng có thể phân tích nhỏ hơn thành tên riêng (ví dụ: Lê Thị Hạnh => Hạnh). Từ khóa liên quan có thể được sinh ra từ các từ khóa chính (ví dụ: chiến tranh => xung đột, chạy giặc, vượt biên, di cư,...) hoặc từ các từ khóa khác trong đoạn văn bản. Vậy nhiệm vụ của bạn là trích xuất các từ khóa quan trọng nhất có thể dùng để tìm kiếm thông tin liên quan đến người mất tích và các từ khóa liên quan có thể sinh ra từ các từ khóa chính. Các từ khóa này có thể là tên riêng, địa danh, năm sinh, địa chỉ, đặc điểm nhận dạng, ký ức hoặc các thông tin khác... . Hãy trả về danh sách các từ khóa và các từ khóa liên quan có thể sinh ra từ các từ khóa chính.

Ví dụ 1:
Đoạn văn bản: Chị Lê Thị Mỹ Duyên tìm bác Lê Viết Thi, đi vượt biên mất liên lạc khoảng năm 1978. Ông Lê Viết Thi sinh năm 1946, quê Quảng Nam. Bố mẹ là cụ Lê Viết Y và cụ Nguyễn Thị Ca. Anh chị em trong gia đình là Viết, Thơ, Dũng, Chung, Mười, Sỹ và Tượng. Khoảng năm 1978, ông Lê Viết Thi đi vượt biên. Từ đó, gia đình không còn nghe tin tức gì về ông.
Các từ khóa quan trọng: Lê Thị Mỹ Duyên, Duyên, Lê Viết Thi, Thi, vượt biên, di cư, chiến tranh, chạy giặc, 1978, 1946, Quảng Nam, Lê Viết Y, Y Nguyễn Thị Ca, Ca, Viết, Thơ, Dũng, Chung, Mười, Sỹ, Tượng

Ví dụ 2:
Đoạn văn bản: Chị Lê Thị Toàn tìm anh Lê Văn Thương, mất liên lạc năm 1984 tại ga Đông Hà, Quảng Trị. Vào năm 1984, gia đình ông Tiên và bà Tẻo từ Thanh Hóa di chuyển vào Quảng Trị. Khi đến ga Đông Hà (Quảng Trị), vì hoàn cảnh quá khó khăn, ông Tiên bị tật ở chân, còn bà Tẻo không minh mẫn nên bà Tèo đã mang con trai Lê Văn Thương vừa mới sinh cho một người phụ nữ ở ga Đông Hà. Người phụ nữ đó có cho bà Tẻo một ít tiền rồi ôm anh Thương đi mất.
Các từ khóa quan trọng: Lê Thị Toàn, Toàn, Lê Văn Thương, Thương, 1984, Đông Hà, Quảng Trị, Tiên, Tẻo, Thanh Hóa, Quảng Trị, di chuyển, di cư, khó khăn, thiếu thốn, nghèo khổ, tật, khiếm khuyết, không minh mẫn, thần kinh, tâm thần, mới sinh, sơ sinh, mới đẻ.

Ví dụ 3:
Đoạn văn bản: Chị Nguyễn Thị Yến tìm ba Nguyễn Văn Đã mất liên lạc năm 1977. Ông Nguyễn Văn Đã, sinh năm 1939, không rõ quê quán. Khoảng năm 1970, bà Vũ Thị Hải gặp ông Nguyễn Văn Đã ở nông trường Sao Đỏ tại Mộc Châu, Sơn La. Ông Đã phụ trách lái xe lương thực cho nông trường. Sau khi sinh chị Yến, ông muốn đưa hai mẹ con về quê ông nhưng bà Hải biết ông Đã đã có vợ ở quê nên không đồng ý và đem con về khu tập thể nhà máy nước Nam Định. Ông Đã vẫn thường lái xe về thăm con gái. Năm 1979, bà Hải mang con về quê bà sinh sống, từ đó chị Yến không hay tin gì về ba nữa.
Các từ khóa quan trọng: Nguyễn Thị Yến, Yến, Nguyễn Văn Đã, Đã, 1977, 1939, 1970, Vũ Thị Hải, Hải, nông trường, Sao Đỏ, Mộc Châu, Sơn La, lái xe, lương thực, nông trường, làm nông, nông nghiệp, khu tập thể, nhà máy nước, Nam Định, 1979

*Chú ý: những từ khóa nào phổ biến, phổ thông quá thì bỏ qua như: gia đình, anh, em, vợ, chồng, tìm kiếm, thất lạc, mất tích, mất liên lạc, không rõ quê quán, không rõ địa chỉ, không rõ thông tin, không rõ năm sinh, không rõ đặc điểm nhận dạng, không rõ ký ức...

Đoạn văn bản hiện tại:
{query}

Các từ khóa quan trọng:"""

    try:
        response = genai.GenerativeModel(model).generate_content(prompt)
        if response.text:
            keywords_str = response.text.strip()
            # Cải thiện việc tách từ khóa, xử lý cả dấu phẩy và xuống dòng
            raw_keywords = []
            for part in keywords_str.split(','):
                raw_keywords.extend(part.split('\n'))

            keywords = [kw.strip() for kw in raw_keywords if kw.strip()]
            # Loại bỏ trùng lặp và giữ nguyên thứ tự (nếu cần)
            keywords = list(dict.fromkeys(keywords))
            return keywords
        else:
            print("Gemini không trả về kết quả trích xuất từ khóa.")
            return []
    except Exception as e:
        print(f"Lỗi khi gọi Gemini để trích xuất từ khóa: {e}")
        return []


# --- Hàm Tìm kiếm kết hợp ChromaDB và Từ khóa ---
def search_combined_chroma(df_original, collection, user_query, top_n_vector=500, top_n_final=300):
    """Thực hiện tìm kiếm kết hợp Vector DB (Chroma) và khớp từ khóa."""
    print("\n--- Bắt đầu Tìm kiếm (Kết hợp Vector DB và Từ Khóa) ---")

    # 1. Trích xuất từ khóa
    keywords = extract_keywords_gemini(user_query)
    print("Từ khóa trích xuất từ Gemini:", keywords)
    if not keywords:
        print("Cảnh báo: Không trích xuất được từ khóa, kết quả có thể kém chính xác hơn.")

    # 2. Tạo embedding cho truy vấn
    query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")
    if query_embedding is None:
        print("Lỗi: Không thể tạo embedding cho truy vấn. Không thể tìm kiếm.")
        return

    # 3. Tìm kiếm trên ChromaDB
    print(f"Đang tìm kiếm {top_n_vector} hồ sơ tương đồng nhất trong Vector DB...")
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n_vector,
            include=['metadatas', 'distances'] # Lấy cả metadata và khoảng cách
        )
        # print("Kết quả thô từ ChromaDB:", results) # Debug
    except Exception as e:
        print(f"Lỗi khi truy vấn ChromaDB: {e}")
        return

    # 4. Xử lý kết quả từ ChromaDB và tính điểm kết hợp
    ranked_results = []
    if results and results.get('ids') and results['ids'][0]:
        vector_ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        print(f"Vector DB trả về {len(vector_ids)} kết quả.")

        # Tính điểm thưởng từ khóa
        KEYWORD_BONUS_FACTOR = 0.05 # Giảm hệ số bonus vì distance thường nhỏ hơn similarity
        keyword_match_counts = {}
        if keywords:
            print("Đang tính điểm khớp từ khóa cho kết quả vector search...")
            # Lấy các hồ sơ tương ứng từ DataFrame gốc để quét từ khóa
            profile_indices_int = [int(id_str) for id_str in vector_ids if id_str.isdigit()]
            relevant_profiles = df_original.loc[df_original.index.intersection(profile_indices_int)]

            for keyword in keywords:
                for col in relevant_profiles.columns:
                    if relevant_profiles[col].dtype == object:
                        try:
                            matches = relevant_profiles[col].str.contains(keyword, case=False, na=False)
                            matched_indices = relevant_profiles[matches].index
                            for idx in matched_indices:
                                keyword_match_counts[idx] = keyword_match_counts.get(idx, 0) + 1
                        except:
                            continue # Bỏ qua lỗi khi quét cột

        # Tính điểm tổng hợp: distance càng nhỏ càng tốt, bonus từ khóa làm giảm distance
        print("Đang xếp hạng kết quả kết hợp...")
        for i, profile_id_str in enumerate(vector_ids):
             try:
                 # Khoảng cách L2/Cosine distance (càng nhỏ càng tốt)
                 distance = distances[i]
                 metadata = metadatas[i]

                 # Lấy số từ khóa khớp
                 profile_idx_int = int(profile_id_str)
                 match_count = keyword_match_counts.get(profile_idx_int, 0)

                 # Tính điểm tổng hợp: distance nhỏ hơn là tốt hơn
                 # Trừ điểm bonus khỏi distance (nhưng không để âm)
                 combined_score = max(0, distance - (match_count * KEYWORD_BONUS_FACTOR))

                 # Lưu trữ thông tin cần thiết để hiển thị và lọc LLM
                 # Lấy thông tin đầy đủ từ DataFrame gốc
                 profile_data = df_original.loc[profile_idx_int].copy() # Lấy bản sao để tránh thay đổi df gốc
                 profile_data['id'] = profile_id_str # Thêm ID dạng string vào dữ liệu profile
                 # profile_data['metadata'] = metadata # Thêm metadata từ ChromaDB nếu cần

                 ranked_results.append({
                    "id": profile_id_str,
                    "distance": distance,
                    "keyword_matches": match_count,
                    "combined_score": combined_score, # Điểm này dùng để sắp xếp (càng nhỏ càng tốt)
                    "profile_data": profile_data, # Dữ liệu đầy đủ để gửi cho LLM
                    "metadata": metadata # Metadata từ chroma
                 })

             except KeyError:
                  print(f"Cảnh báo: Không tìm thấy hồ sơ với index {profile_id_str} trong DataFrame gốc.")
             except ValueError:
                  print(f"Cảnh báo: Không thể chuyển đổi ID '{profile_id_str}' thành số nguyên.")
             except IndexError:
                  print(f"Cảnh báo: Index {i} nằm ngoài phạm vi của distances/metadatas.")


        # Sắp xếp theo combined_score tăng dần (tốt nhất -> tệ nhất)
        ranked_results.sort(key=lambda item: item["combined_score"])

    else:
        print("Vector DB không trả về kết quả nào.")
        return # Không có gì để xử lý tiếp

    # 5. Hiển thị Top N kết quả trước khi lọc LLM
    print(f"\n--- Top {min(top_n_final, len(ranked_results))} Kết quả Tìm kiếm (Trước LLM) ---")
    for i, result in enumerate(ranked_results[:top_n_final]):
        profile = result["profile_data"]
        print(f"  Rank: {i+1} | Score: {result['combined_score']:.4f} | Distance: {result['distance']:.4f} | Keywords: {result['keyword_matches']} | ID: {result['id']} | Tiêu đề: {profile.get('Tiêu đề', 'N/A')} | Họ và tên: {profile.get('Họ và tên', 'N/A')}")
        if i >= 9: # Chỉ hiện 10 kết quả đầu
             print(f"  ... và {min(top_n_final - 10, len(ranked_results) - 10)} kết quả khác")
             break

    # 6. Lọc kết quả bằng LLM
    print("\nĐang xác minh kết quả với Gemini LLM...")
    profiles_for_llm = [res["profile_data"] for res in ranked_results[:top_n_final]] # Gửi dữ liệu profile đầy đủ
    verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=top_n_final) # verified_indices_str là list các string ID

    # 7. Hiển thị kết quả cuối cùng sau khi lọc LLM
    if verified_indices_str:
        print(f"\n=== {len(verified_indices_str)} KẾT QUẢ PHÙ HỢP NHẤT SAU KHI LỌC BẰNG LLM ===")
        # Lấy lại thông tin đầy đủ cho các hồ sơ đã xác minh
        verified_profiles = df_original.loc[df_original.index.astype(str).isin(verified_indices_str)]

        # Sắp xếp kết quả đã xác minh theo thứ tự ban đầu trong ranked_results (nếu muốn)
        verified_map = {str(idx): profile for idx, profile in verified_profiles.iterrows()}
        final_ranked_verified = []
        for res in ranked_results:
            if res['id'] in verified_map:
                final_ranked_verified.append(verified_map[res['id']])

        # Hiển thị kết quả đã lọc và sắp xếp lại
        for profile in final_ranked_verified:
             try:
                 print(f"\nIndex: {profile.name}") # Index gốc từ DataFrame
                 print(f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                 print(f"Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                 print(f"Chi tiết: {str(profile.get(DETAIL_COLUMN_NAME, 'N/A'))[:500]}...") # Lấy từ DataFrame gốc
                 print(f"Link: {profile.get('Link', 'N/A')}")
                 print("-" * 60)
             except Exception as e:
                 print(f"Lỗi khi hiển thị hồ sơ {profile.name}: {e}")

    else:
        print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")


# --- Hàm Chính (Cập nhật) ---
def main():
    print("Bắt đầu quy trình xử lý hồ sơ và tìm kiếm...")

    # 1. Khởi tạo Vector DB
    collection = initialize_vector_db()

    # 2. Đọc dữ liệu CSV
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
        df.columns = df.columns.str.strip() # Làm sạch tên cột
        print(f"Đã đọc dữ liệu từ {CSV_PATH} ({len(df)} hồ sơ)")

        if DETAIL_COLUMN_NAME not in df.columns:
            print(f"Lỗi: Không tìm thấy cột '{DETAIL_COLUMN_NAME}' trong file CSV.")
            print(f"Các cột hiện có: {df.columns.tolist()}")
            return

        # Quan trọng: Reset index để đảm bảo index liên tục từ 0
        # Điều này quan trọng nếu CSV gốc không có index chuẩn hoặc bị thiếu dòng
        df.reset_index(drop=True, inplace=True)
        print("Đã reset index của DataFrame.")
        print("Thông tin DataFrame sau khi reset index:")
        df.info()
        # print("\n5 dòng đầu:")
        # print(df.head())

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}")
        return
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file CSV: {e}")
        return

    # 3. Embed và Upsert dữ liệu vào ChromaDB
    # Bước này sẽ tạo embedding cho TẤT CẢ hồ sơ trong CSV và upsert vào DB.
    # ChromaDB tự xử lý việc cập nhật nếu ID đã tồn tại.
    # Trong thực tế, bạn có thể muốn có cơ chế chỉ embed và upsert
    # những hồ sơ mới hoặc đã thay đổi kể từ lần chạy trước.
    embed_and_upsert_profiles(df, collection)

    # 4. Thử nghiệm Tìm kiếm
    search_combined_chroma(df, collection, user_query=None) # Gọi vòng lặp tìm kiếm

def search_loop(df_original, collection):
     """Vòng lặp nhận truy vấn từ người dùng và thực hiện tìm kiếm."""
     while True:
        user_query = input(f"\nNhập mô tả tìm kiếm (hoặc 'q' để thoát): ")
        if user_query.lower() == 'q':
            break
        if not user_query.strip():
             print("Vui lòng nhập mô tả tìm kiếm.")
             continue

        # Gọi hàm tìm kiếm chính
        search_combined_chroma(df_original, collection, user_query)

# --- Chạy chương trình ---
if __name__ == "__main__":
    # 1. Khởi tạo Vector DB
    collection = initialize_vector_db()

    # 2. Đọc dữ liệu CSV
    try:
        df_main = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')
        df_main.columns = df_main.columns.str.strip()
        print(f"Đã đọc dữ liệu từ {CSV_PATH} ({len(df_main)} hồ sơ)")

        if DETAIL_COLUMN_NAME not in df_main.columns:
            print(f"Lỗi: Không tìm thấy cột '{DETAIL_COLUMN_NAME}'.")
            exit()

        df_main.reset_index(drop=True, inplace=True)
        print("Đã reset index của DataFrame.")
        # df_main.info() # Bỏ comment nếu muốn xem info

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CSV_PATH}")
        exit()
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file CSV: {e}")
        exit()

    # 3. Embed và Upsert (Chạy một lần để đảm bảo DB được cập nhật)
    # Trong ứng dụng thực tế, bước này có thể được tách ra hoặc chỉ chạy khi cần cập nhật DB
    run_upsert = input("Bạn có muốn chạy quá trình embedding và upsert dữ liệu vào Vector DB không? (yes/no): ").lower()
    if run_upsert == 'yes':
         embed_and_upsert_profiles(df_main, collection)
    else:
         print("Bỏ qua bước embedding/upsert. Sử dụng dữ liệu hiện có trong Vector DB.")
         print(f"Số lượng hồ sơ hiện tại trong collection '{collection.name}': {collection.count()}")


    # 4. Bắt đầu vòng lặp tìm kiếm
    search_loop(df_main, collection) # Sử dụng df_main đã đọc

    print("\nChương trình đã kết thúc.")