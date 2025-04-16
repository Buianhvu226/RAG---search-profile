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
GOOGLE_API_KEY = "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM" # <--- CHÚ Ý: Thay thế bằng API key của bạn!

# Đường dẫn file và tên cột
CSV_PATH = 'F:\\missing_people(NCHCCCL)\\data\\profiles_detailed_data_semantically_cleaned.csv'
DETAIL_COLUMN_NAME = 'Chi tiet_sach' # Cột chứa văn bản cần embedding
EMBEDDING_FILE = 'gemini_embeddings_details_only.npy' # File lưu embeddings
INDEX_FILE = 'original_indices_details_only.npy' # File lưu index gốc tương ứng

# Cấu hình cho xác minh LLM
GEMINI_API_KEYS = [
    "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM",
    "AIzaSyDw2a1VhB3MXps3ldFUMyYvi65OTIMqFfM",
    "AIzaSyDats92Eac1yPpk4Z9soGf4nCCiBTh1P64", 
    "AIzaSyBVEQzc89kQ1072ji4xR9wMPtBlzvqCIlY",
    "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64",
    "AIzaSyDoT41uDC4u212LEnJPS0BPmKKjI4QyWZA"
]
# Tối ưu hóa cấu hình để sử dụng tốt hơn các API key
BATCH_SIZE = 3  # Tăng từ 2 lên 3 hồ sơ mỗi batch
MAX_CONCURRENT_REQUESTS = 6  # Tăng lên 6 để sử dụng tất cả các API key
MAX_RETRIES = 5  # Maximum number of retries for API calls
INITIAL_RETRY_DELAY = 5  # Increased from 2 to 5 seconds initial delay
BATCH_GROUP_DELAY = 3  # 3 giây giữa các nhóm batch

# --- Khởi tạo Google AI ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Đã cấu hình Google AI API Key thành công.")
except Exception as e:
    print(f"Lỗi cấu hình Google AI API Key: {e}")
    print("Vui lòng kiểm tra lại API Key hoặc kết nối mạng.")
    exit()

# --- Hàm xác minh hồ sơ bằng LLM ---
def verify_profiles_with_llm(query, profiles, api_key):
    """Verify if profiles match the query using Gemini API."""
    # Create profile strings with proper escaping
    profile_strings = []
    for p in profiles:
        detail = str(p.get('Chi tiet_sach', 'N/A')).replace('\\', '/')[:500]  # Replace backslashes
        profile_str = f"""
Index: {p.name}
Tiêu đề: {p.get('Tiêu đề', 'N/A')}
Họ tên: {p.get('Họ và tên', 'N/A')}
Chi tiết: {detail}
{"-"*40}"""
        profile_strings.append(profile_str)

    prompt = f"""Bạn là 1 nhà phân tích hồ sơ rất chuyên nghiệp. Hãy phân tích các hồ sơ đăng ký tìm người thân thất lạc sau đây và xác định xem có khớp với yêu cầu tìm kiếm không.
    Hãy so sánh các tên riêng, địa danh, năm sinh, địa chỉ, đặc điểm nhận dạng, ký ức hoặc các thông tin khác để xác định sự khớp.
    Mỗi hồ sơ có Index gốc, Tiêu đề, Họ tên và Chi tiết. Trả về danh sách các Index gốc của hồ sơ phù hợp.
    *Lưu ý: Những hồ sơ nào có những tên riêng hoặc danh từ riêng không hề khớp với yêu cầu tìm kiếm thì không cần phân tích (Hồ sơ không phù hợp). Hãy kiểm tra thật kỹ !

    Yêu cầu tìm kiếm: {query}

    Các hồ sơ cần kiểm tra:
    {"-"*40}
    {"".join(profile_strings)}

    Hãy trả về chỉ các Index gốc của hồ sơ phù hợp nhất, mỗi index trên một dòng. Nếu không có hồ sơ nào phù hợp, trả về 'none'. Chỉ trả về hồ sơ nào thật sự phù hợp nhất !
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            
            if response.text:
                matched_indices = [int(idx.strip()) for idx in response.text.split('\n') if idx.strip().isdigit()]
                return matched_indices
        except Exception as e:
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"Quota exhausted for API key. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                print(f"Lỗi khi gọi Gemini API: {e}")
                break  # Break on non-quota errors
    
    return []

def parallel_verify(query, ranked_profiles, max_profiles=300):
    """Perform parallel verification of profiles using multiple API keys."""
    from concurrent.futures import ThreadPoolExecutor
    
    # Use the parameter to control how many profiles to process
    max_profiles = min(max_profiles, len(ranked_profiles))
    ranked_profiles = ranked_profiles[:max_profiles]
    print(f"Xử lý {max_profiles} hồ sơ có điểm số cao nhất để xác minh bằng LLM")
    
    # Split profiles into batches
    batches = [ranked_profiles[i:i + BATCH_SIZE] 
               for i in range(0, len(ranked_profiles), BATCH_SIZE)]
    
    print(f"Chia {len(ranked_profiles)} hồ sơ thành {len(batches)} batch, mỗi batch {BATCH_SIZE} hồ sơ")
    
    verified_indices = []
    
    # Sử dụng tất cả các API key có sẵn
    num_api_keys = len(GEMINI_API_KEYS)
    print(f"Sử dụng {num_api_keys} API key để xử lý song song")
    
    # Process batches in smaller groups to avoid quota issues
    batch_groups = [batches[i:i + MAX_CONCURRENT_REQUESTS] 
                   for i in range(0, len(batches), MAX_CONCURRENT_REQUESTS)]
    
    for group_idx, batch_group in enumerate(batch_groups):
        print(f"Đang xử lý nhóm batch {group_idx+1}/{len(batch_groups)}...")
        
        with ThreadPoolExecutor(max_workers=min(len(batch_group), MAX_CONCURRENT_REQUESTS)) as executor:
            futures = []
            for i, batch in enumerate(batch_group):
                api_key = GEMINI_API_KEYS[i % num_api_keys]
                futures.append(executor.submit(verify_profiles_with_llm, query, batch, api_key))
            
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        verified_indices.extend(result)
                except Exception as e:
                    print(f"Lỗi khi xử lý kết quả: {e}")
        
        # Add delay between batch groups to avoid quota issues
        if group_idx < len(batch_groups) - 1:
            print(f"Chờ {BATCH_GROUP_DELAY} giây trước khi xử lý nhóm batch tiếp theo...")
            time.sleep(BATCH_GROUP_DELAY)
    
    return verified_indices

# --- Hàm lấy Embedding ---
def get_embedding(text, task_type, retry_count=3, model="text-embedding-004"):
    """Lấy embedding từ Google API với cơ chế thử lại."""
    if not isinstance(text, str) or not text.strip() or pd.isna(text):
        return None

    text = text[:8000] # Giới hạn độ dài văn bản

    for attempt in range(retry_count):
        try:
            result = genai.embed_content(
                model=f"models/{model}",
                content=text,
                task_type=task_type
            )
            return result["embedding"]
        except Exception as e:
            print(f"Lỗi tạo embedding cho '{text[:50]}...' (lần {attempt + 1}/{retry_count}): {e}")
            if "API key not valid" in str(e):
                print("Lỗi API Key không hợp lệ. Vui lòng kiểm tra lại.")
                return None
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                print(f"Chờ {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"Không thể tạo embedding sau {retry_count} lần thử.")
                return None
    return None

# --- Hàm trích xuất từ khóa từ truy vấn bằng Gemini ---
def extract_keywords_gemini(query, model="gemini-2.0-flash"):
    """Trích xuất các từ khóa quan trọng từ truy vấn bằng Gemini (có ví dụ)."""
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
            keywords = [kw.strip() for kw in keywords_str.split(',') + keywords_str.split('\n') if kw.strip()]
            return keywords
        else:
            print("Gemini không trả về kết quả trích xuất từ khóa.")
            return []
    except Exception as e:
        print(f"Lỗi khi gọi Gemini để trích xuất từ khóa: {e}")
        return []

# --- Hàm Chính ---
def main():
    print("Bắt đầu quy trình tạo embeddings cho cột 'Chi tiết'...")

    # 1. Đọc dữ liệu CSV
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines='skip', encoding='utf-8')

        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()

        print(f"Đã đọc dữ liệu từ {CSV_PATH} ({len(df)} hồ sơ)")

        # Check for column 'Chi tiet_sach'
        if DETAIL_COLUMN_NAME not in df.columns:
            print(f"Lỗi: Không tìm thấy cột '{DETAIL_COLUMN_NAME}' trong file CSV.")
            print(f"Các cột hiện có: {df.columns.tolist()}")
            return

        # Quan trọng: Reset index
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

    # 2. Kiểm tra và Tải Embeddings/Indices đã lưu
    if os.path.exists(EMBEDDING_FILE) and os.path.exists(INDEX_FILE):
        print(f"Tìm thấy file embedding ({EMBEDDING_FILE}) và index ({INDEX_FILE}) đã lưu.")
        try:
            embeddings_matrix = np.load(EMBEDDING_FILE, allow_pickle=True)
            original_indices = np.load(INDEX_FILE, allow_pickle=True)
            print(f"Đã tải {len(embeddings_matrix)} embeddings và {len(original_indices)} indices.")
            if len(embeddings_matrix) != len(original_indices):
                print("Cảnh báo: Số lượng embeddings và indices không khớp! Sẽ tạo lại.")
                create_new = True
        except Exception as e:
            print(f"Lỗi khi tải file embedding hoặc index: {e}. Sẽ tạo lại.")
            create_new = True
    else:
        print("Không tìm thấy file embedding hoặc index đã lưu. Sẽ tạo mới.")
        create_new = True

    # 3. Tạo Embeddings Mới (nếu cần)
    if create_new:
        print("\nBắt đầu tạo embeddings mới cho cột '{DETAIL_COLUMN_NAME}'...")
        embeddings_list = []
        original_indices_list = []
        for index, text in tqdm(df[DETAIL_COLUMN_NAME].items(), total=len(df), desc="Đang tạo embedding"):
            emb = get_embedding(text, task_type="RETRIEVAL_DOCUMENT")
            if emb is not None:
                embeddings_list.append(emb)
                original_indices_list.append(index)
            time.sleep(0.05)

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
        except Exception as e:
            print(f"Lỗi khi lưu file embedding hoặc index: {e}")

    # 5. Kiểm tra lại embeddings_matrix và original_indices
    if embeddings_matrix is None or original_indices is None or len(embeddings_matrix) == 0:
        print("Không có embeddings để thực hiện tìm kiếm.")
        return
    if len(embeddings_matrix) != len(original_indices):
        print("Lỗi nghiêm trọng: Số lượng embeddings và indices không khớp.")
        return

    # 6. Thử nghiệm Tìm kiếm (kết hợp khớp từ khóa và vector search)
    test_search_combined(df, embeddings_matrix, original_indices)

# --- Hàm Tìm kiếm kết hợp khớp từ khóa và vector search ---
def test_search_combined(df_original, embeddings_matrix, original_indices, top_n_default=300):
    """Thực hiện tìm kiếm kết hợp khớp từ khóa và vector search."""
    print("\n--- Bắt đầu Thử nghiệm Tìm kiếm (Kết hợp Từ Khóa và Vector) ---")
    print(f"Sẵn sàng tìm kiếm trên {len(embeddings_matrix)} hồ sơ đã được mã hóa.")

    while True:
        user_query = input(f"\nNhập mô tả tìm kiếm (hoặc 'q' để thoát): ")
        if user_query.lower() == 'q':
            break

        keywords = extract_keywords_gemini(user_query)
        print("Từ khóa trích xuất từ Gemini:", keywords)

        # 1. Tìm kiếm các hồ sơ chứa ít nhất một từ khóa
        keyword_match_indices = set()
        keyword_match_counts = {}  # <--- Thêm dòng này
        if keywords:
            for keyword in keywords:
                for col in df_original.columns:
                    if df_original[col].dtype == object:
                        try:
                            matches = df_original[col].str.contains(keyword, case=False, na=False)
                            matched_indices = df_original[matches].index
                            keyword_match_indices.update(matched_indices)
                            # Đếm số lượng từ khóa khớp cho từng hồ sơ
                            for idx in matched_indices:
                                keyword_match_counts[idx] = keyword_match_counts.get(idx, 0) + 1
                        except:
                            continue

        # Lọc các indices hợp lệ
        valid_keyword_indices = [idx for idx in keyword_match_indices if idx in df_original.index]

        if not valid_keyword_indices:
            print("\nKhông tìm thấy hồ sơ nào khớp với từ khóa. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.")
            search_indices = df_original.index
            keyword_match_counts = {idx: 0 for idx in search_indices}  # Không có khớp từ khóa
        else:
            print(f"\nTìm thấy {len(valid_keyword_indices)} hồ sơ khớp với ít nhất một từ khóa. Sẽ thực hiện vector search trên các hồ sơ này.")
            search_indices = valid_keyword_indices

        # 2. Tìm kiếm vector trên các hồ sơ đã lọc (hoặc toàn bộ nếu không có khớp từ khóa)
        query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")

        if query_embedding is not None:
            query_embedding_np = np.array(query_embedding).reshape(1, -1)

            # Lấy embeddings và indices tương ứng với các hồ sơ cần tìm kiếm
            search_embeddings = []
            search_original_indices = []
            for i, orig_idx in enumerate(original_indices):
                if orig_idx in search_indices:
                    search_embeddings.append(embeddings_matrix[i])
                    search_original_indices.append(orig_idx)

            if not search_embeddings:
                print("Không có embeddings nào để tìm kiếm trong tập dữ liệu đã lọc.")
                continue

            similarities = cosine_similarity(query_embedding_np, search_embeddings)[0]

            # Tạo kết quả xếp hạng, cộng điểm thưởng cho số từ khóa khớp
            KEYWORD_BONUS = 0.15  # Hệ số điểm thưởng cho mỗi từ khóa khớp (có thể điều chỉnh)
            ranked_results = []
            for i, similarity in enumerate(similarities):
                original_df_index = search_original_indices[i]
                profile = df_original.loc[original_df_index]
                match_count = keyword_match_counts.get(original_df_index, 0)
                # Tính tổng điểm: similarity + điểm thưởng
                total_score = similarity + match_count * KEYWORD_BONUS
                ranked_results.append((match_count, similarity, total_score, profile))

            # Sắp xếp: tổng điểm giảm dần, nếu bằng nhau thì số từ khóa khớp giảm dần, rồi similarity giảm dần
            ranked_results.sort(key=lambda item: (-item[2], -item[0], -item[1]))

            print(f"\n--- Top {top_n_default} Kết quả Tìm kiếm Vector ---")
            count = 0
            for match_count, similarity, total_score, profile in ranked_results[:top_n_default]:
                original_df_index = profile.name
                print(f"  Tổng điểm: {total_score:.4f} | Số từ khóa khớp: {match_count} | Độ tương đồng: {similarity:.4f} | Index gốc: {original_df_index} | Tiêu đề: {profile.get('Tiêu đề', 'N/A')} | Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                count += 1
                if count >= 10:
                    print(f"  ... và {min(top_n_default - 10, len(ranked_results) - 10)} kết quả khác")
                    break

            # 3. Lọc kết quả bằng LLM
            print("\nĐang xác minh kết quả với Gemini LLM...")
            top_profiles = [profile for _, _, _, profile in ranked_results[:top_n_default]]
            verified_indices = parallel_verify(user_query, top_profiles, max_profiles=top_n_default)

            if verified_indices:
                print(f"\n=== {len(verified_indices)} KẾT QUẢ PHÙ HỢP NHẤT SAU KHI LỌC BẰNG LLM ===")
                for idx in verified_indices:
                    try:
                        profile = df_original.loc[idx]
                        print(f"\nIndex: {idx}")
                        print(f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                        print(f"Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                        print(f"Chi tiết: {str(profile.get(DETAIL_COLUMN_NAME, 'N/A'))[:500]}...")
                        print(f"Link: {profile.get('Link', 'N/A')}")
                        print("-" * 60)
                    except KeyError:
                        print(f"Lỗi: Không tìm thấy hồ sơ với index {idx}")
            else:
                print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")
        else:
            print("Không thể tạo embedding cho truy vấn.")

# --- Chạy chương trình ---
if __name__ == "__main__":
    main()
    print("\nChương trình đã kết thúc.")