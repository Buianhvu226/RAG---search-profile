import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import google.generativeai as genai
import re
import os
import time # Để đo thời gian
import concurrent.futures # Cho đa luồng
from threading import Lock # Để bảo vệ truy cập vào tài nguyên dùng chung (nếu cần)

st.set_page_config(layout="wide")
# --- Cấu hình và Khởi tạo ---

# !!! QUAN TRỌNG: Thay thế bằng API Key thực của bạn !!!
GOOGLE_API_KEY = "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64" # <--- THAY BẰNG API KEY CỦA BẠN
# Chọn model Gemini phù hợp (Flash thường nhanh và rẻ hơn)
GEMINI_MODEL_NAME = 'gemini-2.0-flash' # Hoặc 'gemini-2.0-advanced-latest'

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Kiểm tra sự tồn tại của model (tùy chọn)
    # genai.get_generative_model(GEMINI_MODEL_NAME)
except Exception as e:
    st.error(f"Lỗi cấu hình Google API Key hoặc model '{GEMINI_MODEL_NAME}': {e}. Hãy đảm bảo bạn đã cung cấp API Key hợp lệ và model tồn tại.")
    st.stop()

DATA_FILE = "profiles_detailed_data_cleaned.csv"
PERSIST_DIRECTORY = "chroma_db_v3_gemini" # Thư mục DB mới cho phiên bản này
MAX_WORKERS_GEMINI = 8 # Số luồng tối đa để gọi Gemini API đồng thời (điều chỉnh nếu gặp lỗi rate limit)

# Hàm tải dữ liệu
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file dữ liệu tại '{file_path}'.")
        st.stop()

df = load_data(DATA_FILE)

# --- Các Hàm Tiền xử lý (Giữ nguyên từ phiên bản trước) ---

def normalize_name(name):
    if pd.isna(name) or name == 'nan': return ""
    name = str(name).lower().strip()
    name = re.sub(r'[^\wÀ-ỹ\s-]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def extract_years(text):
    if pd.isna(text) or text == 'nan': return []
    text = str(text).lower()
    years = []
    full_years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
    years.extend(full_years)
    short_years = re.findall(r'năm\s+(\d{2})\b', text)
    current_year = pd.Timestamp.now().year
    for year in short_years:
        year_int = int(year)
        if year_int < ((current_year % 100) + 10): full_year = "20" + year
        else: full_year = "19" + year
        if int(full_year) <= current_year: years.append(full_year)
    return sorted(list(set(years)))

def enhance_text(row):
    name = normalize_name(row["Họ và tên"])
    birth_year = str(row["Năm sinh"])
    father = normalize_name(row["Tên cha"])
    mother = normalize_name(row["Tên mẹ"])
    siblings = normalize_name(row["Tên anh-chị-em"])
    details = str(row["Chi tiết"])
    lost_year = str(row["Năm thất lạc"])

    detail_years = extract_years(details)
    all_years = extract_years(birth_year) + extract_years(lost_year) + detail_years
    years_text = " ".join(sorted(list(set(all_years))))

    locations = re.findall(r'(?:ở|tại|quê\s+ở|tỉnh|tp\.|thành\s+phố|quận|huyện|xã)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)', details)
    location_text = " ".join(sorted(list(set(l.strip() for l in locations)))).lower()

    real_name_match = re.search(r'(?:tên\s+thật|tên\s+gọi)\s+(?:là)?\s+([A-ZÀ-Ỹ][a-zà-ỹ]*(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]*){0,2})', details, re.IGNORECASE)
    real_name = normalize_name(real_name_match.group(1)) if real_name_match else ""

    circumstance_keywords = ["nhận nuôi", "cho người", "cho đi", "bị bỏ", "thất lạc", "li dị", "ly dị", "ly hôn", "giận nhau", "mồ côi", "đi lạc", "bán con"]
    circumstance_text = " ".join([kw for kw in circumstance_keywords if kw in details.lower()])

    enhanced = f"Tên: {name}. Tên thật/gọi: {real_name}. Cha: {father}. Mẹ: {mother}. "
    enhanced += f"Năm sinh/thất lạc/sự kiện: {years_text}. "
    enhanced += f"Anh chị em: {siblings}. "
    enhanced += f"Địa điểm: {location_text}. Hoàn cảnh: {circumstance_text}. "
    enhanced += f"Chi tiết khác: {details.lower()}"
    return enhanced

# Áp dụng hàm enhance_text (chỉ chạy nếu cột chưa có)
if "enhanced_text" not in df.columns or df["enhanced_text"].isnull().any():
     print("Đang tạo/cập nhật cột enhanced_text...")
     df["enhanced_text"] = df.apply(enhance_text, axis=1)
     print("Tạo/cập nhật cột enhanced_text hoàn tất.")
     # Lưu lại df đã xử lý để lần sau chạy nhanh hơn (tùy chọn)
     # df.to_csv("profiles_processed_v3.csv", index=False)


# --- Mô hình Embedding và VectorDB (Giữ nguyên) ---

@st.cache_resource
def load_embedding_model():
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Lỗi tải mô hình Sentence Transformer '{model_name}': {e}")
        st.stop()

embedding_model = load_embedding_model()

@st.cache_resource
def init_chroma_db():
    print(f"Khởi tạo hoặc tải ChromaDB từ: {PERSIST_DIRECTORY}")
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection_name = "profiles_semantic_v1" # Đổi tên collection
        collection = client.get_or_create_collection(collection_name)

        if collection.count() < len(df):
            st.warning(f"Collection '{collection_name}' đang trống hoặc chưa đủ dữ liệu. Tiến hành nạp...")
            if collection.count() > 0:
                print(f"Xóa dữ liệu cũ trong collection '{collection_name}'...")
                existing_ids = collection.get(include=[])['ids']
                if existing_ids: collection.delete(ids=existing_ids)

            batch_size = 100
            total_rows = len(df)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:min(i + batch_size, total_rows)]
                ids = [str(idx) for idx in batch_df.index]
                texts_to_embed = batch_df["enhanced_text"].fillna("").tolist()
                embeddings = embedding_model.encode(texts_to_embed).tolist()
                metadatas = batch_df[["Link", "Tiêu đề"]].fillna("N/A").to_dict('records')

                if len(ids) == len(embeddings) == len(metadatas):
                    try: collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                    except Exception as add_err: st.error(f"Lỗi khi thêm batch vào ChromaDB: {add_err}"); continue
                else: st.error(f"Lỗi dữ liệu không khớp kích thước ở batch {i}"); continue

                percent_complete = min((i + batch_size) / total_rows, 1.0)
                progress_bar.progress(percent_complete)
                status_text.text(f"Đã nạp {min(i + batch_size, total_rows)} / {total_rows} hồ sơ.")

            status_text.text(f"Nạp dữ liệu cho collection '{collection_name}' hoàn tất.")
            st.success(f"Đã tạo và nạp dữ liệu cho collection '{collection_name}' thành công!")
        else:
            print(f"Collection '{collection_name}' đã có đủ dữ liệu ({collection.count()} hồ sơ).")
        return collection
    except Exception as e:
        st.error(f"Lỗi khởi tạo hoặc kết nối ChromaDB: {e}")
        st.stop()

collection = init_chroma_db()

# --- Hàm Xử lý Truy vấn và Đánh giá bằng Gemini ---

# Hàm tiền xử lý truy vấn (có thể giữ nguyên hoặc cải thiện prompt)
def preprocess_query_with_gemini(query):
    prompt = f"""
    Phân tích và tóm tắt thông tin chính từ truy vấn tìm người thân sau đây thành một đoạn văn ngắn gọn, mạch lạc bằng tiếng Việt. Tập trung vào các thực thể quan trọng: tên (người tìm, người được tìm, cha, mẹ, anh chị em, người nuôi...), năm (sinh, mất, thất lạc, sự kiện...), địa điểm (quê quán, nơi ở, nơi thất lạc...), hoàn cảnh đặc biệt.

    Truy vấn gốc:
    "{query}"

    Bản tóm tắt súc tích:
    """
    try:
        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = llm.generate_content(prompt)
        return response.text if response.parts else query
    except Exception as e:
        st.warning(f"Lỗi khi xử lý truy vấn bằng Gemini: {e}. Sử dụng truy vấn gốc.")
        return query

# *** HÀM ĐÁNH GIÁ NGỮ NGHĨA BẰNG GEMINI ***
def evaluate_match_with_gemini(query, profile_dict):
    """
    Sử dụng Gemini để đánh giá mức độ phù hợp ngữ nghĩa giữa truy vấn và hồ sơ.
    Trả về điểm số (0-10) và giải thích.
    """
    # Tạo prompt chi tiết
    profile_text_for_prompt = f"""
    Tiêu đề: {profile_dict.get('Tiêu đề', 'N/A')}
    Họ và tên: {profile_dict.get('Họ và tên', 'N/A')}
    Năm sinh: {profile_dict.get('Năm sinh', 'N/A')}
    Tên cha: {profile_dict.get('Tên cha', 'N/A')}
    Tên mẹ: {profile_dict.get('Tên mẹ', 'N/A')}
    Anh chị em: {profile_dict.get('Tên anh-chị-em', 'N/A')}
    Năm thất lạc: {profile_dict.get('Năm thất lạc', 'N/A')}
    Chi tiết: {profile_dict.get('Chi tiết', 'N/A')}
    """

    prompt = f"""
    Bạn là chuyên gia phân tích thông tin tìm kiếm người thân. Hãy đánh giá mức độ phù hợp về mặt NGỮ NGHĨA và CHI TIẾT CỐT LÕI giữa [Truy vấn Tìm kiếm] và [Hồ sơ Ứng viên] dưới đây cho mục đích tìm lại người thân.

    [Truy vấn Tìm kiếm]:
    "{query}"

    [Hồ sơ Ứng viên]:
    {profile_text_for_prompt}

    Hãy phân tích kỹ các yếu tố sau và cho điểm mức độ quan trọng/khớp của từng yếu tố:
    1.  **Tên người được tìm:** So sánh tên (tên thật, tên gọi khác) trong truy vấn với tên trong hồ sơ. Mức độ khớp? (Rất quan trọng)
    2.  **Tên cha/mẹ:** So sánh tên cha/mẹ. Mức độ khớp? (Rất quan trọng)
    3.  **Năm thất lạc/Năm sự kiện chính:** So sánh năm thất lạc hoặc các năm sự kiện quan trọng khác. Mức độ khớp? (Rất quan trọng)
    4.  **Hoàn cảnh thất lạc:** So sánh ý nghĩa của hoàn cảnh được mô tả. Có tương đồng không? (Quan trọng)
    5.  **Năm sinh:** So sánh năm sinh (nếu có). Mức độ khớp? (Quan trọng)
    6.  **Địa điểm:** So sánh địa điểm quê quán, nơi thất lạc, nơi ở. Có liên quan không? (Khá quan trọng)
    7.  **Anh chị em:** So sánh thông tin về anh chị em. Có khớp tên nào không? (Khá quan trọng)
    8.  **Mâu thuẫn:** Có thông tin nào trong hồ sơ mâu thuẫn rõ ràng với truy vấn không? (Yếu tố giảm điểm)

    Dựa trên sự cân nhắc tầm quan trọng và mức độ khớp/mâu thuẫn của các yếu tố trên, hãy cho **Điểm phù hợp tổng thể** từ 0 đến 10 (10 là khớp hoàn hảo về ngữ nghĩa và các chi tiết quan trọng nhất) và một **Giải thích ngắn gọn** lý do cho điểm số đó.

    Chỉ trả lời theo định dạng sau:
    ĐIỂM: [Điểm từ 0-10]
    GIẢI THÍCH: [Giải thích ngắn gọn, tập trung vào lý do chính]
    """
    try:
        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
        # Thêm cài đặt an toàn nếu cần, ví dụ chặn nội dung không phù hợp
        # safety_settings=[...]
        response = llm.generate_content(prompt) #, safety_settings=safety_settings)

        # Phân tích phản hồi của Gemini
        text_response = response.text
        score_match = re.search(r"ĐIỂM:\s*(\d+)", text_response)
        explanation_match = re.search(r"GIẢI THÍCH:\s*(.+)", text_response, re.DOTALL) # DOTALL để khớp qua nhiều dòng

        score = int(score_match.group(1)) if score_match else 0
        explanation = explanation_match.group(1).strip() if explanation_match else "Lỗi: Không thể trích xuất giải thích từ Gemini."

        # Giới hạn điểm trong khoảng 0-10
        score = max(0, min(10, score))

        return score, explanation

    except Exception as e:
        print(f"Lỗi khi gọi Gemini để đánh giá hồ sơ {profile_dict.get('Tiêu đề', 'N/A')}: {e}")
        # Trả về điểm thấp và thông báo lỗi nếu không đánh giá được
        return 0, f"Lỗi API khi đánh giá: {e}"

# --- Hàm Tìm kiếm Chính (Tích hợp Đa luồng) ---

def search_profiles_multithreaded(query, initial_top_k=50, final_top_k=10, min_score=4):
    """
    Tìm kiếm hồ sơ:
    1. Tiền xử lý truy vấn bằng Gemini.
    2. Tìm kiếm vector top `initial_top_k`.
    3. Lọc sơ bộ các ứng viên tiềm năng
    4. Đánh giá ngữ nghĩa từng ứng viên bằng Gemini (đa luồng + hàng loạt).
    5. Sắp xếp theo điểm Gemini và trả về top `final_top_k`.
    """
    start_time = time.time()
    st.info("Bước 1: Phân tích và chuẩn hóa truy vấn...")
    processed_query = preprocess_query_with_gemini(query)
    st.write("**Thông tin trích xuất (dùng để tìm kiếm vector):**", processed_query if processed_query != query else "(Không trích xuất được thêm, dùng truy vấn gốc)")
    query_for_embedding = processed_query # Sử dụng truy vấn đã xử lý cho vector search

    st.info(f"Bước 2: Tìm kiếm {initial_top_k} hồ sơ tiềm năng bằng Vector Search...")
    try:
        query_vector = embedding_model.encode(query_for_embedding).tolist()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=initial_top_k,
            include=["metadatas"] # Chỉ cần metadata để lấy link/tiêu đề
        )
    except Exception as e:
        st.error(f"Lỗi khi truy vấn ChromaDB: {e}")
        return [], None

    retrieved_ids = results.get("ids", [[]])[0]
    if not retrieved_ids:
        st.warning("Vector Search không trả về kết quả nào.")
        return [], None

    # Lấy dữ liệu đầy đủ của các ứng viên từ DataFrame gốc
    retrieved_indices = [int(id_str) for id_str in retrieved_ids if id_str.isdigit()]
    candidate_profiles_df = df.iloc[retrieved_indices].copy()

    # Gắn link/tiêu đề từ metadata
    metadata_map = {id_str: meta for id_str, meta in zip(results["ids"][0], results["metadatas"][0])}
    candidate_profiles_df['Link'] = candidate_profiles_df.index.map(lambda idx: metadata_map.get(str(idx), {}).get('Link', ''))
    candidate_profiles_df['Tiêu đề'] = candidate_profiles_df.index.map(lambda idx: metadata_map.get(str(idx), {}).get('Tiêu đề', candidate_profiles_df.loc[idx, 'Tiêu đề']))

    # ------- PHẦN MỚI: LỌC SƠ BỘ ỨNG VIÊN -------
    total_candidates = len(retrieved_indices)
    st.info(f"Bước 3: Sàng lọc sơ bộ {total_candidates} hồ sơ trước khi phân tích chi tiết...")

    # Trích xuất từ khóa quan trọng nhất từ truy vấn
    key_names = re.findall(r'tên\s+(?:là)?\s+([A-ZÀ-Ỹa-zà-ỹ]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ]+){0,2})', query, re.IGNORECASE)
    key_years = extract_years(query)
    parent_names = (
        re.findall(r'(?:cha|ba|bố)\s+(?:tên|là)\s+([A-ZÀ-Ỹa-zà-ỹ]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ]+){0,2})', query, re.IGNORECASE) + 
        re.findall(r'(?:mẹ|má)\s+(?:tên|là)\s+([A-ZÀ-Ỹa-zà-ỹ]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ]+){0,2})', query, re.IGNORECASE)
    )
    
    # Lọc sơ bộ các ứng viên có khả năng cao
    filtered_candidates = []
    
    for index, profile_row in candidate_profiles_df.iterrows():
        profile_dict = profile_row.to_dict()
        prefilter_score = 0
        
        # Kiểm tra từng từ khóa trong profile
        profile_text = (str(profile_dict.get("Họ và tên", "")) + " " + 
                        str(profile_dict.get("Chi tiết", "")) + " " + 
                        str(profile_dict.get("Tên cha", "")) + " " + 
                        str(profile_dict.get("Tên mẹ", ""))).lower()
        
        # Cho điểm sơ bộ dựa trên tên
        for name in key_names:
            if normalize_name(name) in profile_text:
                prefilter_score += 3
        
        # Cho điểm sơ bộ dựa trên năm
        for year in key_years:
            if year in profile_text:
                prefilter_score += 2
        
        # Cho điểm sơ bộ dựa trên tên cha mẹ
        for parent in parent_names:
            if normalize_name(parent) in profile_text:
                prefilter_score += 3
        
        # Chỉ giữ lại các profile có điểm sơ bộ >= 2
        if prefilter_score >= 2:
            filtered_candidates.append((query, profile_dict))
    
    # Thống kê lọc sơ bộ
    filtered_count = len(filtered_candidates)
    if filtered_count == 0:
        # Nếu không có ứng viên nào qua lọc sơ bộ, lấy 10 ứng viên tốt nhất theo vector
        filtered_candidates = [(query, candidate_profiles_df.iloc[i].to_dict()) for i in range(min(10, len(candidate_profiles_df)))]
        filtered_count = len(filtered_candidates)
        st.warning(f"Không có ứng viên nào qua được bộ lọc sơ bộ. Chọn {filtered_count} hồ sơ tốt nhất theo vector.")
    else:
        st.info(f"Sau khi lọc sơ bộ: {filtered_count}/{total_candidates} hồ sơ được đưa tới Gemini API")
    
    # ------- PHẦN MỚI: ĐÁNH GIÁ HÀNG LOẠT -------
    st.info(f"Bước 4: Đánh giá chi tiết {filtered_count} hồ sơ bằng AI (Gemini) - Sử dụng đánh giá hàng loạt...")
    
    BATCH_SIZE = 3  # Đánh giá 3 hồ sơ mỗi lần gọi API
    batched_tasks = []
    for i in range(0, len(filtered_candidates), BATCH_SIZE):
        batch = filtered_candidates[i:i+BATCH_SIZE]
        batched_tasks.append(batch)
    
    # Chuyển đổi batch profile thành text để gọi Gemini API
    def evaluate_batch_with_gemini(batch):
        """Đánh giá một nhóm profile cùng lúc"""
        query_data, profile_dicts = batch[0][0], [item[1] for item in batch]  # Lấy query từ batch đầu tiên
        
        # Tạo dữ liệu batch để gửi đi
        profiles_text = ""
        for i, profile in enumerate(profile_dicts):
            profiles_text += f"""
            === HỒ SƠ {i+1} ===
            Tiêu đề: {profile.get('Tiêu đề', 'N/A')}
            Họ và tên: {profile.get('Họ và tên', 'N/A')}
            Năm sinh: {profile.get('Năm sinh', 'N/A')}
            Tên cha: {profile.get('Tên cha', 'N/A')}
            Tên mẹ: {profile.get('Tên mẹ', 'N/A')}
            Chi tiết: {profile.get('Chi tiết', 'N/A')}
            Năm thất lạc: {profile.get('Năm thất lạc', 'N/A')}
            """
        
        # Prompt để đánh giá nhiều profile cùng lúc
        prompt = f"""
        Bạn là chuyên gia phân tích thông tin tìm kiếm người thân. Hãy đánh giá mức độ phù hợp về mặt NGỮ NGHĨA và CHI TIẾT CỐT LÕI giữa [Truy vấn Tìm kiếm] và các [Hồ sơ Ứng viên] dưới đây:

        [Truy vấn Tìm kiếm]:
        "{query_data}"

        [Các Hồ sơ Ứng viên]:
        {profiles_text}

        Cho mỗi hồ sơ, hãy phân tích và đánh giá:
        1. Tên người được tìm
        2. Tên cha/mẹ
        3. Năm thất lạc/sự kiện
        4. Hoàn cảnh thất lạc
        5. Năm sinh
        6. Các yếu tố khác có liên quan

        Trả về kết quả theo định dạng chính xác sau đây (cho từng hồ sơ):
        
        HỒ SƠ 1:
        ĐIỂM: [0-10]
        GIẢI THÍCH: [Giải thích ngắn gọn]
        
        HỒ SƠ 2:
        ĐIỂM: [0-10]
        GIẢI THÍCH: [Giải thích ngắn gọn]
        
        [vv...]
        """
        
        try:
            llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
            response = llm.generate_content(prompt)
            text_response = response.text
            
            # Parse kết quả
            results = []
            pattern = r"HỒ SƠ (\d+):\s*ĐIỂM:\s*(\d+)\s*GIẢI THÍCH:\s*(.+?)(?=HỒ SƠ \d+:|$)"
            matches = re.finditer(pattern, text_response, re.DOTALL)
            
            for match in matches:
                profile_num = int(match.group(1))
                if profile_num <= len(profile_dicts):  # Đảm bảo index hợp lệ
                    score = int(match.group(2))
                    explanation = match.group(3).strip()
                    results.append((score, explanation))
            
            # Đảm bảo đủ kết quả cho tất cả profile trong batch
            while len(results) < len(profile_dicts):
                results.append((0, "Lỗi: Không phân tích được"))
                
            return list(zip(profile_dicts, [r[0] for r in results], [r[1] for r in results]))
        
        except Exception as e:
            print(f"Lỗi khi đánh giá batch: {e}")
            # Trả về điểm 0 cho tất cả profile trong batch nếu lỗi
            return [(p, 0, f"Lỗi API: {str(e)}") for p in profile_dicts]
    
    # Thực hiện đánh giá theo batch đa luồng
    evaluated_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    evaluated_count = 0
    lock = Lock()
    
    def process_batch(batch):
        nonlocal evaluated_count
        batch_results = evaluate_batch_with_gemini(batch)
        with lock:
            evaluated_count += len(batch)
            progress = min(evaluated_count / filtered_count, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Đang đánh giá: {evaluated_count}/{filtered_count} hồ sơ...")
        return batch_results
    
    start_eval_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_GEMINI) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batched_tasks}
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results = future.result()
                evaluated_results.extend(batch_results)
            except Exception as exc:
                original_batch = future_to_batch[future]
                print(f"Batch có {len(original_batch)} hồ sơ tạo ra lỗi: {exc}")
                # Thêm lỗi vào kết quả
                for _, profile_dict in original_batch:
                    evaluated_results.append((profile_dict, 0, f"Lỗi đánh giá batch: {exc}"))
                with lock:
                    evaluated_count += len(original_batch)
    
    eval_duration = time.time() - start_eval_time
    status_text.text(f"Hoàn tất đánh giá {filtered_count} hồ sơ trong {eval_duration:.2f} giây.")

    # Tập hợp kết quả và tạo DataFrame mới
    final_profiles_data = []
    for profile_dict, score, explanation in evaluated_results:
        profile_dict['gemini_score'] = score
        profile_dict['gemini_explanation'] = explanation
        final_profiles_data.append(profile_dict)

    if not final_profiles_data:
        st.warning("Không có hồ sơ nào được đánh giá thành công.")
        return [], None

    final_df = pd.DataFrame(final_profiles_data)

    # Sắp xếp theo điểm Gemini và lọc
    final_df.sort_values(by='gemini_score', ascending=False, inplace=True)
    filtered_df = final_df[final_df['gemini_score'] >= min_score]

    # Giới hạn số lượng kết quả cuối cùng
    top_profiles = filtered_df.head(final_top_k).to_dict('records')

    total_duration = time.time() - start_time
    st.success(f"Bước 5: Hoàn tất! Tìm thấy {len(top_profiles)} hồ sơ phù hợp nhất (điểm AI >= {min_score}). Tổng thời gian: {total_duration:.2f} giây.")

    # Trả về DataFrame đã lọc và sắp xếp để hàm check_specific_profile sử dụng
    return top_profiles, final_df

# --- Hàm Sinh Phản hồi Tổng hợp (Dựa trên kết quả Gemini) ---
def generate_final_response_with_gemini(query, profiles_with_scores):
    if not profiles_with_scores:
        return "Không tìm thấy hồ sơ nào có điểm phù hợp đủ cao sau khi AI phân tích."

    # Chỉ lấy top 5 để đưa vào prompt
    top_profiles_for_prompt = profiles_with_scores[:5]

    profiles_text = "\n\n".join([
        f"Hồ sơ {i+1} (Điểm AI: {p['gemini_score']}/10):\n" +
        f"  Tiêu đề: {p.get('Tiêu đề', 'N/A')}\n" +
        f"  Họ và tên: {p.get('Họ và tên', 'N/A')}\n" +
        # f"  Năm sinh: {p.get('Năm sinh', 'N/A')}\n" + # Có thể bỏ bớt nếu quá dài
        # f"  Tên cha: {p.get('Tên cha', 'N/A')}\n" +
        # f"  Tên mẹ: {p.get('Tên mẹ', 'N/A')}\n" +
        f"  Giải thích của AI: {p.get('gemini_explanation', 'Không có giải thích.')}"
        for i, p in enumerate(top_profiles_for_prompt)
    ])

    prompt = f"""
    Dựa trên truy vấn tìm kiếm người thân sau:
    "{query}"

    Và đây là các hồ sơ có điểm phù hợp cao nhất sau khi được AI (Gemini) phân tích chi tiết về ngữ nghĩa và các yếu tố quan trọng (tên, năm, hoàn cảnh...):
    {profiles_text}

    Hãy đưa ra một phân tích tổng hợp cho người dùng:
    1.  Nhận xét về hồ sơ tiềm năng nhất (có điểm AI cao nhất). Tại sao AI lại đánh giá cao hồ sơ đó dựa trên giải thích đã có?
    2.  Liệt kê ngắn gọn 1-2 hồ sơ khác cũng có tiềm năng (nếu có).
    3.  Nếu có sự không chắc chắn hoặc cần thêm thông tin, hãy đề xuất cho người dùng.

    Viết câu trả lời bằng tiếng Việt, giọng văn cảm thông, rõ ràng và tập trung vào kết quả phân tích của AI.
    """
    try:
        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi tạo phản hồi tổng hợp bằng Gemini: {e}")
        return "Đã xảy ra lỗi khi tạo phân tích kết quả."

# --- Hàm kiểm tra hồ sơ cụ thể (cần gọi evaluate_match_with_gemini) ---
def check_specific_profile(query, profile_id_input, all_evaluated_df):
    target_profile_display_info = None
    profile_id_str = profile_id_input.upper()
    if not profile_id_str.startswith("MS"): profile_id_str = "MS" + profile_id_str

    # 1. Tìm trong DataFrame đã được đánh giá bởi Gemini (nếu có)
    if all_evaluated_df is not None and not all_evaluated_df.empty:
        target_in_evaluated = all_evaluated_df[all_evaluated_df['Tiêu đề'].str.contains(profile_id_str, na=False)]
        if not target_in_evaluated.empty:
            target_data = target_in_evaluated.iloc[0].to_dict()
            # Tìm vị trí rank trong DataFrame đã sort theo gemini_score
            try:
                 # Cần reset index để .get_loc hoạt động đúng nếu index không liên tục
                 rank = all_evaluated_df.reset_index(drop=True).index[all_evaluated_df['Tiêu đề'].str.contains(profile_id_str, na=False)][0] + 1
            except IndexError:
                 rank = -1 # Không tìm thấy (trường hợp hiếm)

            target_profile_display_info = {
                "data": target_data, # Chứa cả score và explanation
                "rank": rank,
                "source": "evaluated"
            }
            print(f"Tìm thấy hồ sơ mục tiêu {profile_id_str} trong danh sách đã đánh giá.")
            return target_profile_display_info

    # 2. Nếu không có trong danh sách đã đánh giá, tìm trong DF gốc và đánh giá bằng Gemini
    print(f"Hồ sơ mục tiêu {profile_id_str} không có trong top {len(all_evaluated_df) if all_evaluated_df is not None else 0} đã đánh giá. Tìm trong DF gốc...")
    target_df = df[df["Tiêu đề"].str.contains(profile_id_str, na=False)]
    if not target_df.empty:
        target_profile_dict = target_df.iloc[0].to_dict()
        st.info(f"Đang đánh giá hồ sơ mục tiêu '{profile_id_str}' bằng AI...")
        # Gọi hàm đánh giá (không dùng đa luồng ở đây vì chỉ có 1 hồ sơ)
        gemini_score, gemini_explanation = evaluate_match_with_gemini(query, target_profile_dict)
        st.info(f"Đánh giá hồ sơ mục tiêu '{profile_id_str}' hoàn tất.")

        # Thêm thông tin đánh giá vào dict
        target_profile_dict['gemini_score'] = gemini_score
        target_profile_dict['gemini_explanation'] = gemini_explanation

        target_profile_display_info = {
            "data": target_profile_dict,
            "rank": -1, # Không có rank vì không nằm trong danh sách sort ban đầu
            "source": "on_demand"
        }
        print(f"Đã đánh giá hồ sơ mục tiêu {profile_id_str} (ngoài top): Điểm {gemini_score}")
    else:
        print(f"Không tìm thấy hồ sơ mục tiêu {profile_id_str} trong toàn bộ dữ liệu.")

    return target_profile_display_info


# --- Giao diện Streamlit ---


st.title("🔍 Hệ thống Tìm kiếm Người thân Thất lạc - Phân tích Ngữ nghĩa AI (v3)")
st.markdown("""
Nhập thông tin bạn nhớ về người thân. AI (Gemini) sẽ **phân tích sâu về ngữ nghĩa** để tìm các hồ sơ phù hợp nhất.
*Càng nhiều chi tiết (tên thật, tên cha mẹ, năm, nơi ở, hoàn cảnh...) AI càng dễ phân tích chính xác.*
""")
st.warning("⚠️ **Lưu ý:** Việc sử dụng AI để phân tích chi tiết từng hồ sơ có thể mất thời gian và tốn chi phí API nhiều hơn.")

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area("Nhập thông tin tìm kiếm:", height=200, key="query_input", placeholder="Ví dụ: Tìm anh trai tên Dũng, cha tên Sên, mẹ tên Vạn, quê Bến Tre, bị cho đi nuôi ở Vĩnh Long khoảng năm 1981 do cha mẹ giận nhau...")
    target_profile_id = st.text_input("Kiểm tra hồ sơ cụ thể (Nhập MSxxxx):", "", key="target_id_input")

with col2:
    st.subheader("Tùy chọn tìm kiếm:")
    initial_k = st.slider("Số hồ sơ quét ban đầu (Vector Search):", min_value=20, max_value=200, value=50, step=10, key="initial_k", help="Số lượng ứng viên tiềm năng lấy ra từ Vector Search để AI đánh giá chi tiết.")
    final_k = st.slider("Số kết quả hiển thị cuối cùng:", min_value=5, max_value=30, value=10, step=5, key="final_k", help="Số lượng hồ sơ tốt nhất (theo điểm AI) được hiển thị.")
    min_score_threshold = st.slider("Ngưỡng điểm AI tối thiểu:", min_value=0, max_value=10, value=5, step=1, key="min_score_gemini", help="Chỉ hiển thị các hồ sơ có điểm đánh giá của AI lớn hơn hoặc bằng ngưỡng này (Thang điểm 0-10).")

if st.button("Tìm kiếm bằng AI", key="search_button"):
    if query:
        all_evaluated_profiles_df = None # Khởi tạo để kiểm tra target
        matching_profiles = [] # Khởi tạo
        with st.spinner(f'🚀 Bắt đầu tìm kiếm... (Quét {initial_k} hồ sơ, đánh giá bằng AI có thể mất vài phút)'):
            # 1. Tìm kiếm và đánh giá đa luồng
            matching_profiles, all_evaluated_profiles_df = search_profiles_multithreaded(
                query,
                initial_top_k=initial_k,
                final_top_k=final_k,
                min_score=min_score_threshold
            )

            # 2. Kiểm tra hồ sơ mục tiêu (nếu có)
            target_display_info = None
            if target_profile_id:
                with st.spinner(f"Kiểm tra và đánh giá hồ sơ mục tiêu '{target_profile_id}'..."):
                     target_display_info = check_specific_profile(query, target_profile_id, all_evaluated_profiles_df)

        # --- Hiển thị kết quả ---
        st.markdown("---")
        st.header("📊 Kết quả Tìm kiếm và Phân tích AI")

        # Hiển thị thông tin hồ sơ mục tiêu
        if target_display_info:
            st.subheader(f"Thông tin Hồ sơ Mục tiêu ({target_profile_id})")
            target_data = target_display_info['data']
            target_rank = target_display_info['rank']
            target_source = target_display_info['source']

            if target_rank > 0:
                st.success(f"✅ Hồ sơ được tìm thấy ở vị trí thứ **{target_rank}** trong danh sách đã đánh giá.")
            elif target_source == "on_demand":
                 st.warning(f"⚠️ Hồ sơ được tìm thấy trong dữ liệu gốc nhưng không nằm trong top {initial_k} ứng viên ban đầu hoặc không đạt ngưỡng điểm.")
            else: # source == 'evaluated' nhưng rank = -1 (hiếm)
                 st.warning(f"⚠️ Hồ sơ có trong danh sách đánh giá nhưng không xác định được vị trí.")


            st.markdown(f"**Tiêu đề:** {target_data.get('Tiêu đề', 'N/A')}")
            st.markdown(f"**Họ và tên:** {target_data.get('Họ và tên', 'N/A')}")
            st.markdown(f"**Điểm đánh giá AI:** **{target_data.get('gemini_score', 'N/A')} / 10**")
            st.markdown(f"**Giải thích của AI:** {target_data.get('gemini_explanation', 'N/A')}")
            st.markdown(f"**Link gốc:** [{target_data.get('Link', 'N/A')}]({target_data.get('Link', 'N/A')})")
            # st.json(target_data) # Hiện toàn bộ data để debug (nếu cần)
            st.markdown("---")
        elif target_profile_id: # Nếu nhập ID mà không tìm thấy
             st.warning(f"Không tìm thấy hồ sơ nào có mã số '{target_profile_id}' trong toàn bộ dữ liệu.")
             st.markdown("---")


        # Hiển thị phân tích tổng hợp từ Gemini
        if matching_profiles:
            st.subheader("🤖 Phân tích Tổng hợp từ AI")
            with st.spinner("AI đang tóm tắt kết quả..."):
                 gemini_final_analysis = generate_final_response_with_gemini(query, matching_profiles)
                 st.markdown(gemini_final_analysis)
            st.markdown("---")

            # Hiển thị danh sách các hồ sơ phù hợp
            st.subheader(f"Danh sách Top {len(matching_profiles)} hồ sơ phù hợp nhất (Điểm AI >= {min_score_threshold})")
            for i, profile in enumerate(matching_profiles):
                # Sử dụng st.container để nhóm thông tin của mỗi hồ sơ
                with st.container():
                    st.markdown(f"### {i+1}. **{profile.get('Tiêu đề', 'Hồ sơ không có tiêu đề')}**")
                    st.markdown(f"**Điểm đánh giá AI:** <span style='color: red; font-weight: bold;'>{profile.get('gemini_score', 0)} / 10</span>", unsafe_allow_html=True)

                    col_a, col_b = st.columns([1, 2]) # Cột giải thích rộng hơn
                    with col_a:
                         st.markdown(f"**Họ và tên:** {profile.get('Họ và tên', 'N/A')}")
                         st.markdown(f"**Năm sinh:** {profile.get('Năm sinh', 'N/A')}")
                         st.markdown(f"**Năm thất lạc:** {profile.get('Năm thất lạc', 'N/A')}")
                         st.markdown(f"**Tên cha:** {profile.get('Tên cha', 'N/A')}")
                         st.markdown(f"**Tên mẹ:** {profile.get('Tên mẹ', 'N/A')}")
                         st.markdown(f"**Link gốc:** [{profile.get('Link', 'N/A')}]({profile.get('Link', 'N/A')})")

                    with col_b:
                        st.markdown(f"**Giải thích của AI:**")
                        st.info(f"{profile.get('gemini_explanation', 'Không có giải thích.')}")

                    with st.expander("Xem chi tiết gốc của hồ sơ"):
                        st.markdown(f"**Anh chị em:** {profile.get('Tên anh-chị-em', 'N/A')}")
                        st.markdown(f"**Chi tiết:** {profile.get('Chi tiết', 'Không có chi tiết.')}")
                    st.markdown("---") # Đường kẻ ngăn cách giữa các hồ sơ

        elif not target_display_info: # Chỉ hiển thị nếu không có KQ nào và cũng ko có target phù hợp
            st.error("Rất tiếc, sau khi AI phân tích chi tiết, không tìm thấy hồ sơ nào phù hợp với thông tin bạn cung cấp và đạt ngưỡng điểm tối thiểu.")
            st.info("Mẹo: Hãy thử cung cấp thêm chi tiết, kiểm tra lỗi chính tả trong truy vấn, hoặc giảm nhẹ ngưỡng điểm AI tối thiểu.")

    else:
        st.warning("⚠️ Vui lòng nhập thông tin vào ô tìm kiếm.")

# --- Thông tin Sidebar ---
st.sidebar.title("Giới thiệu")
st.sidebar.info(
    "Hệ thống này sử dụng Vector Search để tìm ứng viên tiềm năng và sau đó dùng AI **Gemini** để **phân tích ngữ nghĩa sâu**, so sánh chi tiết truy vấn và hồ sơ, nhằm đưa ra kết quả chính xác nhất.\n\n"
    "**Đa luồng** được sử dụng để tăng tốc quá trình phân tích của AI."
)
st.sidebar.warning("Việc phân tích sâu bằng AI có thể **tốn thời gian** và **chi phí API** cao hơn.")
st.sidebar.title("Cần cài đặt")
st.sidebar.markdown("Để chạy ứng dụng này cục bộ:")
st.sidebar.code("""
pip install streamlit pandas chromadb \
sentence-transformers google-generativeai \
thefuzz python-Levenshtein
""")
st.sidebar.markdown(f"Và thay thế `YOUR_GOOGLE_API_KEY` bằng API Key của bạn. Đảm bảo model `{GEMINI_MODEL_NAME}` có sẵn cho key của bạn.")
st.sidebar.markdown(f"Số luồng gọi AI tối đa: `{MAX_WORKERS_GEMINI}` (có thể điều chỉnh trong code).")