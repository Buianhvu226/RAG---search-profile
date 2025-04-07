import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import google.generativeai as genai
import re
import os

# Cấu hình API Key cho Gemini
GOOGLE_API_KEY = "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"  # Thay bằng API Key của bạn
genai.configure(api_key=GOOGLE_API_KEY)

# Đọc dữ liệu
@st.cache_data
def load_data():
    return pd.read_csv("profiles_detailed_data_cleaned.csv")

df = load_data()
important_fields = ["Họ và tên", "Năm sinh", "Tên cha", "Tên mẹ", "Tên anh-chị-em", "Chi tiết", "Năm thất lạc"]

# Hàm chuẩn hóa tên
def normalize_name(name):
    if pd.isna(name) or name == 'nan':
        return ""
    name = str(name).lower().strip()
    name = re.sub(r'[^\wÀ-ỹ\s]', '', name)
    return name

# Hàm chuẩn hóa năm
def extract_years(text):
    if pd.isna(text) or text == 'nan':
        return []
    text = str(text)
    short_years = re.findall(r'năm\s+(\d{2})', text)
    full_years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
    
    result = []
    for year in short_years:
        if int(year) < 50:
            result.append("20" + year)
        else:
            result.append("19" + year)
    result.extend(full_years)
    return result

# Hàm trích xuất quan hệ gia đình
def extract_family_relations(text):
    if pd.isna(text) or text == 'nan':
        return []
    text = str(text).lower()
    relations = []
    relation_patterns = [
        r'(cha|ba|bố|bác)\s+(tên|là)\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})',
        r'(mẹ|má|u)\s+(tên|là)\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})',
        r'(anh|chị|em)\s+(tên|là)\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})',
        r'(con\s+trai|con\s+gái)\s+(tên|là)\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})',
        r'(tên\s+thật\s+là|tên\s+là)\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})'
    ]
    for pattern in relation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) >= 2:
                relation = match[0]
                name = match[-1]
                relations.append(f"{relation}_{name}")
    return relations

# Hàm tiền xử lý và tăng trọng số cho text
def enhance_text(row):
    name = str(row["Họ và tên"])
    birth_year = str(row["Năm sinh"])
    father = normalize_name(row["Tên cha"])
    mother = normalize_name(row["Tên mẹ"])
    siblings = normalize_name(row["Tên anh-chị-em"])
    details = str(row["Chi tiết"])
    lost_year = str(row["Năm thất lạc"])
    
    name_parts = name.split()
    name_emphasis = name_parts[-1].lower() if name_parts else ""
    
    years = extract_years(details)
    years.extend(extract_years(lost_year))
    years.extend(extract_years(birth_year))
    years_text = " ".join(years)
    
    locations = re.findall(r'(?:tại|ở|tỉnh|thành phố|quận|huyện|xã)\s+([A-ZÀ-Ỹ][a-zà-ỹ]*(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]*)*)', details)
    location_text = " ".join(locations) if locations else ""
    
    family_relations = extract_family_relations(details)
    family_text = " ".join(family_relations)
    
    real_name_match = re.search(r'tên\s+thật\s+(?:là)?\s+([A-ZÀ-Ỹ][a-zà-ỹ]*(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]*){0,2})', details)
    real_name = real_name_match.group(1).lower() if real_name_match else ""
    
    circumstance_keywords = ["nhận nuôi", "cho người", "đưa đi", "đã mất", "thất lạc", "li dị", "ly dị", "ly hôn", "giận nhau"]
    circumstance_text = " ".join([kw for kw in circumstance_keywords if kw in details.lower()])
    
    enhanced_text = f"{name.lower()} {name_emphasis} {name_emphasis} {real_name} {real_name} {real_name} "
    enhanced_text += f"{father} {father} {mother} {siblings} {years_text} {years_text} "
    enhanced_text += f"{location_text} {location_text} {family_text} {circumstance_text} {circumstance_text} "
    enhanced_text += f"{details.lower()}"
    return enhanced_text

# Tạo text với trọng số tăng cường
df["enhanced_text"] = df.apply(enhance_text, axis=1)

# Sử dụng mô hình đa ngôn ngữ
@st.cache_resource
def load_model():
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    return SentenceTransformer(model_name)

model = load_model()

# Khởi tạo ChromaDB
@st.cache_resource
def init_chroma():
    persist_directory = "chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_directory,
        tenant="default_tenant",
        database="default_database"
    )
    collection = client.get_or_create_collection("enhanced_profiles_v2")
    
    if collection.count() == 0:
        for index, row in df.iterrows():
            vector = model.encode(row["enhanced_text"]).tolist()
            collection.add(
                ids=[str(index)],
                embeddings=[vector],
                metadatas=[{"link": row["Link"], "title": row["Tiêu đề"]}]
            )
    return collection

collection = init_chroma()

# Hàm tiền xử lý truy vấn bằng Gemini
def preprocess_query_with_gemini(query):
    prompt = f"""
    Trích xuất các thông tin chính từ đoạn văn sau bằng tiếng Việt, tập trung vào các chi tiết quan trọng của một người thất lạc:
    "{query}"
    
    Hãy trích xuất càng chi tiết càng tốt:
    1. Tên người cần tìm (bao gồm cả tên thật nếu có nhiều tên khác nhau)
    2. Năm sinh (nếu có)
    3. Năm thất lạc hoặc năm nhận nuôi
    4. Tên cha/mẹ đẻ
    5. Tên người nhận nuôi (nếu có)
    6. Địa điểm liên quan
    7. Anh chị em (nếu có)
    8. Hoàn cảnh thất lạc (nhận nuôi, ly hôn, v.v.)
    
    Trả về dưới dạng một đoạn văn ngắn gọn ghi rõ thông tin, với các từ khóa quan trọng được nhắc lại 2-3 lần để nhấn mạnh (tên thật, tên cha mẹ, năm thất lạc). Không bỏ sót thông tin nào.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Không thể xử lý truy vấn do lỗi: {str(e)}"

# Hàm đánh giá độ tương đồng
def calculate_similarity_score(query_info, profile):
    score = 0
    matches = []
    query_lower = query_info.lower()
    profile_text = (str(profile["Họ và tên"]) + " " + 
                   str(profile["Chi tiết"]) + " " + 
                   str(profile["Tên cha"]) + " " + 
                   str(profile["Tên mẹ"])).lower()
    
    name_matches = re.findall(r'tên\s+(?:thật|gốc)?\s+(?:là)?\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})', query_lower)
    names = [match.strip() for match in name_matches]
    
    for name in names:
        if name and len(name) > 2 and name in profile_text:
            score += 3
            matches.append(f"Tên '{name}' xuất hiện trong hồ sơ")
    
    years = extract_years(query_info)
    for year in years:
        if year in profile_text:
            score += 2
            matches.append(f"Năm {year} xuất hiện trong hồ sơ")
    
    father_matches = re.findall(r'(?:ba|cha|bố|bác)\s+(?:tên|là)\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})', query_lower)
    mother_matches = re.findall(r'(?:mẹ|má)\s+(?:tên|là)\s+([a-zà-ỹ]+(?:\s+[a-zà-ỹ]+){0,2})', query_lower)
    
    father_name = normalize_name(profile["Tên cha"])
    mother_name = normalize_name(profile["Tên mẹ"])
    
    for father in father_matches:
        if father and father.strip() in father_name:
            score += 3
            matches.append(f"Tên cha '{father.strip()}' khớp với hồ sơ")
    
    for mother in mother_matches:
        if mother and mother.strip() in mother_name:
            score += 3
            matches.append(f"Tên mẹ '{mother.strip()}' khớp với hồ sơ")
    
    circumstances = ["nhận nuôi", "cho người", "thất lạc", "li dị", "ly dị", "ly hôn", "giận nhau"]
    for circumstance in circumstances:
        if circumstance in query_lower and circumstance in profile_text:
            score += 1
            matches.append(f"Hoàn cảnh '{circumstance}' xuất hiện trong cả hai")
    
    return score, matches

# Hàm tìm kiếm với VectorDB
def search_profiles(query, top_k=30):
    processed_query = preprocess_query_with_gemini(query)
    st.write("**Thông tin đã trích xuất:**", processed_query)
    
    query_vector = model.encode(processed_query).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    matched_profiles = []
    if "ids" in results and results["ids"] and "metadatas" in results:
        for i, idx in enumerate(results["ids"][0]):
            profile = df.iloc[int(idx)].to_dict()
            profile["link"] = results["metadatas"][0][i]["link"]
            similarity_score, match_reasons = calculate_similarity_score(query + " " + processed_query, profile)
            profile["score"] = similarity_score
            profile["match_reasons"] = match_reasons
            matched_profiles.append(profile)
    
    matched_profiles.sort(key=lambda x: -x["score"])
    filtered_profiles = [p for p in matched_profiles if p["score"] >= 1]
    return filtered_profiles if filtered_profiles else matched_profiles[:5]

# Hàm sinh câu trả lời bằng Gemini
def generate_response_with_gemini(query, profiles):
    profiles_text = "\n\n".join([
        f"Hồ sơ {i+1} (Điểm: {p['score']}):\nTiêu đề: {p['Tiêu đề']}\nHọ và tên: {p['Họ và tên']}\n" +
        f"Năm sinh: {p['Năm sinh']}\nTên cha: {p['Tên cha']}\nTên mẹ: {p['Tên mẹ']}\n" +
        f"Tên anh-chị-em: {p['Tên anh-chị-em']}\nChi tiết: {p['Chi tiết']}\n" +
        f"Năm thất lạc: {p['Năm thất lạc']}\nLý do khớp: {', '.join(p['match_reasons'])}" 
        for i, p in enumerate(profiles[:5])
    ])
    
    prompt = f"""
    Dựa trên truy vấn: "{query}"
    
    Và các hồ sơ sau:
    {profiles_text}
    
    Hãy phân tích mức độ phù hợp giữa truy vấn và từng hồ sơ. Sắp xếp hồ sơ theo thứ tự phù hợp nhất.
    
    Đặc biệt chú ý những yếu tố quan trọng:
    1. Tên người cần tìm (tên thật)
    2. Năm thất lạc/nhận nuôi
    3. Tên cha/mẹ
    4. Hoàn cảnh thất lạc
    
    Với mỗi hồ sơ, nêu lý do tại sao hồ sơ này phù hợp hoặc không phù hợp với truy vấn.
    Đề xuất thêm thông tin người dùng có thể cung cấp để tìm kiếm chính xác hơn.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Không thể tạo phản hồi do lỗi API: {str(e)}"

# Hàm kiểm tra hồ sơ cụ thể
def check_specific_profile(query, profile_id, top_results):
    target_profile = None
    target_rank = -1
    for i, profile in enumerate(top_results):
        if "MS" + profile_id in profile["Tiêu đề"]:
            target_profile = profile
            target_rank = i + 1
            break
    
    if target_profile is None:
        target_df = df[df["Tiêu đề"].str.contains("MS" + profile_id, na=False)]
        if not target_df.empty:
            target_profile = target_df.iloc[0].to_dict()
            similarity_score, match_reasons = calculate_similarity_score(query, target_profile)
            target_profile["score"] = similarity_score
            target_profile["match_reasons"] = match_reasons
    return target_profile, target_rank

# Giao diện Streamlit
st.title("Hệ thống tìm kiếm người thân thất lạc (Tối ưu hóa)")

query = st.text_area("Nhập thông tin bạn nhớ về người thân:", height=150)
top_k = st.slider("Số lượng kết quả tìm kiếm:", min_value=5, max_value=50, value=30)
target_profile_id = st.text_input("MS hồ sơ mục tiêu (không bắt buộc):", "")

if st.button("Tìm kiếm"):
    if query:
        with st.spinner('Đang tìm kiếm... Vui lòng đợi trong giây lát'):
            matching_profiles = search_profiles(query, top_k=top_k)
            if matching_profiles:
                st.success(f"Tìm thấy {len(matching_profiles)} kết quả phù hợp")
                
                if target_profile_id:
                    target_profile, target_rank = check_specific_profile(query, target_profile_id, matching_profiles)
                    if target_profile is not None:
                        if target_rank > 0:
                            st.info(f"✅ Hồ sơ mục tiêu MS{target_profile_id} được tìm thấy ở vị trí thứ {target_rank} trong kết quả")
                        else:
                            st.warning(f"❌ Hồ sơ mục tiêu MS{target_profile_id} không nằm trong top {top_k} kết quả")
                            st.json({
                                "Tiêu đề": target_profile["Tiêu đề"],
                                "Họ và tên": target_profile["Họ và tên"],
                                "Điểm tương đồng": target_profile["score"],
                                "Lý do": target_profile["match_reasons"] if "match_reasons" in target_profile else []
                            })
                
                response = generate_response_with_gemini(query, matching_profiles)
                st.write("**Phân tích kết quả:**")
                st.write(response)
                
                with st.expander("Xem chi tiết các hồ sơ", expanded=True):
                    for i, profile in enumerate(matching_profiles):
                        st.markdown(f"### {i+1}. **{profile['Tiêu đề']}** (Điểm: {profile['score']})")
                        st.markdown(f"**Họ và tên:** {profile['Họ và tên']}")
                        st.markdown(f"**Năm sinh:** {profile['Năm sinh']}")
                        st.markdown(f"**Tên cha:** {profile['Tên cha']}")
                        st.markdown(f"**Tên mẹ:** {profile['Tên mẹ']}")
                        st.markdown(f"**Tên anh-chị-em:** {profile['Tên anh-chị-em']}")
                        st.markdown(f"**Chi tiết:** {profile['Chi tiết']}")
                        st.markdown(f"**Năm thất lạc:** {profile['Năm thất lạc']}")
                        st.markdown(f"**Link:** [{profile['link']}]({profile['link']})")
                        st.markdown(f"**Lý do phù hợp:**")
                        for reason in profile['match_reasons']:
                            st.markdown(f"- {reason}")
                        st.markdown("---")
            else:
                st.error("Không tìm thấy hồ sơ phù hợp.")
    else:
        st.warning("Vui lòng nhập thông tin để tìm kiếm.")