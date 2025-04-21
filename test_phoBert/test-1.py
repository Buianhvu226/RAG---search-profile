# Hệ thống hỗ trợ tìm kiếm người thân thất lạc bằng Vector Search

import pandas as pd
# import torch
from sentence_transformers import SentenceTransformer, util
import faiss
import google.generativeai as genai
import re

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# 1. Đọc dữ liệu CSV
# F:\missing_people(NCHCCCL)\data\processed_missing_persons.csv
df = pd.read_csv("F:\missing_people(NCHCCCL)\data\processed_missing_persons.csv" \
"" \
"")

# 2. Chọn các cột cần thiết
columns = [
    "Chi tiết", "Năm thất lạc", "Người thất lạc đi tìm gia đình",
    "Tên người thất lạc", "Tên người thân", "Quê quán", "Năm sinh",
    "Đặc điểm nhận dạng", "Ký ức", "Bối cảnh thất lạc"
]

df = df[columns].fillna("")

# 3. Gộp nội dung các trường lại thành một văn bản thống nhất để mã hóa

def combine_text(row):
    # Standardize Vietnamese names and locations
    def process_names(names):
        if isinstance(names, list):
            return ", ".join([name.strip() for name in names if pd.notna(name)])
        return str(names) if pd.notna(names) else ""
    
    return (
        f"THÔNG TIN NGƯỜI THẤT LẠC:\n"
        f"- Tên: {process_names(row['Tên người thất lạc'])}\n"
        f"- Năm sinh: {row.get('Năm sinh', '')}\n"
        f"- Năm thất lạc: {row['Năm thất lạc']}\n"
        f"- Quê quán: {process_names(row['Quê quán'])}\n"
        f"- Người thân: {process_names(row['Tên người thân'])}\n"
        f"- Đặc điểm: {row['Đặc điểm nhận dạng']}\n"
        f"- Ký ức: {row['Ký ức']}\n"
        f"- Bối cảnh: {row['Bối cảnh thất lạc']}\n"
        f"CHI TIẾT:\n{row['Chi tiết']}"
    )

texts = df.apply(combine_text, axis=1).tolist()

# 4. Mã hóa văn bản bằng SentenceTransformer với mô hình tiếng Việt
# Better Vietnamese language models to consider:
# 1. Current model (good balance of speed/accuracy)
model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

# 2. For better accuracy (but slower)
# model = SentenceTransformer("keepitreal/vietnamese-sbert")

# 3. For best accuracy (requires GPU)
# model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

print("Đang mã hóa vector hồ sơ...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

# 5. Tạo FAISS index để tìm kiếm nhanh
index = faiss.IndexFlatIP(embeddings.shape[1])
embeddings_np = embeddings.cpu().numpy()
faiss.normalize_L2(embeddings_np)
index.add(embeddings_np)

# 6. Hàm tìm kiếm hồ sơ gần nhất theo mô tả
def preprocess_query(query):
    """Use Gemini to extract structured information from search query"""
    prompt = f"""
    Phân tích chi tiết câu truy vấn tìm kiếm người thất lạc bằng tiếng Việt này và trích xuất thông tin quan trọng theo định dạng JSON:
    {query}
    
    Các trường cần trích xuất:
    - name: Tên đầy đủ hoặc một phần tên được đề cập
    - year: Năm hoặc khoảng thời gian thất lạc
    - location: Địa điểm hoặc nơi được đề cập
    - relatives: Tên bất kỳ thành viên gia đình nào được đề cập
    - features: Đặc điểm nhận dạng nổi bật
    - memories: Ký ức cụ thể được đề cập
    - context: Bối cảnh/hoàn cảnh thất lạc
    
    Hãy phân tích kỹ lưỡng và đầy đủ, đảm bảo không bỏ sót thông tin quan trọng.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')  # Using more capable model
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        start = response.text.find('{')
        end = response.text.rfind('}') + 1
        json_str = response.text[start:end]
        
        # Safely parse JSON
        import json
        query_info = json.loads(json_str)
        
        # Ensure all required fields exist
        required_fields = ['name', 'year', 'location', 'relatives', 'features', 'memories', 'context']
        for field in required_fields:
            if field not in query_info:
                query_info[field] = ''
                
        return query_info
        
    except Exception as e:
        print(f"Gemini query analysis failed: {str(e)}")
        return {
            'name': '', 'year': '', 'location': '',
            'relatives': '', 'features': '',
            'memories': '', 'context': ''
        }

def search_missing_profiles(query, top_k=5):
    if not query or not isinstance(query, str):
        return pd.DataFrame(), []
        
    try:
        # Extract structured info from query
        query_info = preprocess_query(query)
        print("Extracted query information:", query_info)
        
        # Create enhanced query text with more weight on important fields
        enhanced_query = (
            f"THÔNG TIN TÌM KIẾM NGƯỜI THẤT LẠC:\n"
            f"- Tên người: {query_info['name']}\n"
            f"- Năm thất lạc: {query_info['year']}\n"
            f"- Địa điểm: {query_info['location']}\n"
            f"- Người thân: {query_info['relatives']}\n"
            f"- Đặc điểm nhận dạng: {query_info['features']}\n"
            f"- Ký ức: {query_info['memories']}\n"
            f"- Bối cảnh thất lạc: {query_info['context']}\n\n"
            f"CHI TIẾT YÊU CẦU: {query}"
        )
        
        # Perform vector search
        query_embedding = model.encode(enhanced_query, convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(query_embedding_np)
        D, I = index.search(query_embedding_np, top_k*2)  # Get more results initially
        
        # Post-processing to improve matching
        results = []
        scores = []
        
        for idx, score in zip(I[0], D[0]):
            row = df.iloc[idx]
            
            # Calculate field-specific match bonuses
            bonus = 0
            
            # Name matching bonus
            if query_info['name'] and query_info['name'].lower() in str(row['Tên người thất lạc']).lower():
                bonus += 0.15
                
            # Year matching bonus
            if query_info['year'] and query_info['year'] in str(row['Năm thất lạc']):
                bonus += 0.1
                
            # Location matching bonus
            if query_info['location'] and query_info['location'].lower() in str(row['Quê quán']).lower():
                bonus += 0.1
                
            # Apply bonus to score
            adjusted_score = min(score + bonus, 1.0)
            
            results.append((row, adjusted_score))
        
        # Sort by adjusted score and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
        return pd.DataFrame([r[0] for r in results]), [r[1] for r in results]
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return pd.DataFrame(), []

# 7. Ví dụ truy vấn
if __name__ == "__main__":
    print("\nVí dụ truy vấn:")
    query_text = "Tôi tên Sơn, mong muốn tìm lại gia đình của mình ngày xưa. Tôi sinh năm 1962, từng có 1 gia đình tại Vũng Tàu. Năm ấy (năm 2009), tôi có xích mích với vợ con mà bỏ nhà đi tới tận bây giờ."
    results, scores = search_missing_profiles(query_text)

    if len(results) == 0:
        print("Không tìm thấy kết quả phù hợp.")
    else:
        print(f"\nTìm thấy {len(results)} kết quả phù hợp:")
        for i, (_, row) in enumerate(results.iterrows()):
            print(f"\n--- Kết quả #{i+1} (Độ tương đồng: {scores[i]:.2f}) ---")
            print(f"Tên: {row['Tên người thất lạc']}")
            print(f"Năm thất lạc: {row['Năm thất lạc']}")
            print(f"Quê quán: {row['Quê quán']}")
            print(f"Đặc điểm: {row['Đặc điểm nhận dạng']}")
            print(f"Bối cảnh: {row['Bối cảnh thất lạc']}")
            print(f"Chi tiết: {row['Chi tiết'][:200]}...")
            
    # Interactive search mode
    while True:
        print("\n" + "="*50)
        print("Nhập mô tả người cần tìm (hoặc 'quit' để thoát):")
        query = input("> ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        results, scores = search_missing_profiles(query)
        
        if len(results) == 0:
            print("Không tìm thấy kết quả phù hợp.")
        else:
            print(f"\nTìm thấy {len(results)} kết quả phù hợp:")
            for i, (_, row) in enumerate(results.iterrows()):
                print(f"\n--- Kết quả #{i+1} (Độ tương đồng: {scores[i]:.2f}) ---")
                print(f"Tên: {row['Tên người thất lạc']}")
                print(f"Năm thất lạc: {row['Năm thất lạc']}")
                print(f"Quê quán: {row['Quê quán']}")
                print(f"Đặc điểm: {row['Đặc điểm nhận dạng']}")
                print(f"Bối cảnh: {row['Bối cảnh thất lạc']}")
                print(f"Chi tiết: {row['Chi tiết'][:200]}...")
