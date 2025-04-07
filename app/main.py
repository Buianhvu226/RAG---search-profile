import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

from sentence_transformers import SentenceTransformer
from utils import load_data, generate_embeddings, preprocess_text, save_embeddings, load_embeddings
from search import VectorSearcher
import os
import re

DATA_PATH = r"f:\missing_people(NCHCCCL)\data\profiles_detailed_data_cleaned.csv"
EMBEDDINGS_PATH = r"f:\missing_people(NCHCCCL)\embeddings\embeddings.pkl"

def initialize_system():
    df = load_data(DATA_PATH)
    # Verify required columns exist
    required_columns = {'Chi tiết'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV is missing required columns: {missing}")
    
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            print("Loading pre-computed embeddings...")
            embeddings = load_embeddings(EMBEDDINGS_PATH)
            return VectorSearcher(embeddings, df), None
        except:
            print("Existing embeddings incompatible, recomputing...")
            os.remove(EMBEDDINGS_PATH)
    
    print("Computing new embeddings with Gemini model...")
    embeddings = []
    for _, row in df.iterrows():
        text = f"{row['Họ và tên']}, {row['Năm sinh']}, {row['Chi tiết']}"
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )["embedding"]
        embeddings.append(embedding)
    
    save_embeddings(embeddings, EMBEDDINGS_PATH)
    return VectorSearcher(embeddings, df), None

def search_interface(searcher, _):
    while True:
        query = input("Nhập mô tả người cần tìm (hoặc 'quit' để thoát): ")
        if query.lower() == 'quit':
            break
            
        processed_query = preprocess_text(query)
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=processed_query,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]
        
        results = searcher.search(query_embedding)
        
        if not results:
            print("Không tìm thấy kết quả phù hợp nào.")
            continue
            
        print(f"\nTìm thấy {len(results)} kết quả:")
        for i, (result, score) in enumerate(results, 1):
            print(f"\nKết quả #{i} (Độ tương đồng: {score:.2f}):")
            print(f"Họ tên: {result['Họ và tên']}")
            print(f"Năm sinh: {result['Năm sinh']}")
            print(f"Địa điểm: {result.get('Địa điểm', 'Không rõ')}")
            print(f"Chi tiết: {result['Chi tiết'][:300]}...")
            print(f"Năm thất lạc: {result['Năm thất lạc']}")

if __name__ == "__main__":
    searcher, model = initialize_system()
    search_interface(searcher, model)