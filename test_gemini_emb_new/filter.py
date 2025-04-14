import pandas as pd
import google.generativeai as genai
import time
import os
import random
from tqdm import tqdm
import math
# import concurrent.futures # Bỏ comment nếu dùng xử lý song song nâng cao
from itertools import cycle # Để xoay vòng key
import csv # Thêm thư viện csv để dùng hằng số quoting
import traceback # Để in chi tiết lỗi

# --- Configuration ---
# !!! CẢNH BÁO BẢO MẬT !!! Rất không nên hardcode API keys.
API_KEYS = [
    "AIzaSyDoT41uDC4u212LEnJPS0BPmKKjI4QyWZA",
    "AIzaSyCN_flhR6pXNOvQWjZSMAwe_t1DnI_O8IM",
    "AIzaSyBVEQzc89kQ1072ji4xR9wMPtBlzvqCIlY",
    "AIzaSyDats92Eac1yPpk4Z9soGf4nCCiBTh1P64",
    "AIzaSyBCAqTBZSg7wXK_Jg-JnXW0rZkRJ-VRU64",
    "AIzaSyDw2a1VhB3MXps3ldFUMyYvi65OTIMqFfM",
    "AIzaSyAcI-lF8zk9FSEcZR7aShsYw4WJopu-ZlE",
    "AIzaSyAsf3I9ygoCPpYtFjmMGACn783PARRBr1w",
]
API_KEYS = [key.strip() for key in API_KEYS if key and len(key.strip()) > 10]

if not API_KEYS:
    print("Lỗi: Không tìm thấy API Key nào hợp lệ. Vui lòng kiểm tra lại.")
    exit()

print(f"Sử dụng {len(API_KEYS)} API keys.")
key_cycler = cycle(API_KEYS)

INPUT_CSV = 'F:\\missing_people(NCHCCCL)\\data\\profiles_detailed_data_cleaned_full.csv'
OUTPUT_CSV = 'F:\\missing_people(NCHCCCL)\\data\\profiles_detailed_data_semantically_cleaned_full.csv'
CHECKPOINT_CSV = 'F:\\missing_people(NCHCCCL)\\data\\semantic_cleaner_checkpoint_full.csv'

COLUMN_TO_CLEAN = 'Chi tiết'
NEW_COLUMN_NAME = 'Chi tiet_sach'

# !!! SỬA LẠI TÊN MODEL !!!
MODEL_NAME = 'gemini-2.0-flash' # Hoặc 'gemini-pro'
MAX_RETRIES = 5
INITIAL_DELAY = 1
REQUEST_DELAY = 0.5
CHECKPOINT_INTERVAL = 50

# --- Prompt cho Gemini ---
PROMPT_TEMPLATE = """
Bạn là một trợ lý biên tập dữ liệu tìm kiếm người thân, rất giỏi trong việc xác định và loại bỏ thông tin không cần thiết ở đầu các lời nhắn.
Nhiệm vụ của bạn là đọc đoạn văn bản gốc và trả về **chỉ nội dung chính** của lời nhắn tìm kiếm, loại bỏ phần giới thiệu về người đăng ký hoặc các thông tin thừa thãi ở đầu.

Ví dụ 1:
Văn bản gốc: "Bà Trần Thị Phượng Anh Nguyễn Văn Hùng đăng ký tìm mẹ Trần Thị Phượng sinh 1956. Năm 1995, bà Phượng bỏ nhà đi rồi b ..."
Văn bản sạch: "Nguyễn Văn Hùng đăng ký tìm mẹ Trần Thị Phượng sinh 1956. Năm 1995, bà Phượng bỏ nhà đi rồi b ..."

Ví dụ 2:
Văn bản gốc: "Ông Nguyễn Văn Cháu Chị Nguyễn Thị Hương đăng ký tìm ba Nguyễn Văn Cháu sinh 1956. Năm 2000, ông Cháu ở bến xe ..."
Văn bản sạch: "Chị Nguyễn Thị Hương đăng ký tìm ba Nguyễn Văn Cháu sinh 1956. Năm 2000, ông Cháu ở bến xe ..."

Ví dụ 3:
Văn bản gốc: "Tìm em Lê Văn Tâm. Anh Lê Văn Bình đăng ký tìm em Lê Văn Tâm, sinh năm 1980. Quê Thái Bình. Năm 1998, Tâm đi lạc tại ga Sài Gòn..."
Văn bản sạch: "Anh Lê Văn Bình đăng ký tìm em Lê Văn Tâm, sinh năm 1980. Quê Thái Bình. Năm 1998, Tâm đi lạc tại ga Sài Gòn..."

Ví dụ 4:
Văn bản gốc: "Hồ sơ TH123. Anh Trần Văn An tìm mẹ là bà Nguyễn Thị Lan, sinh 1960. Bà Lan thất lạc năm 1990 tại Chợ Lớn..."
Văn bản sạch: "Anh Trần Văn An tìm mẹ là bà Nguyễn Thị Lan, sinh 1960. Bà Lan thất lạc năm 1990 tại Chợ Lớn..."

Bây giờ, hãy làm sạch đoạn văn bản gốc sau đây. Chỉ trả về phần văn bản đã làm sạch, không thêm bất kỳ lời giải thích nào khác.

Văn bản gốc: "{text_to_clean}"
Văn bản sạch:"""

# --- Hàm tương tác với Gemini API ---
def clean_text_with_gemini(text_to_clean, api_key_to_use, current_retries=0):
    """Gửi yêu cầu đến Gemini và xử lý kết quả/lỗi."""
    if not isinstance(text_to_clean, str) or not text_to_clean.strip():
        return text_to_clean

    try:
        genai.configure(api_key=api_key_to_use)
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = PROMPT_TEMPLATE.format(text_to_clean=text_to_clean)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # Xử lý response cẩn thận hơn
        if response.prompt_feedback.block_reason:
             print(f"Cảnh báo: Yêu cầu bị chặn do {response.prompt_feedback.block_reason}. Trả về văn bản gốc.")
             return text_to_clean

        # Truy cập text qua candidates nếu có thể, hoặc parts
        try:
             cleaned_text = response.text.strip()
        except ValueError:
             # Handle cases where response.text is not available (e.g., blocked)
             print("Cảnh báo: Không thể truy cập response.text. Có thể đã bị chặn hoặc lỗi khác.")
             # You might want to inspect response.parts or prompt_feedback here
             return text_to_clean
        except AttributeError:
             print("Cảnh báo: Đối tượng response không có thuộc tính 'text'.")
             # Inspect response structure if this happens
             return text_to_clean


        if len(cleaned_text) < 10 and len(text_to_clean) > 50:
            print(f"Cảnh báo: Kết quả trả về có vẻ quá ngắn ('{cleaned_text}'). Trả về văn bản gốc.")
            return text_to_clean
        return cleaned_text

    except Exception as e:
        print(f"Lỗi API khi xử lý text '{str(text_to_clean)[:50]}...' với key ...{api_key_to_use[-4:]}: {e}")
        if current_retries < MAX_RETRIES:
            wait_time = INITIAL_DELAY * (2 ** current_retries) + random.uniform(0, 1)
            print(f"Thử lại sau {wait_time:.2f} giây... (lần {current_retries + 1}/{MAX_RETRIES})")
            time.sleep(wait_time)
            next_api_key = next(key_cycler)
            return clean_text_with_gemini(text_to_clean, next_api_key, current_retries + 1)
        else:
            print(f"Không thể xử lý sau {MAX_RETRIES} lần thử. Trả về văn bản gốc.")
            return text_to_clean

# --- Hàm đọc CSV với nhiều cách thử ---
def load_data_robustly(filepath, is_checkpoint=False):
    """Tries multiple ways to load a CSV file using Pandas."""
    print(f"Đang thử đọc file: {filepath}...")
    df = None
    # Danh sách các tham số để thử đọc CSV
    read_attempts = [
        # Thử 1: Mặc định UTF-8
        {'encoding': 'utf-8'},
        # Thử 2: Dùng engine='python' (chậm hơn nhưng linh hoạt hơn)
        {'encoding': 'utf-8', 'engine': 'python'},
        # Thử 3: Thử chế độ quoting khác
        {'encoding': 'utf-8', 'quoting': csv.QUOTE_MINIMAL},
        # Thử 4: Thử chế độ quoting khác
        {'encoding': 'utf-8', 'quoting': csv.QUOTE_ALL},
        # Thử 5: Bỏ qua dòng lỗi (Dùng cẩn thận!)
        # {'encoding': 'utf-8', 'on_bad_lines': 'skip'},
        # Thử 6: Cảnh báo dòng lỗi
        {'encoding': 'utf-8', 'on_bad_lines': 'warn'},
        # Thử 7: Nếu nghi ngờ dấu phân cách là chấm phẩy
        # {'encoding': 'utf-8', 'sep': ';'},
        # Thử 8: Nếu nghi ngờ dấu phân cách là tab
        # {'encoding': 'utf-8', 'sep': '\t'},
    ]

    for i, params in enumerate(read_attempts):
        print(f"  Thử lần {i+1} với tham số: {params}")
        try:
            df = pd.read_csv(filepath, **params)
            print(f"  >> Đọc thành công với tham số: {params}")

            # Kiểm tra cơ bản nếu là checkpoint
            if is_checkpoint:
                if df.empty:
                    print("  Cảnh báo: Checkpoint file rỗng.")
                    # Có thể coi là không hợp lệ và trả về None nếu muốn
                    # return None
                if COLUMN_TO_CLEAN not in df.columns:
                     print(f"  Lỗi: Cột gốc '{COLUMN_TO_CLEAN}' không có trong checkpoint. File checkpoint không hợp lệ.")
                     return None # Không thể tiếp tục nếu thiếu cột gốc
                if NEW_COLUMN_NAME not in df.columns:
                     print(f"  Cảnh báo: Cột '{NEW_COLUMN_NAME}' không có trong checkpoint. Sẽ tạo cột mới.")
                     df[NEW_COLUMN_NAME] = pd.NA # Tạo cột nếu thiếu

            return df # Trả về DataFrame nếu đọc thành công
        except FileNotFoundError:
             print(f"  Lỗi: File không tồn tại tại đường dẫn: {filepath}")
             return None # Dừng nếu file không tồn tại
        except Exception as e:
            # In lỗi cụ thể hơn
            error_msg = str(e)
            if "Error tokenizing data" in error_msg or "No columns to parse" in error_msg:
                print(f"  Lỗi đọc CSV (Tokenizing/Parsing): {error_msg}")
            else:
                print(f"  Lỗi không xác định khi đọc: {error_msg}")
            # Không cần in traceback đầy đủ trừ khi muốn debug sâu
            # traceback.print_exc()

    print(f"!!! Không thể đọc file {filepath} bằng pandas sau nhiều lần thử.")
    return None

# --- Hàm chính ---
def main():
    print("--- Bắt đầu làm sạch ngữ nghĩa bằng Gemini ---")
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Checkpoint: {CHECKPOINT_CSV}")

    # 1. Tải dữ liệu với hàm robust
    df = load_data_robustly(CHECKPOINT_CSV, is_checkpoint=True)

    if df is None:
        print("\nKhông thể tải checkpoint hoặc checkpoint không hợp lệ.")
        print("Đang thử tải file gốc...")
        df = load_data_robustly(INPUT_CSV, is_checkpoint=False)
        if df is not None:
             if COLUMN_TO_CLEAN not in df.columns:
                 print(f"Lỗi: Cột '{COLUMN_TO_CLEAN}' không tồn tại trong file gốc {INPUT_CSV}")
                 df = None # Đánh dấu là không hợp lệ
             elif NEW_COLUMN_NAME not in df.columns:
                 print(f"Tạo cột mới '{NEW_COLUMN_NAME}' trong DataFrame.")
                 df[NEW_COLUMN_NAME] = pd.NA
             else:
                 print(f"Cột '{NEW_COLUMN_NAME}' đã tồn tại. Sẽ xử lý các giá trị NA.")

    if df is None:
        print("\n!!! Lỗi nghiêm trọng: Không thể tải dữ liệu từ checkpoint hoặc file gốc.")
        print("Vui lòng kiểm tra lại file CSV đầu vào, đặc biệt các dòng gây lỗi (như dòng 5 được báo cáo).")
        print("Gợi ý: Mở file bằng trình soạn thảo văn bản để tìm dấu phẩy/ngoặc kép bất thường, hoặc thử các tham số đọc khác trong hàm load_data_robustly.")
        return

    # Reset index sau khi đã tải thành công để đảm bảo tính nhất quán
    df.reset_index(drop=True, inplace=True)
    print(f"Đã tải thành công và reset index. Tổng số dòng: {len(df)}")

    # 2. Xác định các dòng cần xử lý
    # Đảm bảo cột mới tồn tại trước khi kiểm tra NA
    if NEW_COLUMN_NAME not in df.columns:
        df[NEW_COLUMN_NAME] = pd.NA

    # Chỉ xử lý từ hồ sơ 800 trở đi
    START_INDEX = 800
    print(f"\nBỏ qua {START_INDEX} hồ sơ đầu tiên đã được xử lý trước đó.")
    
    # Lấy các dòng từ index 800 trở đi và có giá trị NA trong cột mới
    rows_to_process = df.loc[df.index >= START_INDEX].loc[df[NEW_COLUMN_NAME].isna()].index
    
    if not rows_to_process.any():
        print(f"\nKhông có dòng nào cần xử lý mới từ index {START_INDEX} trở đi (dựa trên cột '{NEW_COLUMN_NAME}' không có giá trị NA).")
        # Có thể bạn muốn ghi đè file output cuối cùng tại đây nếu logic yêu cầu
        print(f"Lưu lại file vào {OUTPUT_CSV} để đảm bảo...")
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        print("--- Kết thúc ---")
        return

    print(f"\nTổng số dòng cần xử lý (từ index {START_INDEX}): {len(rows_to_process)}")
    print("Bắt đầu xử lý...\n")

    # 3. Xử lý từng dòng
    start_time = time.time()
    processed_count = 0
    error_count = 0
    try:
        # Sử dụng list(rows_to_process) để tránh lỗi nếu index thay đổi khi đang lặp
        for index in tqdm(list(rows_to_process), desc=f"Đang xử lý ngữ nghĩa từ index {START_INDEX}"):
            # Kiểm tra lại index tồn tại trong df phòng trường hợp df thay đổi bất thường
            if index not in df.index:
                print(f"Cảnh báo: Bỏ qua index {index} không còn tồn tại.")
                continue

            text_original = df.loc[index, COLUMN_TO_CLEAN]
            current_api_key = next(key_cycler)

            # Gọi API để làm sạch
            cleaned_result = clean_text_with_gemini(text_original, current_api_key)

            # Ghi kết quả vào DataFrame một cách an toàn
            df.loc[index, NEW_COLUMN_NAME] = cleaned_result
            # Đếm cả trường hợp trả về gốc do lỗi API là đã xử lý
            processed_count += 1

            # Thêm độ trễ nhỏ
            time.sleep(REQUEST_DELAY)

            # Lưu checkpoint định kỳ
            # Sử dụng processed_count thay vì index để đảm bảo lưu sau khoảng cố định
            if processed_count > 0 and processed_count % CHECKPOINT_INTERVAL == 0:
                print(f"\nĐã xử lý {processed_count} dòng. Đang lưu checkpoint...")
                try:
                    df.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
                    print("Lưu checkpoint thành công.")
                except Exception as e:
                    print(f"Lỗi khi lưu checkpoint: {e}")

    except KeyboardInterrupt:
        print("\nĐã nhận tín hiệu dừng (Ctrl+C).")
    except Exception as e:
        print(f"\nLỗi không mong muốn trong quá trình xử lý: {e}")
        traceback.print_exc()
    finally:
        # Lưu checkpoint cuối cùng khi thoát
        print("\nĐang lưu checkpoint cuối cùng...")
        try:
            df.to_csv(CHECKPOINT_CSV, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
            print(f"Lưu checkpoint cuối cùng thành công vào {CHECKPOINT_CSV}")
        except Exception as e:
            print(f"Lỗi khi lưu checkpoint cuối cùng: {e}")

    # 4. Thống kê và Lưu kết quả cuối cùng
    elapsed = time.time() - start_time
    print("\n--- Thống kê ---")
    print(f"Tổng thời gian xử lý: {elapsed:.2f} giây")
    if processed_count > 0:
       print(f"Đã xử lý tổng cộng: {processed_count} dòng")
       print(f"Tốc độ trung bình: {processed_count / elapsed:.2f} dòng/giây (bao gồm cả thời gian chờ)")
    else:
       print("Không xử lý được dòng nào.")
    # Bạn có thể thêm đếm lỗi nếu cần

    print(f"\nĐang lưu kết quả cuối cùng vào {OUTPUT_CSV}...")
    try:
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        print(f"Đã lưu kết quả cuối cùng thành công!")
    except Exception as e:
        print(f"Lỗi khi lưu file kết quả cuối cùng: {e}")

    print("--- Kết thúc ---")

if __name__ == "__main__":
    main()