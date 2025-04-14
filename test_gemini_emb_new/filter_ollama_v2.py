import requests
import pandas as pd
from tqdm import tqdm
import time

# 👉 NHỚ THAY BẰNG API KEY THẬT CỦA BẠN
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

PROMPT_TEMPLATE = """
Bạn là một trợ lý biên tập dữ liệu tìm kiếm người thân, rất giỏi trong việc xác định và loại bỏ thông tin không cần thiết ở đầu các lời nhắn.
Nhiệm vụ của bạn là đọc đoạn văn bản gốc và trả về **chỉ nội dung chính** của lời nhắn tìm kiếm, loại bỏ phần giới thiệu về người đăng ký hoặc các thông tin thừa thãi ở đầu.
Tuyệt đối **không được loại bỏ** bất kỳ thông tin nào liên quan đến:
- Người cần tìm (tên, tuổi, năm sinh, mô tả)
- Quê quán, địa điểm đã đi qua, địa điểm từng ở
- Họ hàng (cha mẹ, anh chị em, cô chú)
- Thời gian, công việc, nơi làm việc

Hãy giữ lại **toàn bộ phần mô tả về người cần tìm**, dù thông tin xuất hiện sớm.

Ví dụ 1:
Văn bản gốc: "Bà Trần Thị Phượng Anh Nguyễn Văn Hùng đăng ký tìm mẹ Trần Thị Phượng sinh 1956..."
Văn bản sạch: "Nguyễn Văn Hùng đăng ký tìm mẹ Trần Thị Phượng sinh 1956..."

Ví dụ 2:
Văn bản gốc: "Ông Nguyễn Văn Cháu Chị Nguyễn Thị Hương đăng ký tìm ba Nguyễn Văn Cháu sinh 1956. Năm 2000, ông Cháu ở bến xe ..."
Văn bản sạch: "Chị Nguyễn Thị Hương đăng ký tìm ba Nguyễn Văn Cháu sinh 1956. Năm 2000, ông Cháu ở bến xe ..."

Ví dụ 3:
Văn bản gốc: "Tìm em Lê Văn Tâm. Anh Lê Văn Bình đăng ký tìm em Lê Văn Tâm, sinh năm 1980. Quê Thái Bình. Năm 1998, Tâm đi lạc tại ga Sài Gòn..."
Văn bản sạch: "Anh Lê Văn Bình đăng ký tìm em Lê Văn Tâm, sinh năm 1980. Quê Thái Bình. Năm 1998, Tâm đi lạc tại ga Sài Gòn..."

Ví dụ 4:
Văn bản gốc: "Hồ sơ TH123. Anh Trần Văn An tìm mẹ là bà Nguyễn Thị Lan, sinh 1960. Bà Lan thất lạc năm 1990 tại Chợ Lớn..."
Văn bản sạch: "Anh Trần Văn An tìm mẹ là bà Nguyễn Thị Lan, sinh 1960. Bà Lan thất lạc năm 1990 tại Chợ Lớn..."

...

Bây giờ, hãy làm sạch đoạn văn bản gốc sau đây. Chỉ trả về phần văn bản đã làm sạch, không thêm bất kỳ lời giải thích nào khác.

Văn bản gốc: "{text_to_clean}"
Văn bản sạch:"""

def clean_text_with_groq(text_to_clean):
    if not isinstance(text_to_clean, str) or not text_to_clean.strip():
        return ""

    prompt = PROMPT_TEMPLATE.format(text_to_clean=text_to_clean.strip())

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Bạn là một trợ lý xử lý văn bản."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(GROQ_API_URL, headers=HEADERS, json=body, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"❌ Lỗi HTTP {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        print(f"❗ Lỗi khi gọi Groq API: {e}")
        return ""

def process_csv(input_path, output_path):
    print(f"📂 Đang đọc file: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8')

    if "Chi tiết" not in df.columns:
        print("❌ Không tìm thấy cột 'Chi tiết' trong file.")
        return

    print(f"🔍 Tổng cộng {len(df)} dòng sẽ được xử lý...")

    cleaned_list = []
    for text in tqdm(df["Chi tiết"], desc="🧹 Đang làm sạch"):
        try:
            cleaned = clean_text_with_groq(text)
        except Exception as e:
            print("❗ Lỗi khi xử lý dòng:", e)
            cleaned = ""
        cleaned_list.append(cleaned)
        time.sleep(0.1)

    df["Chi tiet_sach"] = cleaned_list

    print(f"\n💾 Ghi kết quả vào: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("✅ Hoàn tất!")


if __name__ == "__main__":
    input_csv = "profiles_detailed_data_cleaned_full.csv"
    output_csv = "du_lieu_sach.csv"
    process_csv(input_csv, output_csv)
