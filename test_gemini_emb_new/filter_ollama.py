import subprocess

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

def clean_text_with_mistral(text_to_clean):
    full_prompt = PROMPT_TEMPLATE.format(text_to_clean=text_to_clean)

    process = subprocess.Popen(
        ["C:\\Users\\UYEN MY\\AppData\\Local\\Programs\\Ollama\\ollama.exe", "run", "mistral:7b-instruct"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=full_prompt.encode('utf-8'))

    # In thông báo lỗi nếu có
    stderr_decoded = stderr.decode('utf-8', errors='ignore').strip()
    if stderr_decoded:
        print("❌ Lỗi khi gọi mô hình:\n", stderr_decoded)

    result = stdout.decode('utf-8', errors='ignore').strip()

    # In kết quả thô để kiểm tra nếu cần
    # print("📜 Raw response:\n", result)

    # Cố gắng tách phần sau "Văn bản sạch:"
    if "Văn bản sạch:" in result:
        cleaned = result.split("Văn bản sạch:")[-1].strip()
        return cleaned

    return result or None


# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    input_text = "Chị Võ Thị Thắm (áo sơ mi sọc) Anh Võ Đại Sừ đăng ký tìm em Võ Thị Thắm sinh năm 1977. Khoảng 1998, chị Thắm vào Tp.HCM làm việc, có viết thư về nhà một thời gian sau đó bặt tin. Chị Võ Thị Thắm sinh năm 1977, có cha tên Võ Đại Bửu, mẹ tên Lê Thị Gái. Có các anh chị em gồm: Bôn, Sừ, Lĩnh, Tể, Be và Kha. Gia đình sinh sống tại Thừa Thiên Huế. Khoảng năm 1998, chị Thắm rời gia đình vào Tp.HCM làm việc. Có thời gian, chị Thắm cùng anh trai tên Lĩnh lên Đà Lạt, Lâm Đồng sinh sống. Thời điểm ở Đà Lạt, chị Thắm bán sữa đậu nành tại đường Phan Bội Châu. Sau đó, có người bà con lên Đà Lạt rủ chị Thắm về lại Tp.HCM làm. Chị Thắm về lại Tp.HCM nhưng không nói rõ ở đâu. Gia đình mất liên lạc với chị từ đó đến nay."
    cleaned = clean_text_with_mistral(input_text)

    print("\n📌 Văn bản gốc:")
    print(input_text)
    print("\n✅ Văn bản đã làm sạch:")
    print(cleaned if cleaned else "⚠️ Không nhận được kết quả.")
