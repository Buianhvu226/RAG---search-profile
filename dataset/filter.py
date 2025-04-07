# import pandas as pd
# df = pd.read_csv("profiles_detailed_data.csv", encoding='utf-8-sig')
# df = df.drop(columns=["url", "Thông tin liên hệ", "Ảnh bổ sung"], errors='ignore')
# # Loại bỏ các hàng có giá trị giống nhau trong cột "Link"
# df = df.drop_duplicates(subset=["Link"])
# # Xóa các đoạn văn bản có giá trị như: "Như chưa hề có cuộc chia ly…”:", "– Hòm thư 005, Bưu điện Trung tâm Sài Gòn, TPHCM", "– Website: haylentieng.vn", "– Youtube: “NHƯ CHƯA HỀ CÓ CUỘC CHIA LY – OFFICIAL”: www.youtube.com/NhuchuahecocuocchialyOfficial", "– Fanpage: https://www.facebook.com/nchcccl", "[Hoạt động thiện nguyện Tìm kiếm, Kết nối và Đoàn tụ thân nhân NHƯ CHƯA HỀ CÓ CUỘC CHIA LY… do Công ty TNHH Xã hội Nối Thân Thương chủ trì]."" ở trong cột "Chi tiết"
# string1 = "“Như chưa hề có cuộc chia ly…”:"
# string2 = "– Hòm thư 005, Bưu điện Trung tâm Sài Gòn, TPHCM"
# string3 = "– Website: haylentieng.vn"
# string4 = "– Kênh Youtube Như chưa hề có cuộc chia ly – Offical: https://bit.ly/2GXUgnu"
# string5 = "– Fanpage: https://www.facebook.com/nchcccl"
# string6 = "[Hoạt động thiện nguyện Tìm kiếm, Kết nối và Đoàn tụ thân nhân NHƯ CHƯA HỀ CÓ CUỘC CHIA LY… do Công ty TNHH Xã hội Nối Thân Thương chủ trì]"
# df = df[~df["Chi tiết"].str.contains(string1, na=False)]
# df = df[~df["Chi tiết"].str.contains(string2, na=False)]
# df = df[~df["Chi tiết"].str.contains(string3, na=False)]
# df = df[~df["Chi tiết"].str.contains(string4, na=False)]
# df = df[~df["Chi tiết"].str.contains(string5, na=False)]
# df = df[~df["Chi tiết"].str.contains(string6, na=False)]

# # Lưu lại dataframe vào file csv
# df.to_csv("profiles_detailed_data.csv", index=False, encoding='utf-8-sig')
# print("Đã lưu dữ liệu vào profiles_detailed_data.csv")


import pandas as pd

# Đọc file CSV với xử lý lỗi định dạng
try:
    df = pd.read_csv("profiles_detailed_data.csv", encoding='utf-8-sig', engine='python')
except:
    # Thử phương pháp thay thế nếu cách trên không hoạt động
    df = pd.read_csv("profiles_detailed_data.csv", 
                     encoding='utf-8-sig', 
                     quoting=pd.io.common.csv.QUOTE_NONE,
                     on_bad_lines='skip')

# Loại bỏ các cột không cần thiết
df = df.drop(columns=["url", "Thông tin liên hệ", "Ảnh bổ sung"], errors='ignore')

# Loại bỏ các hàng có giá trị giống nhau trong cột "Link"
df = df.drop_duplicates(subset=["Link"])

# Danh sách các chuỗi cần xóa trong cột "Chi tiết"
strings_to_remove = [
    "“Như chưa hề có cuộc chia ly…”:",
    "– Hòm thư 005, Bưu điện Trung tâm Sài Gòn, TPHCM",
    "– Website: haylentieng.vn",
    "– Kênh Youtube Như chưa hề có cuộc chia ly – Offical: https://bit.ly/2GXUgnu",
    "– Fanpage: https://www.facebook.com/nchcccl",
    "[Hoạt động thiện nguyện Tìm kiếm, Kết nối và Đoàn tụ thân nhân NHƯ CHƯA HỀ CÓ CUỘC CHIA LY… do Công ty TNHH Xã hội Nối Thân Thương chủ trì]"
]

# Xóa các chuỗi cụ thể từ cột "Chi tiết"
def remove_strings(text):
    if pd.isna(text):  # Kiểm tra nếu giá trị là NaN
        return text
    
    result = text
    for string in strings_to_remove:
        result = result.replace(string, "")
    
    # Xóa các khoảng trắng thừa
    result = ' '.join(result.split())
    return result

# Áp dụng hàm xóa chuỗi cho cột "Chi tiết"
df["Chi tiết"] = df["Chi tiết"].apply(remove_strings)

# Lưu lại dataframe vào file csv
df.to_csv("profiles_detailed_data_cleaned.csv", index=False, encoding='utf-8-sig')
print("Đã lưu dữ liệu vào profiles_detailed_data_cleaned.csv")