# import pandas as pd

# # Đọc file CSV
# file_path = r'F:\missing_people(NCHCCCL)\data\profiles_detailed_data_semantically_cleaned_v2.csv'
# df = pd.read_csv(file_path, quoting=1, quotechar='"')

# # Hàm thêm chi tiết quan trọng vào cuối nội dung chi tiết
# def add_important_details(row):
#     details = []
#     if pd.notna(row['Năm sinh']) and str(row['Năm sinh']).strip():
#         details.append(f"Năm sinh của người bị thất lạc là {row['Năm sinh']}")
    
#     if pd.notna(row['Năm thất lạc']) and str(row['Năm thất lạc']).strip():
#         details.append(f"thời gian thất lạc là vào {row['Năm thất lạc']}")
    
#     if pd.notna(row['Tên cha']) and str(row['Tên cha']).strip():
#         details.append(f"tên cha người bị thất lạc: {row['Tên cha']}")
    
#     if pd.notna(row['Tên mẹ']) and str(row['Tên mẹ']).strip():
#         details.append(f"tên mẹ người bị thất lạc: {row['Tên mẹ']}")
    
#     if pd.notna(row['Tên anh-chị-em']) and str(row['Tên anh-chị-em']).strip():
#         details.append(f"tên các anh chị em: {row['Tên anh-chị-em']}")
    
#     # Nếu có chi tiết, thêm vào cuối nội dung
#     if details:
#         merged_text = str(row['Chi tiet_merged']).strip() if pd.notna(row['Chi tiet_merged']) else ""
#         return f"{merged_text} Thông tin cơ bản: {'; '.join(details)}."
#     else:
#         return row['Chi tiet_merged']

# # Thêm dòng print kiểm tra trước khi cập nhật
# print(f"Tổng số hồ sơ: {len(df)}")
# print("Mẫu trước khi cập nhật:")
# for i in range(2):
#     print(f"\nHồ sơ {i+1}:")
#     print(f"- Chi tiết gốc: {df.iloc[i]['Chi tiet_merged'][:100]}...")
#     print(f"- Năm sinh: {df.iloc[i]['Năm sinh']}")
#     print(f"- Năm thất lạc: {df.iloc[i]['Năm thất lạc']}")
#     print(f"- Tên cha: {df.iloc[i]['Tên cha']}")
#     print(f"- Tên mẹ: {df.iloc[i]['Tên mẹ']}")
#     print(f"- Tên anh chị em: {df.iloc[i]['Tên anh-chị-em']}")

# # Áp dụng hàm vào dataframe
# df['Chi tiet_merged'] = df.apply(add_important_details, axis=1)

# # Kiểm tra kết quả sau cập nhật
# print("\nMẫu sau khi cập nhật:")
# for i in range(2):
#     print(f"\nHồ sơ {i+1} (sau cập nhật):")
#     print(f"- Chi tiết đã cập nhật: {df.iloc[i]['Chi tiet_merged']}")

# # Thống kê số hồ sơ có chi tiết được thêm vào
# added_details_count = sum(df['Chi tiet_merged'].str.contains("Thông tin cơ bản"))
# print(f"\nSố hồ sơ có chi tiết quan trọng được thêm vào: {added_details_count} ({added_details_count/len(df)*100:.2f}%)")

# # Lưu kết quả
# output_file = r'F:\missing_people(NCHCCCL)\data\profiles_detailed_data_final.csv'
# df.to_csv(output_file, index=False, quoting=1, quotechar='"')

# print(f"\nĐã hoàn thành và lưu kết quả vào file: {output_file}")

import pandas as pd

# Đường dẫn đến file gốc
input_file = r"F:\missing_people(NCHCCCL)\data\profiles_detailed_data_final.csv"

# Đường dẫn đến file mới sẽ tạo
output_file = r"F:\missing_people(NCHCCCL)\data\profiles_detailed_data_final_simplified.csv"

# Đọc file CSV gốc
try:
    df = pd.read_csv(input_file)
    
    # Chỉ giữ lại các cột cần thiết
    columns_to_keep = ["Link", "Tiêu đề", "Ảnh", "Chi tiet_merged"]
    
    # Kiểm tra xem các cột có tồn tại trong dataframe không
    for col in columns_to_keep:
        if col not in df.columns:
            print(f"Cột '{col}' không tồn tại trong file gốc. Các cột hiện có: {list(df.columns)}")
    
    # Lọc các cột cần giữ lại
    df_simplified = df[columns_to_keep]
    
    # Lưu vào file CSV mới
    df_simplified.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Đã tạo file CSV mới thành công: {output_file}")
    print(f"Số dòng trong file mới: {len(df_simplified)}")

except Exception as e:
    print(f"Đã xảy ra lỗi: {str(e)}")