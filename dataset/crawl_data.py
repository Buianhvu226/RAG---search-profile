import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hàm crawl một trang danh sách
def crawl_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Không thể truy cập {url} - Mã lỗi: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Lỗi khi truy cập {url}: {e}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    profiles = soup.find('ul', class_='content-block type-profile')
    if not profiles:
        return None
    
    data = []
    for profile in profiles.find_all('li'):
        title_tag = profile.find('h1', class_='single-title')
        title = title_tag.text.strip() if title_tag else "N/A"
        link = title_tag.find('a')['href'] if title_tag and title_tag.find('a') else "N/A"
        
        details = profile.find_all('p')
        profile_info = {"Link": link, "Tiêu đề": title}
        for detail in details:
            text = detail.text.strip()
            if ":" in text:
                key, value = text.split(":", 1)
                profile_info[key.strip()] = value.strip()
        
        img_tag = profile.find('img', class_='wp-post-image')
        profile_info["Ảnh"] = img_tag['src'] if img_tag else "N/A"
        
        data.append(profile_info)
    
    return data

# Hàm crawl chi tiết một hồ sơ
def crawl_profile_detail(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Không thể truy cập {url} - Mã lỗi: {response.status_code}")
            return {"url": url, "Chi tiết": "Không thể truy cập"}
    except requests.RequestException as e:
        print(f"Lỗi khi truy cập {url}: {e}")
        return {"url": url, "Chi tiết": "Không thể truy cập"}
    
    soup = BeautifulSoup(response.text, 'html.parser')
    detail_info = {"url": url}
    
    content_div = soup.find('div', class_='profile-content')
    if content_div:
        paragraphs = content_div.find_all('p')
        detail_info["Chi tiết"] = "\n".join(p.text.strip() for p in paragraphs if p.text.strip())
        img_tag = content_div.find('img', class_='wp-image')
        detail_info["Ảnh bổ sung"] = img_tag['src'] if img_tag else "N/A"
    else:
        detail_info["Chi tiết"] = "Không có thông tin chi tiết"
        detail_info["Ảnh bổ sung"] = "N/A"
    
    contact_div = soup.find('div', class_='profile-contact')
    if contact_div:
        detail_info["Thông tin liên hệ"] = "\n".join(p.text.strip() for p in contact_div.find_all('p') if p.text.strip())
    else:
        detail_info["Thông tin liên hệ"] = "Không có thông tin liên hệ"
    
    return detail_info

# Hàm crawl toàn bộ với đa luồng, hỗ trợ bắt đầu từ trang bất kỳ
def crawl_all_profiles(base_url, start_page=1, max_pages=2080, max_workers=20):
    all_data = []
    
    # Tạo danh sách URL bắt đầu từ start_page
    page_urls = [base_url if page == 1 else f"{base_url}page/{page}/" for page in range(start_page, max_pages + 1)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(crawl_page, url): url for url in page_urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                page_data = future.result()
                if page_data:
                    print(f"Đã crawl xong trang: {url}")
                    # Crawl chi tiết từng hồ sơ trong trang
                    with ThreadPoolExecutor(max_workers=max_workers) as detail_executor:
                        detail_futures = {detail_executor.submit(crawl_profile_detail, profile["Link"]): profile for profile in page_data}
                        for detail_future in as_completed(detail_futures):
                            profile = detail_futures[detail_future]
                            try:
                                detail_data = detail_future.result()
                                print(f"Đã crawl chi tiết: {profile['Link']}")
                                profile.update(detail_data)
                            except Exception as e:
                                print(f"Lỗi khi crawl chi tiết {profile['Link']}: {e}")
                    all_data.extend(page_data)
                else:
                    print(f"Không tìm thấy dữ liệu ở {url}, có thể là trang cuối.")
                    break  # Dừng khi không còn dữ liệu
            except Exception as e:
                print(f"Lỗi khi crawl trang {url}: {e}")
    
    return all_data

# Thực thi từ trang 1048
base_url = "https://haylentieng.vn/profiles/"
data = crawl_all_profiles(base_url, start_page=1048, max_pages=2080, max_workers=20)

# Lưu vào file CSV (nối thêm vào file cũ nếu đã có)
if data:
    df = pd.DataFrame(data)
    # Nối vào file cũ thay vì ghi đè
    try:
        existing_df = pd.read_csv("profiles_detailed_data.csv")
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv("profiles_detailed_data.csv", index=False, encoding='utf-8-sig')
        print("Đã nối dữ liệu mới vào profiles_detailed_data.csv")
    except FileNotFoundError:
        df.to_csv("profiles_detailed_data.csv", index=False, encoding='utf-8-sig')
        print("Đã tạo và lưu dữ liệu vào profiles_detailed_data.csv")
else:
    print("Không có dữ liệu để lưu.")