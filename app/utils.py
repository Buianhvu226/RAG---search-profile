from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
import re  # Add this import for regex operations
import pickle

def load_data(filepath):
    try:
        # Try reading with pandas first
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', on_bad_lines='warn')
            print("Data loaded successfully with pandas. First few rows:")
            print(df.head())
            return df
        except Exception as e:
            print(f"Pandas read_csv failed: {e}")
            print("Falling back to manual CSV parsing...")
            
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
            
            # Get header
            header = lines[0].strip().split(',')
            expected_columns = len(header)
            
            # Process each line
            data = []
            for i, line in enumerate(lines[1:], 1):
                try:
                    row = line.strip().split(',')
                    if len(row) == expected_columns:
                        data.append(row)
                    else:
                        # Try to handle quoted fields with commas
                        try:
                            row = list(csv.reader([line], delimiter=',', quotechar='"'))[0]
                            if len(row) == expected_columns:
                                data.append(row)
                            else:
                                print(f"Skipping line {i+1} - unexpected number of fields: {len(row)}")
                        except:
                            print(f"Skipping line {i+1} - could not parse")
                except Exception as e:
                    print(f"Error processing line {i+1}: {e}")
            
            df = pd.DataFrame(data, columns=header)
            print("Data loaded successfully with manual parsing. First few rows:")
            print(df.head())
            return df
            
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_text(text):
    text = str(text).lower()
    
    # Standardize Vietnamese names and locations
    replacements = {
        r'\bvũng tàu\b': 'vungtau',
        r'\bđà nẵng\b': 'danang',
        r'\bhà nội\b': 'hanoi',
        r'\bsài gòn\b': 'saigon',
        r'\b(\d{2})\b': lambda m: f"19{m.group(1)}" if int(m.group(1)) < 100 else m.group(1)
    }
    
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    
    # Remove common search phrases
    stop_phrases = ["mong muốn", "tìm kiếm", "tìm được", "bị", "đã", "từ"]
    for phrase in stop_phrases:
        text = text.replace(phrase, "")
    
    return ' '.join(text.split())

def save_embeddings(embeddings, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_embeddings(df, model, text_column='Chi tiết'):
    # Combine multiple relevant fields for better context
    df['search_text'] = (
        df[text_column].fillna('') + ' ' +
        df['Họ và tên'].fillna('') + ' ' +
        df['Năm sinh'].astype(str) + ' ' +
        df['Tên cha'].fillna('') + ' ' +
        df['Tên mẹ'].fillna('')
    )
    
    texts = df['search_text'].apply(preprocess_text).tolist()
    return model.encode(texts, show_progress_bar=True, batch_size=32)