# read json file

import json
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from tqdm import tqdm
import sqlite3
import os

## train.json, dev.json, private_test.json, test.json에 대해 모두 적용

def preprocess_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    df = df[['id', 'pre_text', 'post_text', 'table']]

    df['company'] = df['id'].apply(lambda x: x.split('/')[0])
    df['year'] = df['id'].apply(lambda x: x.split('/')[1])
    df['page'] = df['id'].apply(lambda x: x.split('/')[2].split('.')[0].split('_')[1])

    df['pre_text'] = df['pre_text'].apply(lambda x: ' '.join(x))
    df['post_text'] = df['post_text'].apply(lambda x: ' '.join(x))
    df['table'] = df['table'].apply(lambda x: '\n'.join([','.join(i) for i in x]))
    df['dataset'] = file_path.split('/')[-1].split('.')[0]  # Add dataset name

    return df[['id', 'dataset', 'company', 'year', 'page', 'pre_text', 'post_text', 'table']]

train_df = preprocess_json('../dataset/train.json')
dev_df = preprocess_json('../dataset/dev.json')
private_test_df = preprocess_json('../dataset/private_test.json')
test_df = preprocess_json('../dataset/test.json')

# train_df, dev_df, private_test_df, test_df를 하나의 데이터프레임으로 합치기

df_all = pd.concat([train_df, dev_df, private_test_df, test_df], ignore_index=True)
print(f"Total rows: {len(df_all)}")

df_all = df_all.reset_index()
df_all = df_all[df_all['text'] != '.']
df_all = df_all.drop_duplicates(subset=['company', 'year', 'full_text'])




# 1. 임베딩 모델 설정
embedding = OpenAIEmbeddings()

# 2. Chroma 벡터 저장소 설정 
vectordb = Chroma(
    embedding_function=embedding,
    collection_name="finqa",
    persist_directory="test_db_table"  
)

batch_texts = []
batch_metadatas = []
batch_ids = []


BATCH_SIZE = 200
total = 0

for i, row in tqdm(df_all.iterrows(), total=len(df_all)):
    table = row['table']
    base_meta = {
        'index': int(row['index']),
        'company': row['company'],
        'fiscal': int(row['year']),
    }

    doc_id = f"{row['index']}_{row['id']}"  # 고유 ID 기본
    batch_texts.append(f"{row['id']} : {table}")
    batch_metadatas.append(base_meta)
    batch_ids.append(f"{doc_id}")
    total += 1

    # Batch 처리
    if len(batch_texts) >= BATCH_SIZE:
        vectordb.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        batch_texts, batch_metadatas, batch_ids = [], [], []

# # 마지막 남은 배치 처리
if batch_texts:
    vectordb.add_texts(
        texts=batch_texts,
        metadatas=batch_metadatas,
        ids=batch_ids
    )


# SQLite 준비 (full_docs 테이블)
DB_PATH = "./data/full_docs.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Create table if not exists
with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS full_docs (
            text_id INT PRIMARY KEY,
            full_text TEXT
        )
    """)
    conn.commit()

for i, row in tqdm(df_all.iterrows(), total=len(df_all)):
    table = row['table']
    
    text_id = int(row['index'])
    full_text = row['full_text']

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO full_docs (text_id, full_text)
            VALUES (?, ?)
        """, (text_id, full_text))
        conn.commit()
