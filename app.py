import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
import sys

# ==========================================
# 設定
# ==========================================
work_dir = "/Users/ryota/Documents/研究室/研究1/"
key_path = "key1.json"
metadata_filename = "raw.meta.tsv" 
metadata_file_path = os.path.join(work_dir, metadata_filename)
separator = '\t' 

# ==========================================
# 1. メタデータの読み込み (高速化版)
# ==========================================
print(f"📂 メタデータを読み込んでいます: {metadata_file_path}")

if not os.path.exists(metadata_file_path):
    print(f"❌ エラー: ファイルが見つかりません！")
    sys.exit()

try:
    # 読み込み
    df = pd.read_csv(metadata_file_path, sep=separator, dtype=str, on_bad_lines='skip', quotechar='"')
    
    # カラム名の確認と修正
    df.columns = [c.strip().strip('"') for c in df.columns]
    
    # ID列の特定
    id_cols = [c for c in df.columns if 'TRACK_ID' in c.upper() or 'ID' == c.upper()]
    if not id_cols:
        print(f"❌ エラー: ID列が見つかりません。列名: {df.columns.tolist()}")
        sys.exit()
    id_col = id_cols[0]
    
    # タイトル・アーティスト列の特定
    title_col = next((c for c in df.columns if 'TRACK_NAME' in c.upper()), None)
    if not title_col:
        title_col = next((c for c in df.columns if 'TITLE' in c.upper() or 'NAME' in c.upper()), None)

    artist_col = next((c for c in df.columns if 'ARTIST_NAME' in c.upper()), None)
    if not artist_col:
        artist_col = next((c for c in df.columns if 'ARTIST' in c.upper() and 'ID' not in c.upper()), None)
    if not artist_col:
        artist_col = next((c for c in df.columns if 'ARTIST' in c.upper()), None)
    
    print(f"ℹ️ 使用する列: ID={id_col}, Title={title_col}, Artist={artist_col}")

    # ★高速化ポイント: iterrows()をやめてリスト内包表記を使う
    print("⚡️ データを辞書に変換中...")
    
    # 必要な列をリスト化
    raw_ids = df[id_col].tolist()
    titles = df[title_col].fillna('Unknown Title').tolist()
    artists = df[artist_col].fillna('Unknown Artist').tolist()
    
    meta_dict = {}
    
    # zipでまとめてループ（これが爆速です）
    for r_id, title, artist in zip(raw_ids, titles, artists):
        r_id_str = str(r_id).strip().strip('"')
        
        # ID正規化ロジック
        try:
            clean_id = str(int(r_id_str.replace('track_', '')))
        except ValueError:
            clean_id = r_id_str
            
        meta_dict[clean_id] = {
            'title': str(title).strip('"'),
            'artist': str(artist).strip('"')
        }
        
    print(f"✅ {len(meta_dict)}曲分の情報を辞書化しました。")

except Exception as e:
    print(f"❌ エラー: メタデータ処理失敗: {e}")
    sys.exit()

# ==========================================
# 2. Firebase更新
# ==========================================
if not firebase_admin._apps:
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()
batch = db.batch()
batch_count = 0
updated_count = 0

print("🔥 Firebaseのデータを照合中...")
docs = db.collection('songs').stream()

for doc in docs:
    doc_id = doc.id
    track_id_key = doc_id.split('.')[0]
    
    match_found = False
    
    # IDマッチング
    if track_id_key in meta_dict:
        match_found = True
    elif track_id_key.isdigit() and str(int(track_id_key)) in meta_dict:
        track_id_key = str(int(track_id_key))
        match_found = True
        
    if match_found:
        info = meta_dict[track_id_key]
        doc_ref = db.collection('songs').document(doc_id)
        batch.set(doc_ref, {
            'title': info['title'],
            'artist': info['artist']
        }, merge=True)
        
        batch_count += 1
        updated_count += 1
    
    if batch_count >= 400:
        batch.commit()
        batch = db.batch()
        print(f"   -> {updated_count}件 更新済み...")
        batch_count = 0

if batch_count > 0:
    batch.commit()

print(f"\n🎉 完了！合計 {updated_count} 曲の更新に成功しました。")
