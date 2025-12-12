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
# 1. メタデータの読み込み
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
    
    # タイトル・アーティスト列の特定 (優先順位をつける)
    # TITLE, NAME, TRACK_NAME などの候補
    title_col = next((c for c in df.columns if 'TRACK_NAME' in c.upper()), None)
    if not title_col:
        title_col = next((c for c in df.columns if 'TITLE' in c.upper() or 'NAME' in c.upper()), None)

    # ARTIST_NAME, ARTIST などの候補 (IDよりもNAMEを優先)
    artist_col = next((c for c in df.columns if 'ARTIST_NAME' in c.upper()), None)
    if not artist_col:
        artist_col = next((c for c in df.columns if 'ARTIST' in c.upper() and 'ID' not in c.upper()), None)
    if not artist_col:
        # どうしてもなければIDなどが含まれるカラムを使う
        artist_col = next((c for c in df.columns if 'ARTIST' in c.upper()), None)
    
    print(f"ℹ️ 使用する列: ID={id_col}, Title={title_col}, Artist={artist_col}")

    # 辞書化 (IDの前後の空白を除去してキーにする)
    meta_dict = {}
    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip().strip('"')
        
        # ★修正ポイント: IDの正規化ロジック
        # "track_0000214" のような形式から "track_" を取り、数値化してゼロ埋めを消す
        try:
            # "track_" があれば消す -> intにしてゼロ消す -> strに戻す
            clean_id = str(int(raw_id.replace('track_', '')))
        except ValueError:
            # 数値にできない場合はそのまま使う
            clean_id = raw_id

        meta_dict[clean_id] = {
            'title': str(row.get(title_col, 'Unknown Title')).strip('"'),
            'artist': str(row.get(artist_col, 'Unknown Artist')).strip('"')
        }
        
    print(f"✅ {len(meta_dict)}曲分の情報を読み込みました。")
    # サンプル表示 (デバッグ用)
    print(f"   (辞書キーのサンプル: {list(meta_dict.keys())[:5]})")

except Exception as e:
    print(f"❌ エラー: メタデータ読み込み失敗: {e}")
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

debug_print_count = 0

for doc in docs:
    doc_id = doc.id
    
    # IDの抽出: "." より前の部分を取得
    track_id_key = doc_id.split('.')[0]
    
    match_found = False
    
    # 1. そのまま検索
    if track_id_key in meta_dict:
        match_found = True
    # 2. 数値化して検索 (念のため)
    elif track_id_key.isdigit() and str(int(track_id_key)) in meta_dict:
        track_id_key = str(int(track_id_key))
        match_found = True
        
    if not match_found and debug_print_count < 5:
        print(f"⚠️ 不一致: Firebase ID '{track_id_key}' がメタデータ辞書にありません")
        debug_print_count += 1

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
if updated_count == 0:
    print("⚠️ 注意: 1曲もマッチしませんでした。")
    print("ヒント: メタデータのID形式 (track_00...) とFirebaseのID (100...) が合致するように変換ロジックを追加しました。")
    print("それでも合わない場合は、手元のメタデータファイルの中身を確認してください。")
