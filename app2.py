import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
import base64
import random

# ==========================================
# 1. 設定
# ==========================================
st.set_page_config(page_title="Random Music Player", layout="centered")

# Firebase初期化 (app.pyと同じ仕組み)
if not firebase_admin._apps:
    try:
        if "FIREBASE_BASE64" in st.secrets:
            key_str = base64.b64decode(st.secrets["FIREBASE_BASE64"]).decode('utf-8')
            key_dict = json.loads(key_str)
            cred = credentials.Certificate(key_dict)
            firebase_admin.initialize_app(cred)
        else:
            st.error("Secretsエラー: 'FIREBASE_BASE64' が見つかりません。")
            st.stop()
    except Exception as e:
        st.error(f"Firebase接続エラー: {e}")
        st.stop()

db = firestore.client()

# ==========================================
# 2. データロード (URLとタイトルだけ取得)
# ==========================================
@st.cache_data
def load_songs_from_firebase():
    docs = db.collection('songs').stream()
    song_list = []
    
    for doc in docs:
        data = doc.to_dict()
        url = data.get('audio_url')
        
        # URLがあるデータだけリストに入れる
        if url:
            filename = data.get('filename', 'Unknown')
            title = data.get('title', filename)
            artist = data.get('artist', 'Unknown Artist')
            
            # 表示名を作成
            if title != filename:
                display_name = f"{title} / {artist}"
            else:
                display_name = filename

            song_list.append({
                'name': display_name,
                'url': url
            })
            
    return song_list

with st.spinner('Loading song list...'):
    songs = load_songs_from_firebase()

if not songs:
    st.error("楽曲データが見つかりません。")
    st.stop()

# ==========================================
# 3. アプリ画面 UI (ランダム再生)
# ==========================================
st.title("🎲 Random Music Player")
st.caption(f"Randomly selecting from {len(songs)} songs")

# セッション状態で「現在の曲」を管理
if 'current_song_index' not in st.session_state:
    st.session_state['current_song_index'] = random.randint(0, len(songs) - 1)

# 次の曲を選ぶ関数
def next_song():
    st.session_state['current_song_index'] = random.randint(0, len(songs) - 1)

# 現在の曲を取得
current_idx = st.session_state['current_song_index']
current_song = songs[current_idx]

st.markdown("---")

# 曲情報の表示
st.subheader("Now Playing")
st.success(f"🎵 **{current_song['name']}**")

# 再生プレーヤー
st.audio(current_song['url'])

st.markdown("---")

# Nextボタン (幅いっぱいに表示)
if st.button("Next Song ⏭️", use_container_width=True):
    next_song()
    st.rerun()
