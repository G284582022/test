import streamlit as st
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定
# ==========================================
st.set_page_config(page_title="Music Fusion Recommender", layout="wide")

# ★Web版では AUDIO_DIR (ローカルパス) は不要なので削除しました

# Firebase初期化 (Web用の書き方：Secretsを使う)
if not firebase_admin._apps:
    try:
        # Streamlit Cloudの「Secrets」から鍵情報を受け取る
        key_dict = json.loads(st.secrets["FIREBASE_KEY"])
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase接続エラー: {e}")
        st.stop()

db = firestore.client()

# ==========================================
# 2. データロード (URL取得に対応)
# ==========================================
@st.cache_data
def load_data_from_firebase():
    # Firestoreから全曲データを取得
    docs = db.collection('songs').stream()
    
    features_list = []
    filenames_list = []
    
    for doc in docs:
        data = doc.to_dict()
        vec = data.get('features')
        
        # 特徴量があるデータだけを使う
        if vec:
            # Tempoを結合
            if 'tempo' in data:
                vec.append(data['tempo'])
            features_list.append(vec)
            
            # ★重要：ファイル名と一緒に「再生URL」も保存する
            filenames_list.append({
                'name': data.get('filename', 'Unknown'),
                'url': data.get('audio_url', None) 
            })
        
    if not features_list:
        return None, None

    return np.array(features_list), np.array(filenames_list)

with st.spinner('データベースから609曲の情報を取得中...'):
    X, song_data = load_data_from_firebase()

if X is None or len(X) == 0:
    st.error("データベースにデータがありません。")
    st.info("データのアップロードが完了しているか確認してください。")
    st.stop()

# 表示用のファイル名リストを作る
filenames = [item['name'] for item in song_data]

# ==========================================
# 3. クラスタリング & 代表曲選出
# ==========================================
n_clusters = 6
if len(X) < 6: n_clusters = len(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)

# ==========================================
# 4. レーダーチャート
# ==========================================
def plot_radar(vec1, vec2=None, label1="Mix Target", label2="Rec"):
    def get_metrics(vec):
        if len(vec) < 65: return [0,0,0,0]
        tempo = vec[64]
        energy = np.mean(vec[50:57])
        timbre = np.mean(vec[0:13])
        variation = np.mean(vec[13:26])
        return [tempo, energy, timbre, variation]

    all_metrics = np.array([get_metrics(x) for x in X])
    scaler_radar = MinMaxScaler()
    scaler_radar.fit(all_metrics)

    metrics1 = scaler_radar.transform([get_metrics(vec1)])[0].tolist()
    metrics1 += metrics1[:1]
    
    labels = ['Tempo', 'Energy', 'Timbre', 'Variation']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, metrics1, color='#007AFF', linewidth=2, label=label1)
    ax.fill(angles, metrics1, color='#007AFF', alpha=0.2)

    if vec2 is not None:
        metrics2 = scaler_radar.transform([get_metrics(vec2)])[0].tolist()
        metrics2 += metrics2[:1]
        ax.plot(angles, metrics2, color='#FF3B30', linewidth=2, linestyle='--', label=label2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig

# ==========================================
# 5. アプリ画面 UI
# ==========================================
st.title("🎛️ Music Fusion Recommender")
st.caption(f"Connected to Cloud Storage: {len(X)} songs loaded")

# --- サイドバー ---
st.sidebar.header("1. Select 2 Songs")
options = {f"Group {i+1} ({filenames[closest[i]]})": closest[i] for i in range(n_clusters)}
default_sel = list(options.keys())[:2] if len(options)>=2 else list(options.keys())

selected_labels = st.sidebar.multiselect("Select 2 songs:", options.keys(), default=default_sel, max_selections=2)

if len(selected_labels) < 2:
    st.warning("Please select 2 songs.")
    st.stop()

idx1 = options[selected_labels[0]]
idx2 = options[selected_labels[1]]
mixed_vector = (X[idx1] + X[idx2]) / 2

# --- メインエリア ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎚️ Steering")
    d_tempo = st.slider("Tempo", -3.0, 3.0, 0.0)
    d_energy = st.slider("Energy", -3.0, 3.0, 0.0)
    d_timbre = st.slider("Timbre", -3.0, 3.0, 0.0)

    target_vec = mixed_vector.copy()
    if len(target_vec) >= 65:
        target_vec[64] += d_tempo * np.std(X[:, 64]) * 0.5
        target_vec[50:57] += d_energy * np.std(X[:, 50:57]) * 0.2
        target_vec[0:13] += d_timbre * np.std(X[:, 0:13]) * 0.2

    sim_scores = cosine_similarity([target_vec], X)[0]
    sorted_indices = sim_scores.argsort()[::-1]
    rec_indices = [i for i in sorted_indices if i != idx1 and i != idx2]
    top_rec_idx = rec_indices[0] if rec_indices else sorted_indices[0]

with col2:
    st.subheader("🎯 Recommendation")
    
    # 辞書データから情報を取り出す
    rec_data = song_data[top_rec_idx]
    
    st.success(f"**{rec_data['name']}**")
    
    # ★ここが重要！URLを使って再生します
    audio_url = rec_data['url']
    if audio_url:
        st.audio(audio_url)
    else:
        st.warning("音声URLがデータベースに見つかりません")
    
    st.pyplot(plot_radar(target_vec, X[top_rec_idx]))

st.markdown("---")
st.write("### 📜 Other Candidates")
cols = st.columns(3)
for i, r_idx in enumerate(rec_indices[1:4]):
    with cols[i]:
        d = song_data[r_idx]
        st.write(f"**{i+2}. {d['name']}**")
        # 他の候補もURLで再生
        if d['url']:
            st.audio(d['url'])
