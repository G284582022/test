import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ==========================================
# 1. 設定 & Firebase接続
# ==========================================
st.set_page_config(page_title="Music Fusion Recommender", layout="wide")

# GitHub上のフォルダ名
AUDIO_DIR = "song"

# Firebase初期化 (Secrets利用)
if not firebase_admin._apps:
    try:
        # Secretsから鍵情報を読み込む
        key_dict = json.loads(st.secrets["FIREBASE_KEY"])
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase接続エラー: {e}")
        st.stop()

db = firestore.client()

# ==========================================
# 2. データロード
# ==========================================
@st.cache_data
def load_data_from_firebase():
    # 'songs' コレクションから全データを取得
    docs = db.collection('songs').stream()
    
    features_list = []
    filenames_list = []
    
    for doc in docs:
        data = doc.to_dict()
        # 特徴量
        vec = data.get('features')
        # Tempo (保存形式によって場所が違う場合に対応)
        if 'tempo' in data:
            vec.append(data['tempo'])
        
        if vec:
            features_list.append(vec)
            filenames_list.append(data.get('filename'))
        
    if not features_list:
        return None, None

    return np.array(features_list), np.array(filenames_list)

with st.spinner('データベースから楽曲情報を取得中...'):
    X, filenames = load_data_from_firebase()

if X is None or len(X) == 0:
    st.error("データベース(Firestore)に楽曲データがありません。")
    st.info("Spyderでアップロード用のコードを実行して、データを注入してください。")
    st.stop()

# ==========================================
# 3. クラスタリング & 代表曲選出
# ==========================================
# データ数に合わせてクラスター数を調整 (データが少なすぎる場合のエラー回避)
n_clusters = 6
if len(X) < 6:
    n_clusters = len(X) # データが6曲未満ならその数だけグループを作る

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)

# ==========================================
# 4. レーダーチャート描画関数
# ==========================================
def plot_radar(vec1, vec2=None, label1="Mix Target", label2="Recommendation"):
    def get_metrics(vec):
        # [0-12]:Timbre, [13-25]:Var, [26-49]:Chroma, [50-63]:Energy, [64]:Tempo
        # データ長チェック (念のため)
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
    metrics1 += metrics1[:1] # 閉じる
    
    labels = ['Tempo', 'Energy', 'Timbre', 'Variation']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    
    # ターゲット（青）
    ax.plot(angles, metrics1, color='#007AFF', linewidth=2, label=label1)
    ax.fill(angles, metrics1, color='#007AFF', alpha=0.2)

    # 推薦曲（赤）
    if vec2 is not None:
        metrics2 = scaler_radar.transform([get_metrics(vec2)])[0].tolist()
        metrics2 += metrics2[:1]
        ax.plot(angles, metrics2, color='#FF3B30', linewidth=2, linestyle='--', label=label2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    # 凡例を少し調整
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    return fig

# ==========================================
# 5. アプリ画面 UI
# ==========================================
st.title("🎛️ Music Fusion Recommender")
st.markdown("Firebase連携済み: 2曲を選んでミックスし、推薦を行います。")

# --- サイドバー ---
st.sidebar.header("1. Select 2 Songs")

# 選択肢の作成
options = {f"Group {i+1} ({filenames[closest[i]]})": closest[i] for i in range(n_clusters)}

# マルチセレクト（初期値として最初の2つを入れておく）
default_selections = list(options.keys())[:2] if len(options) >= 2 else list(options.keys())

selected_labels = st.sidebar.multiselect(
    "ミックスする曲を選択 (Max 2):",
    options.keys(),
    default=default_selections,
    max_selections=2
)

if len(selected_labels) < 2:
    st.warning("⚠️ 推薦を行うには、最低2曲を選んでください。")
    st.stop()

# インデックス取得
idx1 = options[selected_labels[0]]
idx2 = options[selected_labels[1]]

# ベクトル合成
mixed_vector = (X[idx1] + X[idx2]) / 2

# --- メインエリア ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎚️ 微調整 (Steering)")
    
    d_tempo = st.slider("Tempo (速さ)", -3.0, 3.0, 0.0)
    d_energy = st.slider("Energy (激しさ)", -3.0, 3.0, 0.0)
    d_timbre = st.slider("Timbre (音の厚み)", -3.0, 3.0, 0.0)

    # 調整
    target_vec = mixed_vector.copy()
    if len(target_vec) >= 65:
        target_vec[64] += d_tempo * np.std(X[:, 64]) * 0.5
        target_vec[50:57] += d_energy * np.std(X[:, 50:57]) * 0.2
        target_vec[0:13] += d_timbre * np.std(X[:, 0:13]) * 0.2

    # 検索
    sim_scores = cosine_similarity([target_vec], X)[0]
    sorted_indices = sim_scores.argsort()[::-1]
    
    # 選んだ曲そのものを除外して推薦
    rec_indices = [i for i in sorted_indices if i != idx1 and i != idx2]
    top_rec_idx = rec_indices[0] if rec_indices else sorted_indices[0]

with col2:
    st.subheader("🎯 推薦結果")
    rec_filename = filenames[top_rec_idx]
    st.success(f"**{rec_filename}**")
    
    # --- 再生機能（ここが安全装置！）---
    # GitHub上のパスを確認
    audio_path = os.path.join(AUDIO_DIR, rec_filename)
    
    # Web上でのファイル存在チェックは os.path.exists でOK
    if os.path.exists(audio_path):
        st.audio(audio_path)
    else:
        st.warning("⚠️ 音声ファイル未アップロード")
        st.caption(f"この曲({rec_filename})はデータベースに存在しますが、GitHubにMP3がありません。")

    # レーダーチャート
    st.pyplot(plot_radar(target_vec, X[top_rec_idx]))

# --- その他候補 ---
st.markdown("---")
st.write("### 📜 その他の候補")
cols = st.columns(3)
for i, r_idx in enumerate(rec_indices[1:4]):
    with cols[i]:
        fname = filenames[r_idx]
        st.write(f"**{i+2}. {fname}**")
        path = os.path.join(AUDIO_DIR, fname)
        if os.path.exists(path):
            st.audio(path)
        else:
            st.caption("No Audio File")
