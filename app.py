import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

st.set_page_config(page_title="Music Fusion Recommender", layout="wide")

import json
AUDIO_DIR = "song"

# Firebase初期化
if not firebase_admin._apps:
    try:
        # ★ここが修正ポイント！
        # KEY_PATH ではなく、st.secrets から直接読み込みます
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
    docs = db.collection('songs').stream()
    features_list = []
    filenames_list = []
    
    for doc in docs:
        data = doc.to_dict()
        vec = data['features']
        if 'tempo' in data:
            vec.append(data['tempo'])
        features_list.append(vec)
        filenames_list.append(data['filename'])
        
    if not features_list: return None, None
    return np.array(features_list), np.array(filenames_list)

with st.spinner('Firebaseからデータを取得中...'):
    X, filenames = load_data_from_firebase()

if X is None:
    st.error("データがありません。")
    st.stop()

# ==========================================
# 3. 代表曲の特定 (6曲)
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=6, random_state=42)
labels = kmeans.fit_predict(X_scaled)
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)

# ==========================================
# 4. レーダーチャート関数
# ==========================================
def plot_radar(vec1, vec2=None, label1="Mix Base", label2="Recommendation"):
    def get_metrics(vec):
        # [0-12]:Timbre, [13-25]:Var, [26-49]:Chroma, [50-63]:Energy, [64]:Tempo
        tempo = vec[64]
        energy = np.mean(vec[50:57])
        timbre = np.mean(vec[0:13])
        variation = np.mean(vec[13:26])
        return [tempo, energy, timbre, variation]

    # 全体スケール用
    all_metrics = np.array([get_metrics(x) for x in X])
    scaler_radar = MinMaxScaler()
    scaler_radar.fit(all_metrics)

    # データ準備
    metrics1 = scaler_radar.transform([get_metrics(vec1)])[0].tolist()
    metrics1 += metrics1[:1]
    
    labels = ['Tempo', 'Energy', 'Timbre', 'Variation']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    
    # ベース（2曲のミックス）
    ax.plot(angles, metrics1, color='#007AFF', linewidth=2, label=label1)
    ax.fill(angles, metrics1, color='#007AFF', alpha=0.2)

    # 推薦曲
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
st.markdown("異なる2つの曲を選択して、その「中間」にある曲を探します。")

# --- サイドバー: 2曲選択 ---
st.sidebar.header("1. Select 2 Songs")
st.sidebar.write("ミックスしたい曲調を2つ選んでください")

# 選択肢の作成
options = {f"Group {i+1} ({filenames[closest[i]]})": closest[i] for i in range(6)}
selected_labels = st.sidebar.multiselect(
    "代表曲リスト:",
    options.keys(),
    max_selections=2
)

# --- 2曲選ばれていない場合の処理 ---
if len(selected_labels) < 2:
    st.info("👈 サイドバーから、混ぜ合わせたい曲を **2つ** 選んでください。")
    
    # 参考用に全代表曲を表示
    st.subheader("代表曲リスト (ここから2つ選べます)")
    cols = st.columns(3)
    for i in range(6):
        with cols[i%3]:
            idx = closest[i]
            st.write(f"**Group {i+1}**")
            audio_path = os.path.join(AUDIO_DIR, filenames[idx])
            if os.path.exists(audio_path):
                st.audio(audio_path)
            else:
                st.write(filenames[idx])
    st.stop()

# --- 2曲選ばれたあとの処理 ---
idx1 = options[selected_labels[0]]
idx2 = options[selected_labels[1]]

# ベクトル合成 (平均を取る)
mixed_vector = (X[idx1] + X[idx2]) / 2

st.sidebar.success(f"Mix created from:\n- {filenames[idx1]}\n- {filenames[idx2]}")

# --- メインエリア ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎚️ 微調整 (Steering)")
    st.write("2曲の中間地点から、さらに好みを調整します。")
    
    delta_tempo = st.slider("Tempo (速さ)", -3.0, 3.0, 0.0)
    delta_energy = st.slider("Energy (激しさ)", -3.0, 3.0, 0.0)
    delta_timbre = st.slider("Timbre (音の厚み)", -3.0, 3.0, 0.0)

    # ユーザー調整を加える
    final_target_vector = mixed_vector.copy()
    
    # 統計量を使って調整幅を決める
    final_target_vector[64] += delta_tempo * np.std(X[:, 64]) * 0.5     # Tempo
    final_target_vector[50:57] += delta_energy * np.std(X[:, 50:57]) * 0.2 # Energy
    final_target_vector[0:13] += delta_timbre * np.std(X[:, 0:13]) * 0.2   # Timbre

    # 検索実行
    sim_scores = cosine_similarity([final_target_vector], X)[0]
    sorted_indices = sim_scores.argsort()[::-1]
    
    # 自分自身（選んだ2曲）が1位に出てくるのを防ぐ
    recommendations = []
    for idx in sorted_indices:
        if idx != idx1 and idx != idx2: # 選んだ曲以外
            recommendations.append(idx)
        if len(recommendations) >= 3: # トップ3まで取得
            break
            
    top_rec_idx = recommendations[0]

with col2:
    st.subheader(" 推薦結果 (Fusion Result)")
    
    st.success(f"**Best Match:** {filenames[top_rec_idx]}")
    
    # 再生
    rec_path = os.path.join(AUDIO_DIR, filenames[top_rec_idx])
    if os.path.exists(rec_path):
        st.audio(rec_path)
    else:
        st.warning("File not found")

    # レーダーチャートで比較
    # 青色: 選んだ2曲のミックス + スライダー調整
    # 赤色: 実際に推薦された曲
    st.pyplot(plot_radar(final_target_vector, X[top_rec_idx], 
                         label1="Your Mix Target", label2="Recommended Song"))

st.markdown("---")
st.write("###その他の候補 (Top 2 & 3)")
sub_cols = st.columns(2)
for i, idx in enumerate(recommendations[1:3]):
    with sub_cols[i]:
        st.write(f"**{i+2}. {filenames[idx]}**")
        path = os.path.join(AUDIO_DIR, filenames[idx])
        if os.path.exists(path):
            st.audio(path)
