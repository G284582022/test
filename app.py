import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json

st.title("🚑 データ接続診断モード")

# 1. Secretsの確認
try:
    key_dict = json.loads(st.secrets["FIREBASE_KEY"])
    st.write("✅ Secretsの読み込み: 成功")
    # プロジェクトIDを表示（これで手元のjsonと同じか確認できます）
    st.info(f"接続先プロジェクトID: **{key_dict.get('project_id')}**")
except Exception as e:
    st.error(f"❌ Secretsエラー: {e}")
    st.stop()

# 2. Firebase接続
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        st.write("✅ Firebase初期化: 成功")
    except Exception as e:
        st.error(f"❌ Firebase初期化エラー: {e}")
        st.stop()

db = firestore.client()

# 3. データ取得テスト
st.write("---")
st.write("📂 データベースの中身をチェックします...")

try:
    # コレクション一覧を取得してみる
    cols = db.collections()
    col_names = [c.id for c in cols]
    
    if not col_names:
        st.warning("⚠️ データベース内に『コレクション』が1つもありません！")
        st.write("考えられる原因: アップロードが完了していないか、違うプロジェクトを見ています。")
    else:
        st.success(f"見つかったコレクション: {col_names}")
        
        if 'songs' in col_names:
            # songsの中身を数える
            docs = db.collection('songs').stream()
            count = sum(1 for _ in docs)
            st.metric("songsコレクションのデータ数", f"{count} 曲")
            
            if count == 0:
                st.error("songsコレクションはありますが、中身が空っぽです！")
            else:
                st.balloons()
                st.success("🎉 データは見つかりました！アプリのロジックを見直しましょう。")
        else:
            st.error("❌ 'songs' コレクションが見つかりません。アップロード時の名前を確認してください。")

except Exception as e:
    st.error(f"❌ 通信エラー: {e}")
