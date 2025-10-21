import streamlit as st
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import pandas as pd
from datetime import datetime

# ==== PAGE CONFIG & CSS ====
st.set_page_config(page_title="Bộ lọc Email Spam – MLP", page_icon="🧠", layout="centered")

CUSTOM_CSS = """
<style>
.small {font-size: 0.86rem; opacity: .8}
.badge {display:inline-block;padding:6px 10px;border-radius:999px;font-weight:600;}
.badge-ok {background:#e9fff1;color:#0a7d3b;border:1px solid #bff3d0;}
.badge-spam {background:#fff1f1;color:#b00020;border:1px solid #ffd2d2;}
.card {padding:18px 20px;border:1px solid #eee;border-radius:16px;
       background:#fff;box-shadow:0 2px 14px rgba(0,0,0,.04)}
.kq {font-size:1.25rem;font-weight:700;margin-bottom:0}
.mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
.tips li{margin-bottom:.25rem}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🧠 Bộ lọc Email Spam – MLP (nơ-ron nhẹ)")
st.caption("TF-IDF (char 3–5 gram) + MLPClassifier • Tối ưu cho tiếng Việt • UI đẹp & nhanh")

# ==== LOAD MODEL (có Fallback) ====
@st.cache_resource(show_spinner=False)
def load_or_build_model():
    p = Path("model.pkl")
    if p.exists():
        try:
            model = joblib.load(p)
            return model, "model.pkl (đã huấn luyện)"
        except Exception as e:
            st.warning(f"Không load được model.pkl ({e}). Sẽ dùng model tạm.")

    # Fallback mini model (đủ chạy demo)
    texts = [
        "Nhận quà tặng khủng, bấm link để nhận thưởng ngay",
        "Chúc mừng trúng iPhone 15, xác nhận tại đây",
        "Vay tiền nhanh lãi suất 0%, click ngay",
        "Miễn phí 100% phí dịch vụ, cập nhật theo link",
        "Mời bạn tham dự phỏng vấn vào thứ Hai tuần tới",
        "Đính kèm báo cáo doanh số tháng 10",
        "Lịch họp dự án lúc 9h sáng mai",
        "Chúc mừng bạn đã trúng tuyển, vui lòng xác nhận thời gian",
    ]
    labels = [1,1,1,1,0,0,0,0]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5))),
        ("mlp",  MLPClassifier(hidden_layer_sizes=(128,64), activation="relu",
                                learning_rate_init=1e-3, alpha=1e-4,
                                max_iter=80, random_state=42))
    ]).fit(texts, labels)
    return model, "model MLP tạm trong app"

model, model_source = load_or_build_model()

# ==== SIDEBAR ====
with st.sidebar:
    st.subheader("⚙️ Cài đặt")
    threshold = st.slider("Ngưỡng phân loại", 0.1, 0.9, 0.50, 0.05, help="≥ ngưỡng → SPAM")
    st.markdown(f"**Nguồn mô hình:** `{model_source}`")
    st.markdown("---")
    st.markdown("**📘 Hướng dẫn nhanh**")
    st.markdown(
        "<ul class='tips'>"
        "<li>Nhập tiêu đề + nội dung → bấm *Kiểm tra Spam*.</li>"
        "<li>Tăng ngưỡng nếu chặn nhầm; giảm nếu lọt spam.</li>"
        "<li>Muốn chính xác cao: huấn luyện riêng & upload <code>model.pkl</code>.</li>"
        "</ul>", unsafe_allow_html=True
    )

# ==== TABS ====
tab1, tab2, tab3 = st.tabs(["🔎 Kiểm tra", "✨ Ví dụ nhanh", "📜 Lịch sử"])

# -- TAB 1: KIỂM TRA
with tab1:
    col1, col2 = st.columns([3,1])
    with col1:
        st.text_input("Tiêu đề Email", key="subject", placeholder="VD: Thông báo phỏng vấn")
    st.text_area("Nội dung Email", key="body", height=220,
                 placeholder="Dán nội dung email tiếng Việt tại đây...")

    if st.button("Kiểm tra Spam", use_container_width=True):
        subject = st.session_state.get("subject", "")
        body = st.session_state.get("body", "")
        text = (subject + " " + body).strip()

        if not text:
            st.info("Vui lòng nhập ít nhất tiêu đề hoặc nội dung.")
        else:
            proba = float(model.predict_proba([text])[0, 1])
            is_spam = proba >= threshold
            label = "🚨 SPAM" if is_spam else "✅ Không phải SPAM"

            # Thẻ kết quả
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<div class='kq'>{label}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='small mono'>Xác suất spam: {proba:.3f} • Ngưỡng: {threshold:.2f}</div>",
                unsafe_allow_html=True
            )
            badge = "badge-spam" if is_spam else "badge-ok"
            st.markdown(f"<div class='badge {badge}'>Kết luận</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Lưu lịch sử trong phiên
            row = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "kq": "SPAM" if is_spam else "Not spam",
                "proba": round(proba, 3),
                "subject": subject[:60],
                "excerpt": body[:80].replace("\n", " ")
            }
            st.session_state.setdefault("hist", []).insert(0, row)

# -- TAB 2: VÍ DỤ NHANH
with tab2:
    colA, colB = st.columns(2)
    with colA:
        if st.button("📩 Ví dụ HAM (không spam)", use_container_width=True):
            st.session_state["subject"] = "Mời bạn tham dự phỏng vấn"
            st.session_state["body"] = "Chúng tôi mời bạn tham dự phỏng vấn lúc 9h sáng thứ Hai tuần tới."
            st.success("Đã điền ví dụ vào tab 🔎 Kiểm tra")
    with colB:
        if st.button("🚨 Ví dụ SPAM", use_container_width=True):
            st.session_state["subject"] = "Trúng thưởng iPhone 15"
            st.session_state["body"] = "Chúc mừng! Nhấn vào link để xác nhận và nhận quà ngay hôm nay."
            st.success("Đã điền ví dụ vào tab 🔎 Kiểm tra")

# -- TAB 3: LỊCH SỬ
with tab3:
    hist = st.session_state.get("hist", [])
    if hist:
        df = pd.DataFrame(hist)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Tải lịch sử CSV", csv, "history.csv", "text/csv")
        st.caption("Lưu tạm trong phiên làm việc này (không lưu ra server).")
    else:
        st.info("Chưa có lịch sử. Hãy kiểm tra vài email trước đã.")
