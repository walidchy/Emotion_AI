import streamlit as st
from src.predict import EmotionPredictor

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Emotion Detector",
    layout="centered"
)

# =========================
# Session State Init
# =========================
if "predictor" not in st.session_state:
    st.session_state.predictor = EmotionPredictor("lstm")

if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# =========================
# Helper function (IMPORTANT)
# =========================
def set_example_text(text):
    st.session_state.user_text = text

# =========================
# Title
# =========================
st.title("Emotion Detector")
st.markdown("Analyze emotions in your text")

# =========================
# Emotion Table
# =========================
EMOTION_TABLE = {
    0: {"name": "SADNESS", "color": "#E3F2FD", "icon": "😢"},
    1: {"name": "JOY", "color": "#FFF9C4", "icon": "😊"},
    2: {"name": "LOVE", "color": "#FFE0E0", "icon": "❤️"},
    3: {"name": "ANGER", "color": "#FFEBEE", "icon": "😠"},
    4: {"name": "FEAR", "color": "#F3E5F5", "icon": "😨"},
    5: {"name": "SURPRISE", "color": "#E8F5E8", "icon": "😲"}
}

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Model:",
        ['lstm', 'naive_bayes', 'svm', 'knn', 'rnn', 'random_forest'],
        index=0
    )

    if st.button("Load Model"):
        st.session_state.predictor = EmotionPredictor(model_choice)
        st.success(f"Model {model_choice} loaded")

# =========================
# Input Area
# =========================
st.subheader("Enter your text")

user_input = st.text_area(
    "Type your text here:",
    height=100,
    key="user_text"
)

col1, col2 = st.columns(2)

with col1:
    analyze_btn = st.button(
        "Analyze",
        type="primary",
        use_container_width=True
    )

with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.user_text = ""
        st.rerun()

# =========================
# Analysis
# =========================
if analyze_btn and user_input.strip():
    with st.spinner("Analyzing..."):
        result = st.session_state.predictor.predict(user_input)

    st.divider()
    st.subheader("Result")

    emotion_code = result["emotion"]
    emotion_info = EMOTION_TABLE.get(emotion_code, EMOTION_TABLE[0])

    st.markdown(
        f"""
        <div style="padding:20px;
                    background-color:{emotion_info['color']};
                    border-radius:10px;
                    text-align:center;">
            <div style="font-size:48px;">{emotion_info['icon']}</div>
            <h3>{emotion_info['name']}</h3>
            <p>
                Code: {emotion_code} |
                Confidence: {result['confidence']:.1%}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ❌ SUPPRIMÉ: st.info(f"**Analyzed text:** {user_input}")

# =========================
# Examples (FIXED)
# =========================
st.divider()
st.subheader("Examples")

examples = [
    "I feel sad",
    "I am very happy",
    "I love you so much",
    "I am angry",
    "I am afraid",
    "What a surprise!"
]

cols = st.columns(3)

for i, text in enumerate(examples):
    with cols[i % 3]:
        st.button(
            text,
            key=f"ex_{i}",
            use_container_width=True,
            on_click=set_example_text,
            args=(text,)
        )

# =========================
# Footer
# =========================
st.divider()
st.caption("Emotion Detection System | Classification 0–5")