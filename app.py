import streamlit as st
import joblib
import numpy as np
import time
import os
st.set_page_config(
    page_title="Emotion AI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
def load_custom_css():
    st.markdown("""
    <style>
    /* ── Google Font Import ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset & Base ── */
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Hide Streamlit Default Elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Main Container ── */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1100px;
    }

    /* ── Hero Header Section ── */
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.05) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        border: 1px solid rgba(102, 126, 234, 0.5);
        color: #a78bfa;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #a78bfa 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 1rem 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(200, 200, 230, 0.8);
        font-weight: 400;
        margin: 0;
        letter-spacing: 0.3px;
    }

    .hero-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* ── Stats Row ── */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.8rem;
        flex-wrap: wrap;
    }

    .stat-chip {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.5rem 1.2rem;
        color: rgba(200, 200, 230, 0.9);
        font-size: 0.8rem;
        font-weight: 500;
    }

    .stat-chip span {
        color: #a78bfa;
        font-weight: 700;
    }

    /* ── Card Style ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
        transition: border-color 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
    }

    .card-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Text Area Overrides ── */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 14px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        padding: 1rem 1.2rem !important;
        resize: vertical !important;
        transition: border-color 0.3s ease !important;
        min-height: 130px !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
        outline: none !important;
    }

    .stTextArea textarea::placeholder {
        color: rgba(160, 160, 200, 0.5) !important;
    }

    .stTextArea label {
        color: rgba(200, 200, 230, 0.7) !important;
        font-size: 0.85rem !important;
        display: none !important;
    }

    /* ── Button Overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #7c8ef0 0%, #8a5cb8 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Result Card ── */
    .result-card {
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: slideUp 0.5s ease-out;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    .result-emoji {
        font-size: 5rem;
        display: block;
        margin-bottom: 1rem;
        animation: bounceIn 0.6s ease-out;
        filter: drop-shadow(0 0 20px rgba(255,255,255,0.3));
    }

    @keyframes bounceIn {
        0%   { transform: scale(0); opacity: 0; }
        60%  { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }

    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }

    .result-emotion {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0 0 1rem 0;
        letter-spacing: -0.5px;
    }

    .result-description {
        font-size: 0.95rem;
        opacity: 0.75;
        max-width: 400px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Emotion Themes ── */
    .emotion-joy {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(251, 191, 36, 0.4);
        color: #fbbf24;
    }

    .emotion-sadness {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #60a5fa;
    }

    .emotion-anger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
    }

    .emotion-love {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.15), rgba(190, 24, 93, 0.1));
        border: 1px solid rgba(236, 72, 153, 0.4);
        color: #f472b6;
    }

    .emotion-fear {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(109, 40, 217, 0.1));
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #c4b5fd;
    }

    .emotion-surprise {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
    }

    .emotion-default {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.4);
        color: #a78bfa;
    }

    /* ── Insight Bar ── */
    .insight-container {
        margin-top: 1.5rem;
    }

    .insight-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: rgba(200, 200, 230, 0.6);
        margin-bottom: 0.8rem;
    }

    .insight-tag {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.25rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: rgba(210, 210, 240, 0.9);
    }

    /* ── Sidebar Overrides ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(200, 200, 230, 0.9) !important;
    }

    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 1.5rem;
    }

    .sidebar-logo-icon {
        font-size: 3rem;
        display: block;
        margin-bottom: 0.5rem;
    }

    .sidebar-logo-title {
        font-size: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #a78bfa !important;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(102, 126, 234, 0.2);
    }

    .sidebar-dev-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        text-align: center;
    }

    .dev-avatar {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 0.5rem;
    }

    .dev-name {
        font-size: 0.9rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .dev-role {
        font-size: 0.75rem;
        color: #a78bfa;
        margin-top: 0.2rem;
    }

    .tech-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.12);
        border: 1px solid rgba(102, 126, 234, 0.25);
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        margin: 0.2rem;
    }

    .emotion-legend {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.45rem 0;
        font-size: 0.8rem;
        color: rgba(200, 200, 230, 0.85);
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }

    .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Warning / Info Boxes ── */
    .custom-info {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-left: 3px solid #667eea;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: rgba(200, 210, 255, 0.9);
        font-size: 0.85rem;
        margin-top: 1rem;
    }

    .custom-error {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 3px solid #ef4444;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: rgba(255, 180, 180, 0.9);
        font-size: 0.85rem;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.4), transparent);
        margin: 1.5rem 0;
    }

    /* ── Spinner Override ── */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    /* ── Column gap ── */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: rgba(150, 150, 200, 0.5);
        font-size: 0.78rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 2rem;
    }

    .app-footer a {
        color: #a78bfa;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)

EMOTION_CONFIG = {
    "joy": {
        "emoji": "😊",
        "css_class": "emotion-joy",
        "description": "The text carries a strong sense of happiness, positivity, and delight.",
        "color": "#fbbf24",
        "keywords": ["Positive", "Uplifting", "Happy", "Cheerful"],
        "dot_color": "#fbbf24"
    },
    "happiness": {
        "emoji": "😊",
        "css_class": "emotion-joy",
        "description": "The text radiates warmth, joy, and a positive emotional state.",
        "color": "#fbbf24",
        "keywords": ["Positive", "Joyful", "Content", "Grateful"],
        "dot_color": "#fbbf24"
    },
    "sadness": {
        "emoji": "😢",
        "css_class": "emotion-sadness",
        "description": "The text reflects feelings of sorrow, grief, or emotional pain.",
        "color": "#60a5fa",
        "keywords": ["Melancholy", "Grief", "Longing", "Blue"],
        "dot_color": "#60a5fa"
    },
    "anger": {
        "emoji": "😡",
        "css_class": "emotion-anger",
        "description": "The text conveys frustration, rage, or strong displeasure.",
        "color": "#f87171",
        "keywords": ["Frustrated", "Intense", "Irritated", "Fierce"],
        "dot_color": "#f87171"
    },
    "love": {
        "emoji": "❤️",
        "css_class": "emotion-love",
        "description": "The text expresses deep affection, care, and emotional warmth.",
        "color": "#f472b6",
        "keywords": ["Affectionate", "Warm", "Caring", "Romantic"],
        "dot_color": "#f472b6"
    },
    "fear": {
        "emoji": "😨",
        "css_class": "emotion-fear",
        "description": "The text reflects anxiety, apprehension, or a sense of dread.",
        "color": "#c4b5fd",
        "keywords": ["Anxious", "Nervous", "Scared", "Uneasy"],
        "dot_color": "#c4b5fd"
    },
    "surprise": {
        "emoji": "😲",
        "css_class": "emotion-surprise",
        "description": "The text shows astonishment, amazement, or an unexpected reaction.",
        "color": "#34d399",
        "keywords": ["Amazed", "Unexpected", "Shocked", "Astonished"],
        "dot_color": "#34d399"
    },
    "neutral": {
        "emoji": "😐",
        "css_class": "emotion-default",
        "description": "The text appears balanced with no strong emotional lean.",
        "color": "#a78bfa",
        "keywords": ["Balanced", "Calm", "Neutral", "Objective"],
        "dot_color": "#a78bfa"
    }
}

def get_emotion_config(emotion_label: str) -> dict:
    """
    Retrieve emotion config by label (case-insensitive).
    Falls back to a default config if label is not mapped.
    """
    key = emotion_label.lower().strip()
    if key in EMOTION_CONFIG:
        return EMOTION_CONFIG[key]
    for k, v in EMOTION_CONFIG.items():
        if k in key or key in k:
            return v
    # Default fallback
    return {
        "emoji": "🤔",
        "css_class": "emotion-default",
        "description": f"The model detected a '{emotion_label}' emotional pattern in the text.",
        "color": "#a78bfa",
        "keywords": ["Detected", "Classified", "Analyzed"],
        "dot_color": "#a78bfa"
    }
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the trained SVM pipeline from disk.
    Returns the pipeline object or None on failure.
    """
    try:
        if not os.path.exists("emotion_svm_model.pkl"):
            return None, "❌ Model file 'emotion_svm_model.pkl' not found."
        model = joblib.load("emotion_svm_model.pkl")
        return model, None
    except Exception as e:
        return None, f"❌ Failed to load model: {str(e)}"


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    """
    Load the LabelEncoder from disk.
    Returns the encoder object or None on failure.
    """
    try:
        if not os.path.exists("label_encoder.pkl"):
            return None, "❌ Label encoder file 'label_encoder.pkl' not found."
        encoder = joblib.load("label_encoder.pkl")
        return encoder, None
    except Exception as e:
        return None, f"❌ Failed to load label encoder: {str(e)}"
def predict_emotion(text: str, model, label_encoder) -> dict:
    """
    Run the NLP pipeline and return structured prediction results.

    Args:
        text: Raw input text from user
        model: Loaded SVM pipeline (TF-IDF + LinearSVC)
        label_encoder: Fitted LabelEncoder

    Returns:
        dict with emotion label, config, and analysis metadata
    """
    numeric_pred = model.predict([text])[0]

    emotion_label = label_encoder.inverse_transform([numeric_pred])[0]

    try:
        decision_scores = model.decision_function([text])[0]
        classes = label_encoder.classes_

        min_score = np.min(decision_scores)
        max_score = np.max(decision_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        normalized_scores = (decision_scores - min_score) / score_range
        scored_classes = sorted(
            zip(classes, normalized_scores),
            key=lambda x: x[1],
            reverse=True
        )
        top_score = float(normalized_scores[np.argmax(decision_scores)])
    except Exception:
        scored_classes = [(emotion_label, 1.0)]
        top_score = 1.0
    words = text.split()
    word_count = len(words)

    return {
        "emotion": emotion_label,
        "config": get_emotion_config(emotion_label),
        "scored_classes": scored_classes[:6],   # top 6 for display
        "top_score": top_score,
        "word_count": word_count,
        "char_count": len(text)
    }
def build_sidebar():
    """Render the left sidebar with developer info and metadata."""
    with st.sidebar:
        # ── Logo Block ──
        st.markdown("""
        <div class="sidebar-logo">
            <span class="sidebar-logo-icon">🧠</span>
            <div class="sidebar-logo-title">Emotion AI Classifier</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Developer Card ──
        st.markdown("""
        <div class="sidebar-section-title">👤 Developer</div>
        <div class="sidebar-dev-card">
            <span class="dev-avatar">🧑‍💻</span>
            <div class="dev-name">Mahadi Ur Rehman Siddiqui</div>
            <div class="dev-role">AI / ML Engineer</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Technology Stack ──
        st.markdown("""
        <div class="sidebar-section-title">⚙️ Tech Stack</div>
        """, unsafe_allow_html=True)

        techs = [
            ("🐍", "Python"),
            ("🤖", "Scikit-learn"),
            ("📊", "TF-IDF"),
            ("⚡", "Linear SVM"),
            ("💬", "NLP"),
            ("🌐", "Streamlit"),
            ("📦", "Joblib"),
            ("🔢", "NumPy"),
        ]
        badges_html = "".join(
            f'<span class="tech-badge">{icon} {name}</span>'
            for icon, name in techs
        )
        st.markdown(f'<div style="line-height:2">{badges_html}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-section-title">🎭 Emotion Categories</div>
        """, unsafe_allow_html=True)

        legend_items = [
            ("#fbbf24", "😊", "Joy / Happiness"),
            ("#60a5fa", "😢", "Sadness"),
            ("#f87171", "😡", "Anger"),
            ("#f472b6", "❤️", "Love"),
            ("#c4b5fd", "😨", "Fear"),
            ("#34d399", "😲", "Surprise"),
            ("#a78bfa", "😐", "Neutral"),
        ]
        for color, emoji, label in legend_items:
            st.markdown(f"""
            <div class="emotion-legend">
                <div class="legend-dot" style="background:{color};
                     box-shadow:0 0 6px {color}80;"></div>
                <span>{emoji} {label}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Model Info ──
        st.markdown("""
        <div class="sidebar-section-title">🔬 Model Info</div>
        """, unsafe_allow_html=True)

        model_info = {
            "Algorithm": "Linear SVM",
            "Features": "TF-IDF Vectors",
            "Task": "Multi-class Classification",
            "Framework": "Scikit-learn",
        }
        for k, v in model_info.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding: 0.35rem 0; border-bottom:1px solid rgba(255,255,255,0.05);
                        font-size:0.78rem; color:rgba(200,200,230,0.8)">
                <span style="color:rgba(160,160,210,0.6)">{k}</span>
                <span style="font-weight:600">{v}</span>
            </div>
            """, unsafe_allow_html=True)
def render_result(result: dict):
    """Render the prediction result card and analysis insights."""
    cfg = result["config"]
    emotion = result["emotion"].capitalize()
    st.markdown(f"""
    <div class="result-card {cfg['css_class']}">
        <span class="result-emoji">{cfg['emoji']}</span>
        <div class="result-label" style="color:{cfg['color']}">
            Detected Emotion
        </div>
        <div class="result-emotion" style="color:{cfg['color']}">
            {emotion}
        </div>
        <div class="result-description" style="color:rgba(220,220,255,0.75)">
            {cfg['description']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; font-weight:600; letter-spacing:1.5px;
                text-transform:uppercase; color:rgba(160,160,210,0.6);
                margin-bottom:1rem;">
        📐 SVM Decision Analysis
    </div>
    """, unsafe_allow_html=True)

    # SVM note
    st.markdown("""
    <div class="custom-info">
        <strong>ℹ️ About SVM Scoring:</strong> LinearSVC uses a decision function
        (distance from the hyperplane) rather than probability scores.
        The bars below show <em>relative confidence</em> — how far each class is
        from the decision boundary. A larger distance = stronger classification signal.
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    for class_name, norm_score in result["scored_classes"]:
        bar_config = get_emotion_config(class_name)
        is_predicted = class_name.lower() == result["emotion"].lower()

        label_html = (
            f'<strong style="color:{bar_config["color"]}">'
            f'{bar_config["emoji"]} {class_name.capitalize()}'
            f'{"  ✓" if is_predicted else ""}</strong>'
        )
        pct = int(norm_score * 100)

        col1, col2 = st.columns([3, 7])
        with col1:
            st.markdown(
                f'<div style="text-align:right; font-size:0.82rem; '
                f'padding-top:0.4rem;">{label_html}</div>',
                unsafe_allow_html=True
            )
        with col2:
            bar_html = f"""
            <div style="background:rgba(255,255,255,0.06); border-radius:50px;
                        height:10px; margin-top:0.55rem; overflow:hidden;">
                <div style="width:{pct}%; height:100%; border-radius:50px;
                            background: linear-gradient(90deg,
                                {bar_config['dot_color']}aa,
                                {bar_config['dot_color']});
                            box-shadow: 0 0 8px {bar_config['dot_color']}60;
                            transition: width 1s ease;">
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    tags_html = "".join(
        f'<span class="insight-tag">{tag}</span>'
        for tag in cfg["keywords"]
    )
    st.markdown(f"""
    <div class="insight-container">
        <div class="insight-title">🏷️ Associated Attributes</div>
        {tags_html}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    stat_col1, stat_col2, stat_col3 = st.columns(3)

    with stat_col1:
        st.markdown(f"""
        <div style="text-align:center; padding:1rem;
                    background:rgba(255,255,255,0.03); border-radius:12px;
                    border:1px solid rgba(255,255,255,0.07)">
            <div style="font-size:1.6rem; font-weight:700;
                        color:{cfg['color']}">{result['word_count']}</div>
            <div style="font-size:0.7rem; color:rgba(160,160,200,0.7);
                        margin-top:0.2rem; text-transform:uppercase;
                        letter-spacing:1px">Words</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col2:
        st.markdown(f"""
        <div style="text-align:center; padding:1rem;
                    background:rgba(255,255,255,0.03); border-radius:12px;
                    border:1px solid rgba(255,255,255,0.07)">
            <div style="font-size:1.6rem; font-weight:700;
                        color:{cfg['color']}">{result['char_count']}</div>
            <div style="font-size:0.7rem; color:rgba(160,160,200,0.7);
                        margin-top:0.2rem; text-transform:uppercase;
                        letter-spacing:1px">Characters</div>
        </div>
        """, unsafe_allow_html=True)

    with stat_col3:
        avg_len = (
            round(result['char_count'] / result['word_count'], 1)
            if result['word_count'] > 0 else 0
        )
        st.markdown(f"""
        <div style="text-align:center; padding:1rem;
                    background:rgba(255,255,255,0.03); border-radius:12px;
                    border:1px solid rgba(255,255,255,0.07)">
            <div style="font-size:1.6rem; font-weight:700;
                        color:{cfg['color']}">{avg_len}</div>
            <div style="font-size:0.7rem; color:rgba(160,160,200,0.7);
                        margin-top:0.2rem; text-transform:uppercase;
                        letter-spacing:1px">Avg Word Len</div>
        </div>
        """, unsafe_allow_html=True)

def render_examples() -> str | None:
    """Render example prompt buttons; return clicked text or None."""
    examples = [
        ("😊", "I am so happy today!"),
        ("😢", "I feel so lonely and broken."),
        ("😡", "This is absolutely unacceptable!"),
        ("❤️", "I love you with all my heart."),
        ("😨", "I'm terrified of what might happen."),
        ("😲", "I can't believe this just happened!"),
    ]

    st.markdown("""
    <div style="font-size:0.75rem; font-weight:600; letter-spacing:1.5px;
                text-transform:uppercase; color:rgba(160,160,210,0.6);
                margin-bottom:0.8rem;">
        ⚡ Quick Examples — Click to Try
    </div>
    """, unsafe_allow_html=True)

    clicked_text = None
    cols = st.columns(len(examples))

    for col, (emoji, text) in zip(cols, examples):
        with col:
            if st.button(f"{emoji}", key=f"ex_{text[:10]}", use_container_width=True,
                         help=text):
                clicked_text = text

    return clicked_text


def main():
    
    load_custom_css()

    
    build_sidebar()

    st.markdown("""
    <div class="hero-section">
        <span class="hero-icon">🧠</span>
        <div class="hero-badge">✦ Powered by Machine Learning & NLP</div>
        <h1 class="hero-title">Emotion AI Classifier</h1>
        <p class="hero-subtitle">
            Detect human emotions using Machine Learning and NLP
        </p>
        <div class="stats-row">
            <div class="stat-chip">Algorithm: <span>Linear SVM</span></div>
            <div class="stat-chip">Features: <span>TF-IDF</span></div>
            <div class="stat-chip">Emotions: <span>6 Classes</span></div>
            <div class="stat-chip">Framework: <span>Scikit-learn</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model, model_err = load_model()
    label_encoder, enc_err = load_label_encoder()

    if model_err:
        st.markdown(f'<div class="custom-error">{model_err}</div>',
                    unsafe_allow_html=True)
    if enc_err:
        st.markdown(f'<div class="custom-error">{enc_err}</div>',
                    unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-label">✍️ Input Text</div>
        """, unsafe_allow_html=True)

        
        example_text = render_examples()

        st.write("")

        default_text = example_text if example_text else ""
        user_text = st.text_area(
            label="hidden",
            value=default_text,
            placeholder=(
                "Type or paste your text here...\n\n"
                "e.g. 'I feel so excited about this new opportunity!'"
            ),
            height=160,
            key="user_input",
            label_visibility="collapsed"
        )

        char_count = len(user_text)
        word_count = len(user_text.split()) if user_text.strip() else 0
        counter_color = (
            "#34d399" if char_count <= 500
            else "#fbbf24" if char_count <= 1000
            else "#f87171"
        )
        st.markdown(f"""
        <div style="text-align:right; font-size:0.72rem;
                    color:{counter_color}; margin-top:0.3rem;">
            {word_count} words · {char_count} characters
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        analyze_clicked = st.button(
            "🔍  Analyze Emotion",
            key="analyze_btn",
            use_container_width=True
        )

        st.markdown("""
        <div class="custom-info">
            💡 <strong>Tip:</strong> For best results, write at least one complete
            sentence expressing a clear feeling or thought.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-label">📊 Analysis Result</div>
        """, unsafe_allow_html=True)

        if analyze_clicked:
            
            if not user_text or not user_text.strip():
                st.markdown("""
                <div class="custom-error">
                    ⚠️ Please enter some text before clicking Analyze.
                </div>
                """, unsafe_allow_html=True)

            elif model is None or label_encoder is None:
                st.markdown("""
                <div class="custom-error">
                    ⚠️ Model or encoder not loaded. Please check your .pkl files.
                </div>
                """, unsafe_allow_html=True)

            else:
                # ── Run Prediction ──
                with st.spinner("Analyzing emotion..."):
                    time.sleep(0.6)   # small UX delay for realism
                    try:
                        result = predict_emotion(
                            user_text.strip(), model, label_encoder
                        )
                        render_result(result)
                    except Exception as e:
                        st.markdown(f"""
                        <div class="custom-error">
                            ❌ Prediction failed: {str(e)}<br>
                            Please ensure your model pipeline is compatible.
                        </div>
                        """, unsafe_allow_html=True)
        else:
            
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem;
                        color:rgba(160,160,210,0.5);">
                <div style="font-size:4rem; margin-bottom:1rem; opacity:0.4;">
                    🎭
                </div>
                <div style="font-size:0.9rem; font-weight:500; margin-bottom:0.5rem;">
                    No Analysis Yet
                </div>
                <div style="font-size:0.78rem;">
                    Enter text and click <strong style="color:#a78bfa">
                    Analyze Emotion</strong> to detect the emotional tone.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section-title" style="text-align:center; font-size:0.8rem;">
        🔄 How It Works
    </div>
    """, unsafe_allow_html=True)

    how_cols = st.columns(4)
    steps = [
        ("1️⃣", "Text Input", "User enters raw text in the input area."),
        ("2️⃣", "TF-IDF", "Text is vectorized using TF-IDF feature extraction."),
        ("3️⃣", "Linear SVM", "SVM classifies the vector using learned hyperplanes."),
        ("4️⃣", "Emotion", "Label decoded and displayed with visual insights."),
    ]
    for col, (num, title, desc) in zip(how_cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:1.2rem 0.8rem;
                        background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px; height:100%;">
                <div style="font-size:2rem; margin-bottom:0.5rem">{num}</div>
                <div style="font-size:0.85rem; font-weight:600;
                            color:#a78bfa; margin-bottom:0.4rem">{title}</div>
                <div style="font-size:0.75rem; color:rgba(160,160,200,0.7);
                            line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-footer">
        Emotion AI Classifier &nbsp;•&nbsp;
        Built by <strong style="color:#a78bfa">Mahadi Ur Rehman Siddiqui</strong>
        &nbsp;•&nbsp; Scikit-learn · TF-IDF · Linear SVM · Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()