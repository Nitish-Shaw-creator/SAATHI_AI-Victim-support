"""
Streamlit App & Assessment Backend Combined — NHAA 14566
--------------------------------------------------------
Combined single-file implementation for the Stress & Vulnerability Assessment Prototype.
UI restyled to match the "SAATHI — first support, with care" design.

Run with:
    streamlit run app.py
"""

import hashlib
import os
import re
import tempfile
import time
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from langdetect import detect
from deep_translator import GoogleTranslator

# Script ranges used to auto-detect which language a transcript actually came back in
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_BENGALI_RE = re.compile(r"[\u0980-\u09FF]")


def speech_audio_to_text(audio_bytes: bytes) -> tuple:
    """Transcribes microphone recordings using English/en-IN recognition only.

    English and Hinglish speech are kept in Latin/English script. Hindi and
    Bengali recognition/conversion is intentionally not attempted here.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        st.error(
            "The `SpeechRecognition` package is required for speak-to-text. "
            "Install it using `pip install SpeechRecognition`."
        )
        return "", ""

    recognizer = sr.Recognizer()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        # Only the final English recognizer is used. This prevents Google
        # Speech Recognition from trying to reinterpret English speech as
        # Hindi/Bengali and returning Devanagari/Bengali text.
        try:
            en_text = recognizer.recognize_google(audio_data, language="en-IN")
            if en_text:
                return en_text, "English/Hinglish"
        except (sr.UnknownValueError, sr.RequestError):
            pass

        st.warning(
            "Sorry, the recording wasn't clear enough to understand. "
            "Please try speaking again."
        )
        return "", ""
    except Exception as e:
        st.error(f"Error processing recording: {e}")
        return "", ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# 1. BACKEND LOGIC: LEXICON & FUNCTIONS  (UNCHANGED)
# =============================================================================

# Trauma / Distress Keyword Lexicon (with English and Hinglish/Romanized terms)
TRAUMA_LEXICON = {
    "suicidal_ideation": {
        "weight": 1.0,
        "keywords": [
            "suicide", "kill myself", "end my life", "want to die", "no reason to live",
            "jaan dena", "mar jana", "marna chahta", "marna chahti", "marne"
        ],
    },
    "violence_threat": {
        "weight": 0.9,
        "keywords": [
            "threat", "threatened", "kill", "murder", "beaten", "assault", "attacked", "gun", "weapon",
            "dhamki", "marne ki dhamki", "hathiyar", "peeta", "maar dalunga", "hamla"
        ],
    },
    "sexual_violence": {
        "weight": 1.0,
        "keywords": [
            "rape", "gang rape", "molest", "sexually assaulted", "balatkar", "chhedchhad"
        ],
    },
    "fear_intimidation": {
        "weight": 0.6,
        "keywords": [
            "scared", "afraid", "terrified", "intimidated", "unsafe", "hiding",
            "darr", "khauf", "asurakshit", "chupna", "chhipe"
        ],
    },
    "social_isolation": {
        "weight": 0.5,
        "keywords": [
            "boycott", "isolated", "excluded", "outcast", "alone", "no one helps me",
            "akela", "akeli", "koi madad nahi", "alag thalag"
        ],
    },
    "displacement_loss": {
        "weight": 0.6,
        "keywords": [
            "displaced", "lost my home", "evicted", "family died", "lost my son", "lost my daughter",
            "ghar chhoot", "ghar chheen"
        ],
    },
    "prolonged_distress": {
        "weight": 0.4,
        "keywords": [
            "years", "still waiting", "no justice", "case pending", "exhausted", "hopeless",
            "thak", "intezaar", "pareshaan", "insaaf nahi", "help me"
        ],
    },
}

# Action Recommendation Mapping
ACTION_MAP = {
    "Low": [
        "Talk to a trusted person and keep a simple record of what happened.",
        "For emotional support, call Tele-MANAS 14416 (24x7).",
        "For SC/ST atrocity-related grievance support, call NHAA 14566 (24x7)."
    ],
    "Moderate": [
        "Speak with a counsellor or trusted person soon; do not handle the situation alone.",
        "For emotional support, call Tele-MANAS 14416 (24x7).",
        "For legal help, call NALSA 15100; for SC/ST atrocity grievances, call NHAA 14566."
    ],
    "High": [
        "Immediate counsellor callback",
        "Legal aid referral",
        "Notify local police liaison"
    ],
    "Critical": [
        "Emergency response dispatch",
        "Witness protection assessment",
        "Immediate medical + psychiatric support",
        "Escalate to District Administration & SC/ST Protection Cell"
    ],
}


def translate_to_english_local(text: str, source_lang: str) -> str:
    """Translates Hindi or Bengali text to English using deep-translator."""
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception as e:
        return text


def to_english(text: str) -> str:
    """Detect language of `text`; translate Hindi or Bengali locally to English if needed."""
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    if lang in ("hi", "bn"):
        return translate_to_english_local(text, lang)

    return text


def text_stress_score(raw_text: str) -> float:
    """Returns a 0-100 stress score based on sentiment and keyword hits."""
    text = to_english(raw_text)
    text_lower = text.lower()
    text_lower = re.sub(r"[^a-z\s]", " ", text_lower)

    # Sentiment component
    polarity = TextBlob(text).sentiment.polarity
    sentiment_component = (1 - polarity) * 25

    # Keyword component
    keyword_component = 0.0
    for category, data in TRAUMA_LEXICON.items():
        hit = any(kw in text_lower for kw in data["keywords"])
        if hit:
            keyword_component += data["weight"] * 10

    keyword_component = min(keyword_component, 50)
    total = sentiment_component + keyword_component
    return round(min(total, 100), 2)


def voice_stress_score(audio_path: str) -> float:
    """Extracts simple prosodic features from an audio file and maps to a score."""
    try:
        import librosa
    except ImportError:
        st.error("The `librosa` package is required for voice scoring. Install it using `pip install librosa`.")
        return 0.0

    try:
        # Loading through librosa is the actual runtime check that the installed
        # librosa/audio backend can read the uploaded recording.
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        if y is None or len(y) == 0 or sr <= 0:
            st.error("The uploaded recording could not be read by librosa.")
            return 0.0

        # Pitch variability
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # Pause ratio
        intervals = librosa.effects.split(y, top_db=25)
        voiced_duration = sum((end - start) for start, end in intervals) / sr
        total_duration = len(y) / sr
        pause_ratio = 1 - (voiced_duration / total_duration) if total_duration > 0 else 0

        pitch_component = min(pitch_std / 50, 1) * 50
        pause_component = min(pause_ratio * 2, 1) * 50

        return round(pitch_component + pause_component, 2)
    except Exception as e:
        st.error(f"Error processing audio file with librosa: {e}")
        return 0.0


def compute_svi(text_score: float, voice_score: float = None) -> float:
    """Combines text and (optional) voice scores into one SVI (0-100)."""
    if voice_score is None:
        return text_score
    return round((0.45 * text_score) + (0.55 * voice_score), 2)


def categorize_risk(svi: float) -> str:
    """Categorizes the SVI score into risk bands."""
    if svi < 25:
        return "Low"
    elif svi < 50:
        return "Moderate"
    elif svi < 75:
        return "High"
    else:
        return "Critical"


def recommend_action(risk_category: str) -> list:
    """Returns list of actions based on risk category."""
    return ACTION_MAP[risk_category]


def assess_complainant(text: str, audio_path: str = None) -> dict:
    """Runs the full pipeline for one complainant interaction and returns a report."""
    t_score = text_stress_score(text)
    v_score = voice_stress_score(audio_path) if audio_path else None

    svi = compute_svi(t_score, v_score)
    risk = categorize_risk(svi)
    actions = recommend_action(risk)

    return {
        "text_score": t_score,
        "voice_score": v_score,
        "SVI": svi,
        "risk_category": risk,
        "recommended_actions": actions,
        "immediate_action_required": risk in ("High", "Critical"),
    }


# =============================================================================
# 2. FRONTEND LOGIC: STREAMLIT UI  (SAATHI design — polished to match UI Idea)
# =============================================================================

st.set_page_config(
    page_title="SAATHI — first support, with care",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

RISK_COLORS = {
    "Low": "#2D6A4F",
    "Moderate": "#B98C2E",
    "High": "#C1652F",
    "Critical": "#A63A3A",
}

# ---------------------------------------------------------------------------
# Design tokens / global CSS — refined to match accessibility & UX requirements
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --teal-dark: #0F3D34;
            --teal: #1A5C4B;
            --teal-mid: #2D6A4F;
            --cream: #F7F5F0;
            --card: #FFFFFF;
            --border: #E5E1D8;
            --muted: #5C6B63;
            --orange: #C1652F;
            --soft-green: #E8F0EB;
            --soft-peach: #FDF0E9;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background-color: var(--cream);
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 820px;
        }

        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }

        h1, h2, h3, h4 { color: var(--teal-dark); font-weight: 700; letter-spacing: -0.02em; }
        p { color: var(--muted); }

        .saathi-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .saathi-logo {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(145deg, #0F3D34, #1A5C4B);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.15rem;
            box-shadow: 0 2px 8px rgba(15, 61, 52, 0.25);
        }
        .saathi-name {
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--teal-dark);
            line-height: 1.15;
            letter-spacing: 0.02em;
        }
        .saathi-tagline {
            font-size: 0.72rem;
            color: var(--muted);
            font-weight: 400;
        }

        .eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            color: var(--orange);
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .eyebrow::before {
            content: "";
            width: 18px;
            height: 2.5px;
            background: var(--orange);
            border-radius: 2px;
        }

        .subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
            margin-bottom: 1.4rem;
        }

        .saathi-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 28px 28px;
            margin-bottom: 1.2rem;
            box-shadow: 0 1px 3px rgba(15, 61, 52, 0.04);
        }

        .feature-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px 20px;
            height: 100%;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        .feature-card:hover {
            box-shadow: 0 6px 20px rgba(15, 61, 52, 0.08);
            transform: translateY(-2px);
        }
        .feature-card b { color: var(--teal-dark); font-size: 0.95rem; }

        .step-tile {
            border-radius: 14px;
            padding: 22px 20px;
            height: 100%;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .step-tile:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(15, 61, 52, 0.12);
        }
        .step-tile-dark {
            background: linear-gradient(160deg, #0F3D34 0%, #1A5C4B 100%);
            color: #F0EDE5;
        }
        .step-tile-light {
            background: var(--card);
            border: 1px solid var(--border);
            color: var(--teal-dark);
        }
        .step-tile .num {
            font-size: 0.72rem;
            opacity: 0.7;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin-bottom: 10px;
        }

        .notice-box {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 0.88rem;
            color: var(--muted);
            margin: 1rem 0 1.4rem 0;
            line-height: 1.5;
        }
        .notice-box.warn {
            background: var(--soft-peach);
            border-color: #E8C4A8;
            color: #7A3E1C;
        }
        .safety-ok {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            background: #EAF3EC;
            border: 1px solid #C5DCC9;
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 0.88rem;
            color: var(--teal-dark);
            margin: 1rem 0;
            line-height: 1.5;
        }

        .mint-panel {
            background: linear-gradient(135deg, #E7EEE9 0%, #DCE8E0 100%);
            border-radius: 16px;
            padding: 24px 22px;
            margin: 1.4rem 0;
        }

        .step-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0;
            margin: 1.2rem 0 2rem 0;
            padding: 0 10px;
        }
        .step-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            min-width: 80px;
        }
        .step-circle {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.82rem;
            font-weight: 600;
            flex-shrink: 0;
            transition: all 0.25s ease;
        }
        .step-active {
            background: var(--teal-dark);
            color: white;
            box-shadow: 0 2px 8px rgba(15, 61, 52, 0.3);
        }
        .step-done {
            background: var(--teal-mid);
            color: white;
        }
        .step-pending {
            background: transparent;
            border: 1.5px solid var(--border);
            color: var(--muted);
        }
        .step-label {
            font-size: 0.78rem;
            color: var(--muted);
            font-weight: 500;
        }
        .step-label-active {
            color: var(--teal-dark);
            font-weight: 600;
        }
        .step-connector {
            flex: 1;
            height: 2px;
            background: var(--border);
            margin: 0 4px;
            margin-bottom: 22px;
            max-width: 60px;
        }
        .step-connector.done {
            background: var(--teal-mid);
        }

        .svi-card {
            background: linear-gradient(165deg, #0F3D34 0%, #1A5C4B 100%);
            border-radius: 20px;
            padding: 28px 24px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 28px rgba(15, 61, 52, 0.22);
            position: relative;
            overflow: hidden;
        }
        .svi-card::before {
            content: "";
            position: absolute;
            top: -40%;
            right: -20%;
            width: 180px;
            height: 180px;
            background: rgba(255,255,255,0.04);
            border-radius: 50%;
        }
        .svi-badge-tag {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 0.68rem;
            letter-spacing: 0.06em;
            font-weight: 600;
            margin-bottom: 16px;
        }
        .svi-number {
            font-size: 3.4rem;
            font-weight: 700;
            line-height: 1;
            letter-spacing: -0.03em;
        }
        .svi-denom {
            font-size: 1rem;
            opacity: 0.55;
            font-weight: 500;
        }
        .svi-risk {
            margin-top: 14px;
            font-size: 1.15rem;
            font-weight: 600;
        }
        .svi-risk-sub {
            font-size: 0.75rem;
            opacity: 0.6;
            margin-top: 2px;
        }

        .gauge-wrap {
            position: relative;
            width: 140px;
            height: 140px;
            margin: 12px auto 8px;
        }
        .gauge-wrap svg {
            transform: rotate(-90deg);
        }
        .gauge-center {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }

        .indicator-row {
            margin-bottom: 14px;
        }
        .indicator-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.84rem;
            color: var(--teal-dark);
            margin-bottom: 5px;
            font-weight: 500;
        }
        .indicator-track {
            background: #EDEAE3;
            border-radius: 8px;
            height: 9px;
            overflow: hidden;
        }
        .indicator-fill {
            height: 9px;
            border-radius: 8px;
            transition: width 0.6s ease;
        }

        .pathway-card {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--card);
            padding: 20px 18px;
            height: 100%;
            margin-bottom: 0.9rem;
            transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
            cursor: default;
        }
        .pathway-card:hover {
            box-shadow: 0 6px 20px rgba(15, 61, 52, 0.08);
            border-color: #C5D5CB;
            transform: translateY(-2px);
        }
        .pathway-card b {
            color: var(--teal-dark);
            font-size: 0.95rem;
        }
        .pathway-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: var(--soft-green);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            margin-bottom: 10px;
        }

        .reviewer-panel {
            background: linear-gradient(135deg, #E8F0EB 0%, #DCE8E0 100%);
            border: 1px solid #C5D5CB;
            border-radius: 16px;
            padding: 24px;
            margin: 1.2rem 0;
        }

        .signal-box {
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 0.88rem;
            background: white;
            text-align: center;
        }
        .signal-box b {
            display: block;
            color: var(--teal-dark);
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 4px;
            opacity: 0.7;
        }

        /* =========================================================================
           BUTTON STYLES (Upgraded for Accessibility, Black Text & Explicit Hover UX)
           ========================================================================= */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            padding: 0.55rem 1.2rem !important;
            transition: all 0.2s ease-in-out !important;
            color: #000000 !important; /* Explicit black text */
            cursor: pointer !important;
        }

        /* Ensure all nested typography inside buttons is black */
        .stButton > button * {
            color: #000000 !important;
        }

        /* Primary Button: Light sage background (#D8E8DD) providing 16.4:1 contrast ratio with black text (WCAG AAA compliant) */
        .stButton > button[kind="primary"] {
            background: #D8E8DD !important;
            color: #000000 !important;
            border: 1.5px solid #1A5C4B !important;
            box-shadow: 0 2px 8px rgba(15, 61, 52, 0.15) !important;
        }

        /* Primary Button Hover State: Clearly defined with darkened sage background, border shift, lift and shadow */
        .stButton > button[kind="primary"]:hover {
            background: #C2DCCB !important; /* 14.2:1 contrast ratio against black */
            color: #000000 !important;
            border-color: #0F3D34 !important;
            box-shadow: 0 4px 16px rgba(15, 61, 52, 0.28) !important;
            transform: translateY(-2px) !important;
        }

        /* Primary Button Active State */
        .stButton > button[kind="primary"]:active {
            background: #AED0B8 !important;
            transform: translateY(0px) !important;
            box-shadow: 0 1px 4px rgba(15, 61, 52, 0.2) !important;
        }

        /* Secondary Button: Clean white background (#FFFFFF) providing 21:1 contrast ratio with black text */
        .stButton > button[kind="secondary"] {
            background: #FFFFFF !important;
            color: #000000 !important;
            border: 1.5px solid #C5D0C9 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }

        /* Secondary Button Hover State: Defined with soft tint, crisp border, lift and shadow */
        .stButton > button[kind="secondary"]:hover {
            background: #F0F4F1 !important; /* 19.5:1 contrast ratio against black */
            color: #000000 !important;
            border-color: #1A5C4B !important;
            box-shadow: 0 4px 14px rgba(15, 61, 52, 0.14) !important;
            transform: translateY(-2px) !important;
        }

        /* Secondary Button Active State */
        .stButton > button[kind="secondary"]:active {
            background: #E2EBE5 !important;
            transform: translateY(0px) !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        }

        /* Accessible focus outline for keyboard navigation */
        .stButton > button:focus-visible {
            outline: 2.5px solid #0F3D34 !important;
            outline-offset: 2px !important;
        }

        /* Loading spinner theme accent */
        .stSpinner > div {
            border-top-color: #1A5C4B !important;
        }

        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1.5px solid var(--border) !important;
            font-size: 0.95rem !important;
            line-height: 1.55 !important;
            padding: 14px !important;
            background: #FAFAF8 !important;
        }
        .stTextArea textarea:focus {
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 3px rgba(26, 92, 75, 0.12) !important;
        }

        .stSelectbox > div > div {
            border-radius: 10px !important;
        }

        .footer-line {
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 2.5rem;
            border-top: 1px solid var(--border);
            padding-top: 14px;
            text-align: center;
            opacity: 0.85;
        }
        .caveat {
            margin-top: 2rem;
            font-size: 0.78rem;
            color: var(--muted);
            border-left: 3px solid var(--border);
            padding-left: 14px;
            line-height: 1.5;
        }

        .meta-row {
            display: flex;
            gap: 18px;
            justify-content: center;
            font-size: 0.78rem;
            color: var(--muted);
            margin-top: 1.2rem;
            flex-wrap: wrap;
        }
        .meta-row span {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .choice-pill {
            background: #E8F0EB;
            border: 1px solid #C5D5CB;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 0.88rem;
            color: var(--teal-dark);
            line-height: 1.45;
        }

        .transcript-box {
            background: #F8F7F3;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 18px;
            font-size: 0.95rem;
            color: var(--teal-dark);
            line-height: 1.6;
            margin: 12px 0;
        }

        .hero-visual {
            background: linear-gradient(160deg, #1A5C4B 0%, #0F3D34 100%);
            border-radius: 20px;
            padding: 32px 28px;
            color: #EFEDE5;
            text-align: center;
            position: relative;
            overflow: hidden;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .hero-visual::before {
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: rgba(255,255,255,0.05);
            top: -60px;
            right: -40px;
        }

        .check-item {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.92rem;
            color: var(--teal-dark);
        }
        .check-item:last-child { border-bottom: none; }
        .check-num {
            font-weight: 700;
            color: var(--teal);
            min-width: 28px;
            font-size: 0.85rem;
        }

        .ai-card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px 20px;
        }
        .ai-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .ai-tag {
            background: #F0F4F1;
            color: var(--teal);
            font-size: 0.7rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 6px;
            letter-spacing: 0.04em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state & Localization (I18N)
# ---------------------------------------------------------------------------
AVAILABLE_LANGUAGES = ["English", "Hindi", "Bengali", "Hinglish"]

SAMPLE_STORIES = {
    "English": (
        "I have been receiving threatening messages from someone I know for the past few weeks. "
        "I am afraid and feel completely unsafe leaving my home, and I feel isolated with no one to help me. "
        "I am exhausted and still waiting for justice, and I need immediate confidential guidance on how to stay safe."
    ),
    "Hindi": (
        "मुझे पिछले कुछ हफ्तों से किसी परिचित व्यक्ति से धमकी भरे संदेश मिल रहे हैं। "
        "मुझे डर लग रहा है और घर से बाहर निकलने में बहुत असुरक्षित महसूस हो रहा है, और मुझे लगता है कि मैं बिल्कुल अकेली हूँ और कोई मदद करने वाला नहीं है। "
        "मैं बहुत थक चुकी हूँ और इंसाफ का इंतजार कर रही हूँ, मुझे सुरक्षित रहने के लिए तुरंत गोपनीय मार्गदर्शन चाहिए।"
    ),
    "Bengali": (
        "গত কয়েক সপ্তাহ ধরে আমি পরিচিত একজনের কাছ থেকে হুমকিমূলক বার্তা পাচ্ছি। "
        "আমি খুব ভয় পাচ্ছি এবং বাড়ি থেকে বের হতে সম্পূর্ণ অনিরাপদ বোধ করছি, কেউ আমার পাশে নেই এবং আমি নিজেকে একা মনে করছি। "
        "আমি অত্যন্ত ক্লান্ত এবং এখনও বিচারের অপেক্ষায় আছি, সুরক্ষিত থাকার জন্য আমার অবিলম্বে গোপনীয় দিকনির্দেশনা প্রয়োজন।"
    ),
    "Hinglish": (
        "Mujhe pichhle kuch hafton se ek jaan-pehchan wale se threatening messages mil rahe hain. "
        "Main bohot darr mein hoon aur ghar se bahar nikalne mein bilkul unsafe feel kar raha hoon, koi meri help nahi kar raha aur main akela mehsoos kar raha hoon. "
        "Main bohot thak chuka hoon aur justice ka wait kar raha hoon, mujhe safe rehne ke liye immediate confidential guidance chahiye."
    ),
}

SAMPLE_TEXT = SAMPLE_STORIES["English"]

I18N = {
    "English": {
        "brand_tagline": "first support, with care",
        "reviewer_view": "Reviewer view",
        "quick_exit": "✕ Quick exit",
        "emergency_notice": "<b>SAATHI is not an emergency service.</b> If you are in immediate danger, contact local emergency services or a trusted person now.",
        "step_consent": "Consent",
        "step_story": "Your story",
        "step_review": "Review",
        "landing_eyebrow": "SAATHI / Safe first contact",
        "landing_title": "Support begins with being heard.",
        "landing_subtitle": "A calm first step for sharing what happened, understanding immediate concerns, and finding the right next support pathway.",
        "start_checkin": "Start a safe check-in  →",
        "explore_demo": "▶  Explore the demo",
        "hero_heading": "A gentle beginning",
        "hero_sub": "There is room for your story. Pause whenever you need to.",
        "feat_private_title": "Private by design",
        "feat_private_desc": "You choose what to share. Nothing is stored beyond this session.",
        "feat_human_title": "Human support",
        "feat_human_desc": "When you request it. AI only surfaces indicators — people decide.",
        "steps_eyebrow": "A clear path, one step at a time",
        "steps_heading": "From first words to a next step.",
        "steps_subtitle": "You stay in control throughout. You can pause, edit, or leave at any point.",
        "step1_tile_title": "Share what happened",
        "step1_tile_desc": "Type or speak in English, Hindi, Bengali, or Hinglish.",
        "step2_tile_title": "Understand the picture",
        "step2_tile_desc": "Review an initial vulnerability screening — not a diagnosis.",
        "step3_tile_title": "Choose support",
        "step3_tile_desc": "Explore options and request authorized human review if needed.",
        "sensitive_eyebrow": "Designed for sensitive moments",
        "sensitive_heading": "A first step should feel clear, not clinical.",
        "sensitive_subtitle": "SAATHI is designed around respect, plain language, and a simple distinction: a screening can surface patterns, but only people can add human context.",
        "guardrail1": "No automatic police or emergency contact",
        "guardrail2": "AI findings are clearly separated from human observations",
        "guardrail3": "Synthetic demo data only in this prototype",
        "reviewer_eyebrow": "For authorized reviewers",
        "reviewer_heading": "Keep the signal. Add the human context.",
        "reviewer_subtitle": "See how safety flags, AI indicators, notes, and status can stay distinct.",
        "open_reviewer": "Open authorized reviewer view  ↗",
        "footer_text": "SAATHI · Initial support prototype · Privacy-first · Not an emergency service",
        "consent_eyebrow": "SAATHI / A safe space to begin",
        "consent_title": "A safe space to begin",
        "consent_subtitle": "You decide what to share. There is no need to tell your whole story today.",
        "consent_step_label": "STEP 1 / CONSENT & PRIVACY",
        "consent_card_heading": "Before you begin",
        "consent_disclaimer": "This is an initial support and vulnerability screening. It is not therapy, medical advice, legal advice, or emergency dispatch.",
        "check1": "You can stop or leave at any time.",
        "check2": "AI may help organize language-based indicators; it does not make a diagnosis.",
        "check3": "A human review request is optional and only available through this workflow.",
        "back_btn": "←  Back",
        "consent_continue": "I understand — continue  →",
        "story_step_label": "STEP 2 / YOUR STORY",
        "story_card_heading": "What would you like us to understand?",
        "story_card_sub": "Share only what feels safe. You can write in English, Hindi, Bengali, or Hinglish.",
        "story_placeholder": "For example: I have been feeling unsafe because…",
        "speak_btn": "🎤  Tap to speak",
        "demo_story_btn": "✍️  Please write something",
        "audio_uploader_label": "Optional: attach a voice recording (wav / mp3 / m4a) for speech-based stress scoring. Please provide a voice recording so the AI can understand you well.",
        "write_prompt_notice": "Please write something in the box below in your own words.",
        "mic_recorder_label": "Tap to record, then speak — your speech will be converted to text.",
        "transcribing_label": "Converting your speech to text...",
        "transcribed_success": "Your recording was added to the text box below. Feel free to edit it.",
        "review_edit_notice": "🔊 You can review and edit your words before continuing.",
        "continue_safety": "Continue to safety check  →",
        "story_empty_warn": "Please share a few words before continuing.",
        "review_step_label": "STEP 3 / REVIEW",
        "review_card_heading": "Review before screening",
        "review_card_sub": "This is the text that will be used for an initial language-based screening.",
        "transcript_label": "TRANSCRIPT / TEXT",
        "edit_story_btn": "✏️  Edit story",
        "safety_sep_notice": "<b>Safety check is separate from the score.</b> Immediate danger language is surfaced independently for human attention.",
        "run_screening_btn": "Run initial screening  →",
        "meta_private": "🔒 Private prototype session",
        "meta_time": "⏱ About 3 minutes",
        "results_eyebrow": "Initial vulnerability screening",
        "results_heading": "Your first read, with care",
        "results_subtitle": "This result is a structured prototype indicator based on the words shared. It is not a clinical diagnosis or professional determination.",
        "danger_detected_title": "Immediate danger language detected.",
        "danger_detected_desc": "If you are in danger right now, contact local emergency services or a trusted person immediately.",
        "safety_ok_title": "No immediate safety flag surfaced.",
        "safety_ok_desc": "This language-based screening did not surface an immediate danger phrase. You know your situation best; seek help if you feel unsafe.",
        "text_signal": "Text signal",
        "voice_signal": "Voice signal",
        "not_provided": "Not provided",
        "ai_detected": "AI DETECTED",
        "indicators_surfaced": "Language indicators surfaced",
        "proto_logic": "prototype logic",
        "ind_emotional_distress": "Emotional distress",
        "ind_fear_anxiety": "Fear / anxiety",
        "ind_intimidation_threat": "Intimidation / threat",
        "ind_social_vulnerability": "Social vulnerability",
        "ind_trauma_indicators": "Trauma indicators",
        "ind_immediate_safety": "Immediate safety",
        "risk_low": "Low",
        "risk_moderate": "Moderate",
        "risk_high": "High",
        "risk_critical": "Critical",
        "risk_sub": "Prototype band · not a diagnosis",
        "proto_caveat": "Results shown here are derived from a deterministic prototype heuristic. A production version would label any model-assisted evidence separately.",
        "rec_steps_heading": "Recommended next steps",
        "start_another": "Start another check-in",
        "choose_pathway": "Choose a next support pathway  →",
        "results_caveat": "Prototype only — the text scoring uses a keyword lexicon + TextBlob sentiment, and the voice scoring uses basic pitch/pause heuristics via librosa. Nothing here is a validated clinical trauma assessment, and no data is stored or transmitted beyond this local session.",
        "pathways_eyebrow": "Support pathways",
        "pathways_heading": "Choose a next support pathway",
        "pathways_subtitle": "These are options, not promises of availability. You can request a human review without selecting a pathway.",
        "choice_stays": "🧭 <b>Choice stays with you.</b> No pathway is selected automatically.",
        "pathway_counsel_title": "Counselling / emotional support",
        "pathway_counsel_desc": 'For emotional support or counselling, call Tele-MANAS 14416 (24x7). Alternate: 1800-89-14416.',
        "pathway_legal_title": "Legal aid information",
        "pathway_legal_desc": 'For legal aid, call NALSA 15100. For SC/ST atrocity grievances, call NHAA 14566 (24x7).',
        "pathway_med_title": "Medical assistance",
        "pathway_med_desc": 'For urgent medical or other emergency assistance, call 112.',
        "pathway_shelter_title": "Shelter / support services",
        "pathway_shelter_desc": 'For immediate danger or emergency protection, call 112. For child-related support, call 1098.',
        "auth_review_label": "AUTHORIZED HUMAN REVIEW",
        "auth_review_title": "Request authorized human review",
        "auth_review_desc": "A reviewer can look at the safety flag, AI-derived indicators, and your words separately. This does not guarantee a real-world response.",
        "request_review_btn": "Request human review  →",
        "review_success": "A request for authorized human review has been logged for this session.",
        "view_case_status": "View case status  ↗",
        "case_status_prefix": "Current status:",
    },
    "Hindi": {
        "brand_tagline": "सुरक्षित पहला सहारा, संवेदनशीलता के साथ",
        "reviewer_view": "समीक्षक दृश्य",
        "quick_exit": "✕ तुरंत बाहर निकलें",
        "emergency_notice": "<b>SAATHI कोई आपातकालीन सेवा नहीं है।</b> यदि आप तत्काल खतरे में हैं, तो अभी स्थानीय आपातकालीन सेवाओं या किसी विश्वसनीय व्यक्ति से संपर्क करें।",
        "step_consent": "सहमति",
        "step_story": "आपकी बात",
        "step_review": "समीक्षा",
        "landing_eyebrow": "SAATHI / सुरक्षित पहला संपर्क",
        "landing_title": "सहारे की शुरुआत आपकी बात सुने जाने से होती है।",
        "landing_subtitle": "जो हुआ उसे बिना किसी झिझक के साझा करने, प्राथमिक चिंताओं को समझने और सही सहायता मार्ग खोजने का एक शांत पहला कदम।",
        "start_checkin": "सुरक्षित चेक-इन शुरू करें  →",
        "explore_demo": "▶  डेमो देखें",
        "hero_heading": "एक सहज और सुरक्षित शुरुआत",
        "hero_sub": "अपनी बात रखने के लिए पूरा समय है। जब भी ज़रूरत हो आप रुक सकते हैं।",
        "feat_private_title": "पूर्णतः गोपनीय",
        "feat_private_desc": "आप चुनते हैं कि क्या साझा करना है। इस सत्र के बाद कुछ भी सुरक्षित नहीं रखा जाता।",
        "feat_human_title": "मानवीय सहारा",
        "feat_human_desc": "जब आप अनुरोध करते हैं। एआई केवल संकेत दिखाता है — अंतिम निर्णय इंसान लेते हैं।",
        "steps_eyebrow": "एक स्पष्ट मार्ग, कदम दर कदम",
        "steps_heading": "शुरुआती शब्दों से अगले कदम तक।",
        "steps_subtitle": "पूरी प्रक्रिया पर आपका नियंत्रण रहता है। आप कभी भी रुक सकते हैं, बदलाव कर सकते हैं या बाहर निकल सकते हैं।",
        "step1_tile_title": "अपनी बात साझा करें",
        "step1_tile_desc": "अंग्रेजी, हिंदी, बंगाली या हिंग्लिश में लिखें या बोलें।",
        "step2_tile_title": "स्थिति को समझें",
        "step2_tile_desc": "प्रारंभिक संवेदनशीलता स्क्रीनिंग की समीक्षा करें — यह कोई निदान नहीं है।",
        "step3_tile_title": "सहायता चुनें",
        "step3_tile_desc": "विकल्प देखें और आवश्यकता होने पर अधिकृत मानव समीक्षा का अनुरोध करें।",
        "sensitive_eyebrow": "संवेदनशील पलों के लिए निर्मित",
        "sensitive_heading": "पहला कदम स्पष्ट होना चाहिए, नैदानिक नहीं।",
        "sensitive_subtitle": "SAATHI सम्मान, सरल भाषा और एक स्पष्ट अंतर पर आधारित है: स्क्रीनिंग केवल पैटर्न दिखा सकती है, लेकिन मानवीय संदर्भ केवल इंसान जोड़ सकते हैं।",
        "guardrail1": "पुलिस या आपातकालीन सेवाओं को कोई स्वचालित संपर्क नहीं",
        "guardrail2": "एआई निष्कर्ष मानवीय टिप्पणियों से पूरी तरह अलग रखे जाते हैं",
        "guardrail3": "इस प्रोटोटाइप में केवल सिंथेटिक डेमो डेटा उपयोग किया जाता है",
        "reviewer_eyebrow": "अधिकृत समीक्षकों के लिए",
        "reviewer_heading": "संकेतों को समझें। मानवीय संदर्भ जोड़ें।",
        "reviewer_subtitle": "देखें कि सुरक्षा संकेत, एआई संकेतक, नोट्स और स्थिति कैसे अलग-अलग रहते हैं।",
        "open_reviewer": "अधिकृत समीक्षक दृश्य खोलें  ↗",
        "footer_text": "SAATHI · प्रारंभिक सहायता प्रोटोटाइप · गोपनीयता-प्रथम · आपातकालीन सेवा नहीं",
        "consent_eyebrow": "SAATHI / शुरुआत के लिए एक सुरक्षित स्थान",
        "consent_title": "शुरुआत के लिए एक सुरक्षित स्थान",
        "consent_subtitle": "आप तय करते हैं कि क्या साझा करना है। आज पूरी बात बताना आवश्यक नहीं है।",
        "consent_step_label": "चरण 1 / सहमति और गोपनीयता",
        "consent_card_heading": "शुरू करने से पहले",
        "consent_disclaimer": "यह एक प्रारंभिक सहायता और संवेदनशीलता स्क्रीनिंग है। यह कोई चिकित्सा, कानूनी सलाह या आपातकालीन सहायता नहीं है।",
        "check1": "आप किसी भी समय रुक सकते हैं या बाहर निकल सकते हैं।",
        "check2": "एआई भाषा-आधारित संकेतों को समझने में मदद करता है; यह कोई डॉक्टरी निदान नहीं करता।",
        "check3": "मानव समीक्षा का अनुरोध वैकल्पिक है और केवल इस प्रक्रिया के माध्यम से उपलब्ध है।",
        "back_btn": "←  पीछे जाएं",
        "consent_continue": "मैं समझता/समझती हूँ — आगे बढ़ें  →",
        "story_step_label": "चरण 2 / आपकी बात",
        "story_card_heading": "आप हमें क्या समझाना चाहते हैं?",
        "story_card_sub": "केवल वही साझा करें जो सुरक्षित लगे। आप अंग्रेजी, हिंदी, बंगाली या हिंग्लिश में लिख सकते हैं।",
        "story_placeholder": "उदाहरण के लिए: मैं असुरक्षित महसूस कर रहा/रही हूँ क्योंकि…",
        "speak_btn": "🎤  बोलने के लिए टैप करें",
        "demo_story_btn": "✍️  कृपया कुछ लिखें",
        "audio_uploader_label": "वैकल्पिक: आवाज-आधारित तनाव मूल्यांकन के लिए वॉयस रिकॉर्डिंग (wav / mp3 / m4a) संलग्न करें। कृपया एक वॉयस रिकॉर्डिंग दें ताकि AI आपको बेहतर ढंग से समझ सके।",
        "review_edit_notice": "🔊 आप आगे बढ़ने से पहले अपने शब्दों की समीक्षा और संपादन कर सकते हैं।",
        "continue_safety": "सुरक्षा जांच के लिए आगे बढ़ें  →",
        "story_empty_warn": "कृपया आगे बढ़ने से पहले कुछ शब्द साझा करें।",
        "review_step_label": "चरण 3 / समीक्षा",
        "review_card_heading": "स्क्रीनिंग से पहले समीक्षा",
        "review_card_sub": "यह वह पाठ है जिसका उपयोग प्रारंभिक भाषा-आधारित स्क्रीनिंग के लिए किया जाएगा।",
        "transcript_label": "प्रलेख / पाठ",
        "edit_story_btn": "✏️  कहानी में बदलाव करें",
        "safety_sep_notice": "<b>सुरक्षा जांच स्कोर से अलग है।</b> तत्काल खतरे की भाषा मानवीय ध्यान के लिए स्वतंत्र रूप से दिखाई जाती है।",
        "run_screening_btn": "प्रारंभिक स्क्रीनिंग चलाएं  →",
        "meta_private": "🔒 निजी प्रोटोटाइप सत्र",
        "meta_time": "⏱ लगभग 3 मिनट",
        "results_eyebrow": "प्रारंभिक संवेदनशीलता स्क्रीनिंग",
        "results_heading": "आपकी स्थिति की प्रारंभिक समझ",
        "results_subtitle": "यह परिणाम साझा किए गए शब्दों पर आधारित एक प्रोटोटाइप संकेतक है। यह कोई नैदानिक या पेशेवर निर्णय नहीं है।",
        "danger_detected_title": "तत्काल खतरे के संकेत पाए गए।",
        "danger_detected_desc": "यदि आप अभी खतरे में हैं, तो तुरंत स्थानीय आपातकालीन सेवाओं या किसी विश्वसनीय व्यक्ति से संपर्क करें।",
        "safety_ok_title": "कोई तत्काल सुरक्षा चेतावनी नहीं मिली।",
        "safety_ok_desc": "इस भाषा स्क्रीनिंग में कोई तत्काल खतरे वाला शब्द नहीं मिला। आप अपनी स्थिति सबसे बेहतर जानते हैं; असुरक्षित लगने पर मदद लें।",
        "text_signal": "पाठ संकेत",
        "voice_signal": "ध्वनि संकेत",
        "not_provided": "उपलब्ध नहीं",
        "ai_detected": "एआई द्वारा पहचाना गया",
        "indicators_surfaced": "उजागर हुए भाषाई संकेतक",
        "proto_logic": "प्रोटोटाइप तर्क",
        "ind_emotional_distress": "भावनात्मक तनाव",
        "ind_fear_anxiety": "डर / चिंता",
        "ind_intimidation_threat": "धमकी / भय",
        "ind_social_vulnerability": "सामाजिक संवेदनशीलता",
        "ind_trauma_indicators": "आघात संकेतक",
        "ind_immediate_safety": "तात्कालिक सुरक्षा",
        "risk_low": "निम्न",
        "risk_moderate": "मध्यम",
        "risk_high": "उच्च",
        "risk_critical": "अत्यधिक गंभीर",
        "risk_sub": "प्रोटोटाइप स्तर · निदान नहीं",
        "proto_caveat": "यहाँ दिखाए गए परिणाम एक प्रोटोटाइप नियम पर आधारित हैं। वास्तविक प्रणाली में मॉडल-सहायक साक्ष्य अलग से चिह्नित किए जाएंगे।",
        "rec_steps_heading": "अनुशंसित अगले कदम",
        "start_another": "नया चेक-इन शुरू करें",
        "choose_pathway": "अगला सहायता मार्ग चुनें  →",
        "results_caveat": "केवल प्रोटोटाइप — टेक्स्ट स्कोरिंग कीवर्ड और सेंटिमेंट का उपयोग करती है, और वॉयस स्कोरिंग बुनियादी पिच/पॉज पर आधारित है। यह कोई चिकित्सीय मूल्यांकन नहीं है और कोई डेटा संग्रहीत नहीं होता है।",
        "pathways_eyebrow": "सहायता मार्ग",
        "pathways_heading": "अगला सहायता मार्ग चुनें",
        "pathways_subtitle": "ये केवल विकल्प हैं, उपलब्धता का आश्वासन नहीं। आप कोई मार्ग चुने बिना भी मानव समीक्षा का अनुरोध कर सकते हैं।",
        "choice_stays": "🧭 <b>निर्णय आपका ही रहेगा।</b> कोई भी मार्ग अपने आप नहीं चुना जाता।",
        "pathway_counsel_title": "परामर्श / भावनात्मक सहायता",
        "pathway_counsel_desc": 'भावनात्मक सहायता या परामर्श के लिए Tele-MANAS 14416 (24x7) पर कॉल करें। वैकल्पिक: 1800-89-14416।',
        "pathway_legal_title": "कानूनी सहायता जानकारी",
        "pathway_legal_desc": 'कानूनी सहायता के लिए NALSA 15100 पर कॉल करें। SC/ST अत्याचार शिकायत के लिए NHAA 14566 (24x7) पर कॉल करें।',
        "pathway_med_title": "चिकित्सा सहायता",
        "pathway_med_desc": 'तत्काल चिकित्सा या अन्य आपातकालीन सहायता के लिए 112 पर कॉल करें।',
        "pathway_shelter_title": "आश्रय / सहायता सेवाएं",
        "pathway_shelter_desc": 'तत्काल खतरे या आपातकालीन सुरक्षा के लिए 112 पर कॉल करें। बच्चों से संबंधित सहायता के लिए 1098 पर कॉल करें।',
        "auth_review_label": "अधिकृत मानव समीक्षा",
        "auth_review_title": "अधिकृत मानव समीक्षा का अनुरोध करें",
        "auth_review_desc": "एक समीक्षक सुरक्षा चेतावनी, एआई-व्युत्पन्न संकेतकों और आपके शब्दों को अलग-अलग देख सकता है। यह किसी वास्तविक त्वरित कार्रवाई की गारंटी नहीं देता।",
        "request_review_btn": "मानव समीक्षा का अनुरोध करें  →",
        "review_success": "इस सत्र के लिए अधिकृत मानव समीक्षा का अनुरोध दर्ज कर लिया गया है।",
        "view_case_status": "केस की स्थिति देखें  ↗",
        "case_status_prefix": "वर्तमान स्थिति:",
    },
    "Bengali": {
        "brand_tagline": "প্রাথমিক সহায়তা, সতর্কতার সাথে",
        "reviewer_view": "পর্যালোচক ভিউ",
        "quick_exit": "✕ দ্রুত প্রস্থান",
        "emergency_notice": "<b>SAATHI কোনো জরুরি সেবা নয়।</b> আপনি যদি তাৎক্ষণিক বিপদে থাকেন, অবিলম্বে স্থানীয় জরুরি পরিষেবা বা বিশ্বস্ত ব্যক্তির সাথে যোগাযোগ করুন।",
        "step_consent": "সম্মতি",
        "step_story": "আপনার কথা",
        "step_review": "পর্যালোচনা",
        "landing_eyebrow": "SAATHI / নিরাপদ প্রথম যোগাযোগ",
        "landing_title": "সহায়তা শুরু হয় আপনার কথা শোনার মাধ্যমে।",
        "landing_subtitle": "কী ঘটেছে তা শেয়ার করার, তাৎক্ষণিক উদ্বেগ বোঝার এবং সঠিক সহায়তা পথ খোঁজার একটি শান্ত প্রথম পদক্ষেপ।",
        "start_checkin": "নিরাপদ চেক-ইন শুরু করুন  →",
        "explore_demo": "▶  ডেমো দেখুন",
        "hero_heading": "একটি শান্ত এবং নিরাপদ সূচনা",
        "hero_sub": "আপনার কথা বলার পর্যাপ্ত সুযোগ রয়েছে। প্রয়োজন হলে যেকোনো সময় থামতে পারেন।",
        "feat_private_title": "সম্পূর্ণ ব্যক্তিগত",
        "feat_private_desc": "আপনি যা শেয়ার করতে চান তাই করবেন। এই সেশনের বাইরে কিছুই সংরক্ষণ করা হয় না।",
        "feat_human_title": "মানবিক সহায়তা",
        "feat_human_desc": "যখন আপনি অনুরোধ করেন। এআই শুধুমাত্র নির্দেশক দেখায় — মানুষই সিদ্ধান্ত নেয়।",
        "steps_eyebrow": "একটি স্পষ্ট পথ, ধাপে ধাপে",
        "steps_heading": "প্রথম শব্দ থেকে পরবর্তী পদক্ষেপে।",
        "steps_subtitle": "সম্পূর্ণ নিয়ন্ত্রণে আপনিই থাকবেন। যেকোনো সময় থামাতে, সম্পাদনা করতে বা প্রস্থান করতে পারেন।",
        "step1_tile_title": "কী ঘটেছে তা শেয়ার করুন",
        "step1_tile_desc": "ইংরেজি, হিন্দি, বাংলা বা হিংলিশে টাইপ করুন বা বলুন।",
        "step2_tile_title": "পরিস্থিতি বুঝুন",
        "step2_tile_desc": "প্রাথমিক দুর্বলতা স্ক্রীনিং পর্যালোচনা করুন — এটি কোনো রোগ নির্ণয় নয়।",
        "step3_tile_title": "সহায়তা নির্বাচন করুন",
        "step3_tile_desc": "বিকল্পগুলি অন্বেষণ করুন এবং প্রয়োজনে অনুমোদিত মানবিক পর্যালোচনার অনুরোধ করুন।",
        "sensitive_eyebrow": "সংবেদনশীল মুহূর্তের জন্য তৈরি",
        "sensitive_heading": "প্রথম পদক্ষেপটি স্পষ্ট হওয়া উচিত, যান্ত্রিক নয়।",
        "sensitive_subtitle": "SAATHI সম্মান, সহজ ভাষা এবং একটি স্পষ্ট পার্থক্যের উপর প্রতিষ্ঠিত: স্ক্রিনিং প্যাটার্ন চিহ্নিত করতে পারে, কিন্তু মানবিক দৃষ্টিভঙ্গি কেবল মানুষই দিতে পারে।",
        "guardrail1": "পুলিশ বা জরুরি পরিষেবার সাথে কোনো স্বয়ংক্রিয় যোগাযোগ নেই",
        "guardrail2": "এআই ফলাফলগুলি মানবিক পর্যবেক্ষণ থেকে স্পষ্টভাবে আলাদা",
        "guardrail3": "এই প্রোটোটাইপে শুধুমাত্র সিন্থেটিক ডেমো ডেটা ব্যবহৃত",
        "reviewer_eyebrow": "অনুমোদিত পর্যালোচকদের জন্য",
        "reviewer_heading": "সংকেত রাখুন। মানবিক প্রসঙ্গ যুক্ত করুন।",
        "reviewer_subtitle": "দেখুন কিভাবে সুরক্ষা পতাকা, এআই নির্দেশক, নোট এবং স্থিতি আলাদা রাখা যায়।",
        "open_reviewer": "অনুমোদিত পর্যালোচক ভিউ খুলুন  ↗",
        "footer_text": "SAATHI · প্রাথমিক সহায়তা প্রোটোটাইপ · গোপনীয়তা-প্রথম · কোনো জরুরি সেবা নয়",
        "consent_eyebrow": "SAATHI / শুরুর জন্য একটি নিরাপদ স্থান",
        "consent_title": "শুরুর জন্য একটি নিরাপদ স্থান",
        "consent_subtitle": "আপনি ঠিক করবেন কী শেয়ার করবেন। আজ পুরো ঘটনা জানানোর প্রয়োজন নেই।",
        "consent_step_label": "ধাপ ১ / সম্মতি ও গোপনীয়তা",
        "consent_card_heading": "শুরু করার আগে",
        "consent_disclaimer": "এটি একটি প্রাথমিক সহায়তা ও দুর্বলতা স্ক্রিনিং। এটি কোনো থেরাপি, চিকিৎসা বা আইনি পরামর্শ নয়।",
        "check1": "আপনি যেকোনো সময় থামতে বা প্রস্থান করতে পারেন।",
        "check2": "এআই ভাষা-ভিত্তিক নির্দেশক সাজাতে সাহায্য করতে পারে; এটি কোনো রোগ নির্ণয় করে না।",
        "check3": "মানবিক পর্যালোচনার অনুরোধ ঐচ্ছিক এবং শুধুমাত্র এই কার্যপ্রণালীর মাধ্যমে উপলব্ধ।",
        "back_btn": "←  পিছনে",
        "consent_continue": "আমি বুঝতে পেরেছি — এগিয়ে যান  →",
        "story_step_label": "ধাপ ২ / আপনার কথা",
        "story_card_heading": "আপনি আমাদের কী জানাতে চান?",
        "story_card_sub": "শুধুমাত্র যা নিরাপদ মনে হয় তাই শেয়ার করুন। ইংরেজি, হিন্দি, বাংলা বা হিংলিশে লিখতে পারেন।",
        "story_placeholder": "যেমন: আমি অনিরাপদ বোধ করছি কারণ…",
        "speak_btn": "🎤  কথা বলতে ট্যাপ করুন",
        "demo_story_btn": "✍️  অনুগ্রহ করে কিছু লিখুন",
        "audio_uploader_label": "ঐচ্ছিক: ভয়েস-ভিত্তিক স্ট্রেস স্কোরের জন্য একটি ভয়েস রেকর্ডিং (wav / mp3 / m4a) সংযুক্ত করুন। অনুগ্রহ করে একটি ভয়েস রেকর্ডিং দিন যাতে AI আপনাকে ভালোভাবে বুঝতে পারে।",
        "review_edit_notice": "🔊 এগিয়ে যাওয়ার আগে আপনি আপনার বক্তব্য পর্যালোচনা ও সম্পাদনা করতে পারেন।",
        "continue_safety": "সুরক্ষা পরীক্ষার জন্য এগিয়ে যান  →",
        "story_empty_warn": "অনুগ্রহ করে এগিয়ে যাওয়ার আগে কয়েকটি কথা লিখুন।",
        "review_step_label": "ধাপ ৩ / পর্যালোচনা",
        "review_card_heading": "স্ক্রীনিংয়ের আগে পর্যালোচনা",
        "review_card_sub": "এই লেখাটি প্রাথমিক ভাষা-ভিত্তিক স্ক্রীনিংয়ের জন্য ব্যবহৃত হবে।",
        "transcript_label": "প্রতিলিপি / পাঠ্য",
        "edit_story_btn": "✏️  লেখা সম্পাদনা করুন",
        "safety_sep_notice": "<b>সুরক্ষা পরীক্ষা স্কোরের থেকে আলাদা।</b> তাৎক্ষণিক বিপদের ভাষা মানবিক মনোযোগের জন্য আলাদাভাবে উপস্থাপিত হয়।",
        "run_screening_btn": "প্রাথমিক স্ক্রীনিং শুরু করুন  →",
        "meta_private": "🔒 ব্যক্তিগত প্রোটোটাইপ সেশন",
        "meta_time": "⏱ প্রায় ৩ মিনিট",
        "results_eyebrow": "প্রাথমিক দুর্বলতা স্ক্রীনিং",
        "results_heading": "আপনার পরিস্থিতির যত্নশীল প্রাথমিক মূল্যায়ন",
        "results_subtitle": "এই ফলাফলটি শেয়ার করা শব্দের উপর ভিত্তি করে একটি প্রোটোটাইপ নির্দেশক। এটি কোনো ক্লিনিকাল রোগ নির্ণয় নয়।",
        "danger_detected_title": "তাৎক্ষণিক বিপদের ভাষা শনাক্ত হয়েছে।",
        "danger_detected_desc": "আপনি যদি এই মুহূর্তে বিপদে থাকেন, অবিলম্বে স্থানীয় জরুরি পরিষেবা বা কোনো বিশ্বস্ত ব্যক্তির সাথে যোগাযোগ করুন।",
        "safety_ok_title": "কোনো তাৎক্ষণিক বিপদ সংকেত পাওয়া যায়নি।",
        "safety_ok_desc": "এই ভাষা-ভিত্তিক স্ক্রীনিংয়ে তাৎক্ষণিক বিপদের কোনো বাক্যাংশ পাওয়া যায়নি। আপনি আপনার পরিস্থিতি সবচেয়ে ভালো বোঝেন; প্রয়োজনে সাহায্য নিন।",
        "text_signal": "টেক্সট সংকেত",
        "voice_signal": "ভয়েস সংকেত",
        "not_provided": "প্রদান করা হয়নি",
        "ai_detected": "এআই দ্বারা চিহ্নিত",
        "indicators_surfaced": "প্রকাশিত ভাষা নির্দেশক",
        "proto_logic": "প্রোটোটাইপ যুক্তি",
        "ind_emotional_distress": "মানসিক কষ্ট",
        "ind_fear_anxiety": "ভয় / উদ্বেগ",
        "ind_intimidation_threat": "ভীতি প্রদর্শন / হুমকি",
        "ind_social_vulnerability": "সামাজিক দুর্বলতা",
        "ind_trauma_indicators": "ট্রমা নির্দেশক",
        "ind_immediate_safety": "তাৎক্ষণিক নিরাপত্তা",
        "risk_low": "কম",
        "risk_moderate": "মাঝারি",
        "risk_high": "উচ্চ",
        "risk_critical": "মারাত্মক",
        "risk_sub": "প্রোটোটাইপ স্তর · রোগ নির্ণয় নয়",
        "proto_caveat": "এখানে প্রদর্শিত ফলাফল একটি প্রোটোটাইপ হিউরিস্টিক থেকে প্রাপ্ত। উৎপাদন সংস্করণে মডেল-সহায়তাপ্রাপ্ত প্রমাণ আলাদাভাবে চিহ্নিত করা হবে।",
        "rec_steps_heading": "প্রস্তাবিত পরবর্তী পদক্ষেপ",
        "start_another": "আরেকটি চেক-ইন শুরু করুন",
        "choose_pathway": "পরবর্তী সহায়তা পথ বেছে নিন  →",
        "results_caveat": "শুধুমাত্র প্রোটোটাইপ — টেক্সট স্কোরিং কীওয়ার্ড ও সেন্টিমেন্ট ব্যবহার করে এবং ভয়েস স্কোরিং পিচ/পজের ওপর ভিত্তি করে। এটি ক্লিনিকাল মূল্যায়ন নয় এবং ডেটা সংরক্ষণ করা হয় না।",
        "pathways_eyebrow": "সহায়তার পথ",
        "pathways_heading": "একটি পরবর্তী সহায়তা পথ বেছে নিন",
        "pathways_subtitle": "এগুলি বিকল্প, প্রাপ্যতার প্রতিশ্রুতি নয়। কোনো পথ নির্বাচন না করেও আপনি মানবিক পর্যালোচনার অনুরোধ করতে পারেন।",
        "choice_stays": "🧭 <b>সিদ্ধান্ত আপনার হাতেই থাকে।</b> কোনো পথ স্বয়ংক্রিয়ভাবে নির্বাচিত হয় না।",
        "pathway_counsel_title": "কাউন্সেলিং / মানসিক সহায়তা",
        "pathway_counsel_desc": 'মানসিক সহায়তা বা কাউন্সেলিংয়ের জন্য Tele-MANAS 14416 (24x7)-এ কল করুন। বিকল্প: 1800-89-14416।',
        "pathway_legal_title": "আইনি সহায়তা তথ্য",
        "pathway_legal_desc": 'আইনি সহায়তার জন্য NALSA 15100-এ কল করুন। SC/ST অত্যাচারের অভিযোগের জন্য NHAA 14566 (24x7)-এ কল করুন।',
        "pathway_med_title": "চিকিৎসা সহায়তা",
        "pathway_med_desc": 'জরুরি চিকিৎসা বা অন্য জরুরি সহায়তার জন্য 112-এ কল করুন।',
        "pathway_shelter_title": "আশ্রয় / সহায়তা পরিষেবা",
        "pathway_shelter_desc": 'তাৎক্ষণিক বিপদ বা জরুরি সুরক্ষার জন্য 112-এ কল করুন। শিশুদের সহায়তার জন্য 1098-এ কল করুন।',
        "auth_review_label": "অনুমোদিত মানবিক পর্যালোচনা",
        "auth_review_title": "অনুমোদিত মানবিক পর্যালোচনার অনুরোধ করুন",
        "auth_review_desc": "একজন পর্যালোচক সুরক্ষা ফ্ল্যাগ, এআই নির্দেশক এবং আপনার বক্তব্য আলাদাভাবে দেখতে পারেন। এটি নিশ্চিত প্রত্যুত্তরের নিশ্চয়তা দেয় না।",
        "request_review_btn": "মানবিক পর্যালোচনার অনুরোধ করুন  →",
        "review_success": "এই সেশনের জন্য অনুমোদিত মানবিক পর্যালোচনার একটি অনুরোধ নথিভুক্ত করা হয়েছে।",
        "view_case_status": "মামলার স্থিতি দেখুন  ↗",
        "case_status_prefix": "বর্তমান স্থিতি:",
    },
    "Hinglish": {
        "brand_tagline": "first support, poori care ke saath",
        "reviewer_view": "Reviewer view",
        "quick_exit": "✕ Quick exit",
        "emergency_notice": "<b>SAATHI koi emergency service nahi hai.</b> Agar aap immediate danger mein hain, toh turant local emergency services ya kisi trusted person se contact karein.",
        "step_consent": "Consent",
        "step_story": "Aapki story",
        "step_review": "Review",
        "landing_eyebrow": "SAATHI / Safe first contact",
        "landing_title": "Support shuru hota hai suni jaane wali baat se.",
        "landing_subtitle": "Jo hua use bina darr ke share karne, immediate concerns ko samajhne aur sahi support pathway dhoondhne ka calm first step.",
        "start_checkin": "Safe check-in shuru karein  →",
        "explore_demo": "▶  Demo explore karein",
        "hero_heading": "Ek gentle aur safe shuruwaat",
        "hero_sub": "Aapki baat sunne ke liye poori jagah hai. Jab chahein pause le sakte hain.",
        "feat_private_title": "Private by design",
        "feat_private_desc": "Aap decide karte hain kya share karna hai. Session ke baad kuch store nahi hota.",
        "feat_human_title": "Human support",
        "feat_human_desc": "Jab aap request karein. AI sirf indicators surface karta hai — decisions log lete hain.",
        "steps_eyebrow": "Ek clear path, step by step",
        "steps_heading": "First words se next step tak.",
        "steps_subtitle": "Control poora aapke paas hai. Aap kisi bhi point par pause, edit ya exit kar sakte hain.",
        "step1_tile_title": "Share karein kya hua",
        "step1_tile_desc": "English, Hindi, Bengali ya Hinglish mein type karein ya bolein.",
        "step2_tile_title": "Picture ko samjhein",
        "step2_tile_desc": "Initial vulnerability screening dekhein — yeh koi diagnosis nahi hai.",
        "step3_tile_title": "Support choose karein",
        "step3_tile_desc": "Options explore karein aur zaroorat ho toh authorized human review request karein.",
        "sensitive_eyebrow": "Sensitive moments ke liye designed",
        "sensitive_heading": "Pehla step clear aur respectful hona chahiye.",
        "sensitive_subtitle": "SAATHI respect aur simple language par based hai: screening patterns highlight karti hai, par human context sirf log samajh sakte hain.",
        "guardrail1": "Police ya emergency services ko koi automatic contact nahi",
        "guardrail2": "AI findings human observations se clearly alag rehte hain",
        "guardrail3": "Is prototype mein sirf synthetic demo data use hota hai",
        "reviewer_eyebrow": "Authorized reviewers ke liye",
        "reviewer_heading": "Signals ko track karein. Human context add karein.",
        "reviewer_subtitle": "Dekhein kaise safety flags, AI indicators, notes aur status distinct rehte hain.",
        "open_reviewer": "Authorized reviewer view open karein  ↗",
        "footer_text": "SAATHI · Initial support prototype · Privacy-first · Emergency service nahi hai",
        "consent_eyebrow": "SAATHI / Shuruwaat ke liye safe space",
        "consent_title": "Shuruwaat ke liye ek safe space",
        "consent_subtitle": "Aap decide karte hain kya share karna hai. Aaj poori kahani batana zaroori nahi hai.",
        "consent_step_label": "STEP 1 / CONSENT & PRIVACY",
        "consent_card_heading": "Shuru karne se pehle",
        "consent_disclaimer": "Yeh ek initial support aur vulnerability screening hai. Yeh therapy, medical advice ya legal dispatch nahi hai.",
        "check1": "Aap kisi bhi samay pause kar sakte hain ya exit kar sakte hain.",
        "check2": "AI language indicators ko organize karne mein help karta hai; koi medical diagnosis nahi karta.",
        "check3": "Human review request optional hai aur isi workflow ke through available hai.",
        "back_btn": "←  Back",
        "consent_continue": "Main samajhta/samajhti hoon — aage badhein  →",
        "story_step_label": "STEP 2 / AAPKI STORY",
        "story_card_heading": "Aap humein kya batana chahte hain?",
        "story_card_sub": "Sirf wahi share karein jo safe lage. Aap English, Hindi, Bengali ya Hinglish mein likh sakte hain.",
        "story_placeholder": "Jaise ki: Main unsafe feel kar raha/rahi hoon kyunki…",
        "speak_btn": "🎤  Bolne ke liye tap karein",
        "demo_story_btn": "✍️  Kripya kuch likhein",
        "audio_uploader_label": "Optional: Voice recording attach karein (wav / mp3 / m4a) speech stress score ke liye. Please AI ko aapko behtar samajhne ke liye ek voice recording dein.",
        "review_edit_notice": "🔊 Aap aage badhne se pehle apne words review aur edit kar sakte hain.",
        "continue_safety": "Safety check ke liye continue karein  →",
        "story_empty_warn": "Please continue karne se pehle kuch words share karein.",
        "review_step_label": "STEP 3 / REVIEW",
        "review_card_heading": "Screening se pehle review karein",
        "review_card_sub": "Isi text ke basis par initial language screening ki jayegi.",
        "transcript_label": "TRANSCRIPT / TEXT",
        "edit_story_btn": "✏️  Story edit karein",
        "safety_sep_notice": "<b>Safety check score se alag hai.</b> Immediate danger language human review ke liye independently surface hoti hai.",
        "run_screening_btn": "Initial screening run karein  →",
        "meta_private": "🔒 Private prototype session",
        "meta_time": "⏱ Lagbhag 3 minutes",
        "results_eyebrow": "Initial vulnerability screening",
        "results_heading": "Aapki first read, poori care ke saath",
        "results_subtitle": "Yeh result shared words par based ek structured prototype indicator hai. Yeh koi diagnosis nahi hai.",
        "danger_detected_title": "Immediate danger language detect hui hai.",
        "danger_detected_desc": "Agar aap abhi kisi danger mein hain, toh turant local emergency services ya trusted contact se baat karein.",
        "safety_ok_title": "Koi immediate safety flag detect nahi hua.",
        "safety_ok_desc": "Is screening mein immediate danger phrase nahi mila. Aap apni situation sabse behtar jaante hain; help zaroor lein agar unsafe lage.",
        "text_signal": "Text signal",
        "voice_signal": "Voice signal",
        "not_provided": "Not provided",
        "ai_detected": "AI DETECTED",
        "indicators_surfaced": "Language indicators surfaced",
        "proto_logic": "prototype logic",
        "ind_emotional_distress": "Emotional distress",
        "ind_fear_anxiety": "Fear / anxiety",
        "ind_intimidation_threat": "Threat / intimidation",
        "ind_social_vulnerability": "Social isolation / vulnerability",
        "ind_trauma_indicators": "Trauma indicators",
        "ind_immediate_safety": "Immediate safety",
        "risk_low": "Low",
        "risk_moderate": "Moderate",
        "risk_high": "High",
        "risk_critical": "Critical",
        "risk_sub": "Prototype band · not a diagnosis",
        "proto_caveat": "Yeh results deterministic prototype heuristic par based hain. Production version mein model-assisted evidence separately label hoga.",
        "rec_steps_heading": "Recommended next steps",
        "start_another": "Doosra check-in shuru karein",
        "choose_pathway": "Next support pathway choose karein  →",
        "results_caveat": "Sirf prototype — text scoring keywords aur sentiment analysis use karti hai. Koi bhi data session ke bahar store ya transmit nahi hota.",
        "pathways_eyebrow": "Support pathways",
        "pathways_heading": "Next support pathway choose karein",
        "pathways_subtitle": "Yeh options hain, availability ka promise nahi. Aap bina koi pathway select kiye bhi human review request kar sakte hain.",
        "choice_stays": "🧭 <b>Choice hamesha aapke paas hai.</b> Koi pathway automatically select nahi hota.",
        "pathway_counsel_title": "Counselling / emotional support",
        "pathway_counsel_desc": 'Emotional support ya counselling ke liye Tele-MANAS 14416 (24x7) par call karein. Alternate: 1800-89-14416.',
        "pathway_legal_title": "Legal aid information",
        "pathway_legal_desc": 'Legal aid ke liye NALSA 15100 par call karein. SC/ST atrocity grievance ke liye NHAA 14566 (24x7) par call karein.',
        "pathway_med_title": "Medical assistance",
        "pathway_med_desc": 'Urgent medical ya doosri emergency assistance ke liye 112 par call karein.',
        "pathway_shelter_title": "Shelter / support services",
        "pathway_shelter_desc": 'Immediate danger ya emergency protection ke liye 112 par call karein. Child-related support ke liye 1098 par call karein.',
        "auth_review_label": "AUTHORIZED HUMAN REVIEW",
        "auth_review_title": "Authorized human review request karein",
        "auth_review_desc": "Ek reviewer safety flag, AI indicators aur aapke words ko separately review kar sakta hai.",
        "request_review_btn": "Human review request karein  →",
        "review_success": "Is session ke liye authorized human review request log kar li gayi hai.",
        "view_case_status": "Case status dekhein  ↗",
        "case_status_prefix": "Current status:",
    },
}

ACTION_I18N = {
    "Log complaint": {
        "English": "Log complaint",
        "Hindi": "शिकायत दर्ज करें",
        "Bengali": "অভিযোগ নথিভুক্ত করুন",
        "Hinglish": "Complaint log karein",
    },
    "Send self-help/counselling resources by SMS": {
        "English": "Send self-help/counselling resources by SMS",
        "Hindi": "एसएमएस द्वारा स्वयं-सहायता / परामर्श संसाधन भेजें",
        "Bengali": "এসএমএসের মাধ্যমে স্ব-সহায়তা / কাউন্সেলিং রিসোর্স পাঠান",
        "Hinglish": "SMS dwara self-help / counselling resources bhejein",
    },
    "Assign to counsellor for follow-up call within 24 hrs": {
        "English": "Assign to counsellor for follow-up call within 24 hrs",
        "Hindi": "24 घंटे के भीतर फॉलो-अप कॉल के लिए परामर्शदाता नियुक्त करें",
        "Bengali": "২৪ ঘণ্টার মধ্যে ফলো-আপ কলের জন্য কাউন্সেলর নিযুক্ত করুন",
        "Hinglish": "24 hrs ke andar follow-up call ke liye counsellor assign karein",
    },
    "Flag for district welfare officer review": {
        "English": "Flag for district welfare officer review",
        "Hindi": "जिला कल्याण अधिकारी समीक्षा के लिए चिह्नित करें",
        "Bengali": "জেলা কল্যাণ কর্মকর্তার পর্যালোচনার জন্য চিহ্নিত করুন",
        "Hinglish": "District welfare officer review ke liye flag karein",
    },
    "Immediate counsellor callback": {
        "English": "Immediate counsellor callback",
        "Hindi": "तत्काल परामर्शदाता कॉलबैक",
        "Bengali": "অবিলম্বে কাউন্সেলর কলব্যাক",
        "Hinglish": "Immediate counsellor callback",
    },
    "Legal aid referral": {
        "English": "Legal aid referral",
        "Hindi": "कानूनी सहायता रेफरल",
        "Bengali": "আইনি সহায়তা রেফারেল",
        "Hinglish": "Legal aid referral",
    },
    "Notify local police liaison": {
        "English": "Notify local police liaison",
        "Hindi": "स्थानीय पुलिस संपर्क अधिकारी को सूचित करें",
        "Bengali": "স্থানীয় পুলিশ লিয়াজোঁ কর্মকর্তাকে অবহিত করুন",
        "Hinglish": "Local police liaison ko notify karein",
    },
    "Emergency response dispatch": {
        "English": "Emergency response dispatch",
        "Hindi": "आपातकालीन प्रतिक्रिया प्रेषण",
        "Bengali": "জরুরি প্রতিক্রিয়া দল প্রেরণ",
        "Hinglish": "Emergency response dispatch",
    },
    "Witness protection assessment": {
        "English": "Witness protection assessment",
        "Hindi": "गवाह सुरक्षा मूल्यांकन",
        "Bengali": "সাক্ষী সুরক্ষা মূল্যায়ন",
        "Hinglish": "Witness protection assessment",
    },
    "Immediate medical + psychiatric support": {
        "English": "Immediate medical + psychiatric support",
        "Hindi": "तत्काल चिकित्सा और मनोरोग सहायता",
        "Bengali": "অবিলম্বে চিকিৎসা ও মানসিক সহায়তা",
        "Hinglish": "Immediate medical + psychiatric support",
    },
    "Escalate to District Administration & SC/ST Protection Cell": {
        "English": "Escalate to District Administration & SC/ST Protection Cell",
        "Hindi": "जिला प्रशासन और एससी/एसटी सुरक्षा प्रकोष्ठ को अग्रेषित करें",
        "Bengali": "জেলা প্রশাসন ও এসসি/এসটি সুরক্ষা সেলে প্রেরণ করুন",
        "Hinglish": "District Administration aur SC/ST Protection Cell ko escalate karein",
    },
}


def t(key: str) -> str:
    """Returns the localized string for `key` according to current session language."""
    lang = st.session_state.get("language", "English")
    if lang not in I18N:
        lang = "English"
    return I18N[lang].get(key, I18N["English"].get(key, key))


def translate_action(action: str) -> str:
    """Translates a recommended action string to current session language."""
    lang = st.session_state.get("language", "English")
    if action in ACTION_I18N:
        return ACTION_I18N[action].get(lang, action)
    return action


def get_sample_story() -> str:
    """Returns the synthetic demo story corresponding to current session language."""
    lang = st.session_state.get("language", "English")
    return SAMPLE_STORIES.get(lang, SAMPLE_STORIES["English"])


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_defaults = {
    "step": "landing",
    "language": "English",
    "lang_select": "English",
    "reviewer_view": False,
    "consent_given": False,
    "narrative": "",
    "narrative_input": "",
    "audio_file": None,
    "report": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def go(step: str):
    """
    Updates navigation step and triggers immediate rerun to fix the double-click bug.
    Ensures seamless one-click transitions between views.
    """
    st.session_state.step = step
    st.rerun()


# ---------------------------------------------------------------------------
# Shared UI pieces
# ---------------------------------------------------------------------------
def render_topbar():
    col_brand, col_mid, col_lang, col_exit = st.columns([3.2, 2.2, 1.8, 1.5])

    with col_brand:
        st.markdown(
            f"""
            <div class="saathi-brand">
                <div class="saathi-logo">सा</div>
                <div>
                    <div class="saathi-name">SAATHI</div>
                    <div class="saathi-tagline">{t("brand_tagline")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_mid:
        reviewer_label = t("reviewer_view")
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:10px; font-size:0.82rem; color:var(--muted);">
                {f'<span style="color:var(--teal); font-weight:600;">● {reviewer_label}</span>' if st.session_state.reviewer_view else reviewer_label}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.reviewer_view = st.toggle(
            "Reviewer",
            value=st.session_state.reviewer_view,
            label_visibility="collapsed",
            key="reviewer_toggle",
        )

    with col_lang:
        current_lang = st.session_state.get("language", "English")
        if current_lang not in AVAILABLE_LANGUAGES:
            current_lang = "English"
            st.session_state.language = "English"

        def on_lang_change():
            new_lang = st.session_state.lang_select
            old_lang = st.session_state.get("language", "English")
            st.session_state.language = new_lang

            # If the user currently has a synthetic demo story loaded, update it to the new language's story
            curr_text = st.session_state.get("narrative_input", "").strip()
            sample_vals = [s.strip() for s in SAMPLE_STORIES.values()]
            if curr_text in sample_vals or not curr_text:
                if curr_text in sample_vals:
                    new_story = SAMPLE_STORIES.get(new_lang, SAMPLE_STORIES["English"])
                    st.session_state.narrative_input = new_story
                    st.session_state.narrative = new_story

        st.selectbox(
            "Language",
            AVAILABLE_LANGUAGES,
            index=AVAILABLE_LANGUAGES.index(current_lang),
            label_visibility="collapsed",
            key="lang_select",
            on_change=on_lang_change,
        )

    with col_exit:
        if st.button(t("quick_exit"), use_container_width=True, key="quick_exit"):
            st.session_state.step = "landing"
            st.session_state.narrative = ""
            st.session_state.narrative_input = ""
            st.session_state.report = None
            st.session_state.consent_given = False
            st.rerun()

    st.markdown(
        '<div style="height:1px; background:var(--border); margin:4px 0 18px 0;"></div>',
        unsafe_allow_html=True,
    )


def render_steps(current):
    steps = [("consent", t("step_consent")), ("story", t("step_story")), ("review", t("step_review"))]
    idx_map = {s[0]: i for i, s in enumerate(steps)}
    current_idx = idx_map.get(current, 0)

    html = '<div class="step-indicator">'
    for i, (key, label) in enumerate(steps):
        if i < current_idx:
            circle_cls, label_cls = "step-circle step-done", "step-label"
            content = "✓"
        elif i == current_idx:
            circle_cls, label_cls = "step-circle step-active", "step-label step-label-active"
            content = str(i + 1)
        else:
            circle_cls, label_cls = "step-circle step-pending", "step-label"
            content = str(i + 1)

        html += f'<div class="step-item"><div class="{circle_cls}">{content}</div><div class="{label_cls}">{label}</div></div>'
        if i < len(steps) - 1:
            conn_cls = "step-connector done" if i < current_idx else "step-connector"
            html += f'<div class="{conn_cls}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def safety_notice_landing():
    st.markdown(
        f"""
        <div class="notice-box">
            <span style="font-size:1.15rem;">🛡️</span>
            <span>{t("emergency_notice")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _circular_gauge(score: float, risk: str, color: str) -> str:
    """SVG circular progress gauge for SVI."""
    pct = min(max(score / 100, 0), 1)
    radius = 54
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - pct)
    risk_label = t(f"risk_{risk.lower()}")
    risk_sub = t("risk_sub")
    initial_screening = t("results_eyebrow").upper()

    return f"""
    <div class="svi-card">
        <div class="svi-badge-tag">{initial_screening}</div>
        <div class="gauge-wrap">
            <svg width="140" height="140" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="{radius}" fill="none"
                    stroke="rgba(255,255,255,0.12)" stroke-width="10"/>
                <circle cx="70" cy="70" r="{radius}" fill="none"
                    stroke="{color}" stroke-width="10"
                    stroke-linecap="round"
                    stroke-dasharray="{circumference:.1f}"
                    stroke-dashoffset="{offset:.1f}"
                    style="transition: stroke-dashoffset 0.8s ease;"/>
            </svg>
            <div class="gauge-center">
                <div class="svi-number">{int(round(score))}</div>
                <div class="svi-denom">/ 100</div>
            </div>
        </div>
        <div class="svi-risk">{risk_label}</div>
        <div class="svi-risk-sub">{risk_sub}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_landing():
    st.markdown(f'<div class="eyebrow">{t("landing_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f"#{t('landing_title')}")
    st.markdown(
        f'<div class="subtitle">{t("landing_subtitle")}</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.4, 1])
    with c1:
        if st.button(t("start_checkin"), type="primary", use_container_width=True, key="start_checkin"):
            go("consent")
    with c2:
        if st.button(t("explore_demo"), use_container_width=True, key="explore_demo"):
            demo_text = get_sample_story()
            st.session_state.narrative = demo_text
            st.session_state.narrative_input = demo_text
            go("consent")

    safety_notice_landing()

    st.markdown(
        f"""
        <div class="hero-visual">
            <div style="font-size:2.4rem; margin-bottom:10px;">🤝</div>
            <div style="font-size:1.15rem; font-weight:600; margin-bottom:6px;">{t("hero_heading")}</div>
            <div style="font-size:0.9rem; opacity:0.8; max-width:320px;">
                {t("hero_sub")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            f"""
            <div class="feature-card">
                <div style="font-size:1.3rem; margin-bottom:6px;">🔒</div>
                <b>{t("feat_private_title")}</b><br>
                <span style="font-size:0.84rem; color:var(--muted);">{t("feat_private_desc")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            f"""
            <div class="feature-card">
                <div style="font-size:1.3rem; margin-bottom:6px;">🤝</div>
                <b>{t("feat_human_title")}</b><br>
                <span style="font-size:0.84rem; color:var(--muted);">{t("feat_human_desc")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{t("steps_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('steps_heading')}")
    st.markdown(
        f'<div class="subtitle">{t("steps_subtitle")}</div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            f"""
            <div class="step-tile step-tile-dark">
                <div class="num">01</div>
                <div style="font-size:1.2rem; margin:6px 0;">💬</div>
                <b>{t("step1_tile_title")}</b><br>
                <span style="font-size:0.82rem; opacity:0.85;">
                    {t("step1_tile_desc")}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f"""
            <div class="step-tile step-tile-light">
                <div class="num">02</div>
                <div style="font-size:1.2rem; margin:6px 0;">📊</div>
                <b>{t("step2_tile_title")}</b><br>
                <span style="font-size:0.82rem; color:var(--muted);">
                    {t("step2_tile_desc")}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            f"""
            <div class="step-tile step-tile-light">
                <div class="num">03</div>
                <div style="font-size:1.2rem; margin:6px 0;">🛤️</div>
                <b>{t("step3_tile_title")}</b><br>
                <span style="font-size:0.82rem; color:var(--muted);">
                    {t("step3_tile_desc")}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">{t("sensitive_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('sensitive_heading')}")
    st.markdown(
        f'<div class="subtitle">{t("sensitive_subtitle")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:10px; margin:1rem 0 1.5rem;">
            <div style="display:flex; gap:10px; align-items:center; font-size:0.92rem; color:var(--teal-dark);">
                <span style="color:#2D6A4F; font-size:1.1rem;">✓</span> {t("guardrail1")}
            </div>
            <div style="display:flex; gap:10px; align-items:center; font-size:0.92rem; color:var(--teal-dark);">
                <span style="color:#2D6A4F; font-size:1.1rem;">✓</span> {t("guardrail2")}
            </div>
            <div style="display:flex; gap:10px; align-items:center; font-size:0.92rem; color:var(--teal-dark);">
                <span style="color:#2D6A4F; font-size:1.1rem;">✓</span> {t("guardrail3")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown(f'<div class="eyebrow">{t("reviewer_eyebrow")}</div>', unsafe_allow_html=True)
        st.markdown(f"### {t('reviewer_heading')}")
        st.markdown(
            f'<div class="subtitle">{t("reviewer_subtitle")}</div>',
            unsafe_allow_html=True,
        )
        if st.button(t("open_reviewer"), key="open_reviewer"):
            st.session_state.reviewer_view = True
            go("results" if st.session_state.report else "landing")

    st.markdown(
        f'<div class="footer-line">{t("footer_text")}</div>',
        unsafe_allow_html=True,
    )


def page_consent():
    st.markdown(f'<div class="eyebrow">{t("consent_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('consent_title')}")
    st.markdown(
        f'<div class="subtitle">{t("consent_subtitle")}</div>',
        unsafe_allow_html=True,
    )
    render_steps("consent")

    st.markdown(
        f"""
        <div class="saathi-card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
                <div style="width:40px; height:40px; border-radius:10px; background:#E8F0EB;
                    display:flex; align-items:center; justify-content:center; font-size:1.2rem;">🔒</div>
                <div>
                    <div style="font-size:0.72rem; letter-spacing:0.08em; color:var(--muted); font-weight:600;">
                        {t("consent_step_label")}
                    </div>
                    <div style="font-size:1.25rem; font-weight:700; color:var(--teal-dark); margin-top:2px;">
                        {t("consent_card_heading")}
                    </div>
                </div>
            </div>
            <p style="color:var(--muted); font-size:0.95rem; line-height:1.55; margin-bottom:18px;">
                {t("consent_disclaimer")}
            </p>
            <div class="check-item">
                <div class="check-num">01</div>
                <div>{t("check1")}</div>
            </div>
            <div class="check-item">
                <div class="check-num">02</div>
                <div>{t("check2")}</div>
            </div>
            <div class="check-item">
                <div class="check-num">03</div>
                <div>{t("check3")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1, 1.4])
    with b1:
        if st.button(t("back_btn"), use_container_width=True, key="consent_back"):
            go("landing")
    with b2:
        if st.button(t("consent_continue"), type="primary", use_container_width=True, key="consent_continue"):
            st.session_state.consent_given = True
            go("story")


def page_story():
    st.markdown(f'<div class="eyebrow">{t("consent_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('consent_title')}")
    st.markdown(
        f'<div class="subtitle">{t("consent_subtitle")}</div>',
        unsafe_allow_html=True,
    )
    render_steps("story")

    st.markdown(
        f"""
        <div class="saathi-card" style="padding-bottom:12px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
                <div style="width:40px; height:40px; border-radius:10px; background:#E8F0EB;
                    display:flex; align-items:center; justify-content:center; font-size:1.2rem;">💬</div>
                <div>
                    <div style="font-size:0.72rem; letter-spacing:0.08em; color:var(--muted); font-weight:600;">
                        {t("story_step_label")}
                    </div>
                    <div style="font-size:1.25rem; font-weight:700; color:var(--teal-dark); margin-top:2px;">
                        {t("story_card_heading")}
                    </div>
                </div>
            </div>
            <p style="color:var(--muted); font-size:0.9rem; margin-bottom:4px;">
                {t("story_card_sub")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Voice-recording mode: if a file is uploaded, the user only needs to
    # provide the recording. The transcript is generated internally and is
    # never placed in a text box for the user to write/edit.
    audio_file = st.file_uploader(
        t("audio_uploader_label"),
        type=["wav", "mp3", "m4a"],
        key="audio_uploader",
    )
    st.session_state.audio_file = audio_file

    if audio_file is not None:
        st.session_state.audio_only_mode = True
        st.session_state.narrative = ""
        st.session_state.narrative_input = ""

        st.markdown(
            '<div class="notice-box"><span style="font-size:1.1rem;">🎙️</span>'
            '<span>Recording received. SAATHI will analyze the voice features and the '
            'spoken content automatically. You do not need to type anything.</span></div>',
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns([1, 1.4])
        with b1:
            if st.button(t("back_btn"), use_container_width=True, key="story_back_audio"):
                go("consent")
        with b2:
            if st.button(t("continue_safety"), type="primary", use_container_width=True, key="story_continue_audio"):
                go("review")
        return

    st.session_state.audio_only_mode = False

    # Existing text / microphone workflow remains unchanged when no uploaded
    # recording is present.
    if "show_text_writer" not in st.session_state:
        st.session_state.show_text_writer = False
    if "transcript_language" not in st.session_state:
        st.session_state.transcript_language = ""

    if "narrative_input" not in st.session_state:
        st.session_state.narrative_input = st.session_state.get("narrative", "")
    elif st.session_state.get("narrative") and not st.session_state.narrative_input:
        st.session_state.narrative_input = st.session_state.narrative

    pending_transcript = st.session_state.pop("pending_transcript", "")
    pending_transcript_lang = st.session_state.pop("pending_transcript_lang", "")
    if pending_transcript:
        existing_text = st.session_state.narrative_input.strip()
        st.session_state.narrative_input = (
            f"{existing_text} {pending_transcript}".strip() if existing_text else pending_transcript
        )
        st.session_state.narrative = st.session_state.narrative_input
        # A spoken recording supplies its own transcript; do not open the
        # manual writing box unless the user explicitly asks to write.
        st.session_state.show_text_writer = False

    def prompt_to_write():
        st.session_state.show_text_writer = True
        st.session_state.scroll_to_story_box = True

    def toggle_mic_recorder():
        st.session_state.show_mic_recorder = not st.session_state.get("show_mic_recorder", False)

    # The text box is hidden until the user explicitly selects "Tap to write".
    # This prevents an unused chatbot/text area from appearing automatically.
    ca, cb = st.columns([1, 1])
    with ca:
        st.button(t("speak_btn"), use_container_width=True, key="speak_btn", on_click=toggle_mic_recorder)
    with cb:
        st.button(
            t("demo_story_btn"),
            on_click=prompt_to_write,
            use_container_width=True,
            key="demo_story",
        )

    if st.session_state.get("show_text_writer", False):
        st.markdown('<div id="story-text-anchor"></div>', unsafe_allow_html=True)
        st.text_area(
            "Your story",
            height=160,
            placeholder=t("story_placeholder"),
            label_visibility="collapsed",
            key="narrative_input",
        )
        st.session_state.narrative = st.session_state.narrative_input

        if st.session_state.get("scroll_to_story_box"):
            st.session_state.scroll_to_story_box = False
            st.info(t("write_prompt_notice"))
            components.html(
                """
                <script>
                    const doc = window.parent.document;
                    const anchor = doc.getElementById('story-text-anchor');
                    if (anchor) { anchor.scrollIntoView({behavior: 'smooth', block: 'center'}); }
                    const box = doc.querySelector('textarea');
                    if (box) { box.focus(); }
                </script>
                """,
                height=0,
            )

    if st.session_state.get("show_mic_recorder", False):
        mic_audio = st.audio_input(t("mic_recorder_label"), key="mic_recorder")
        if mic_audio is not None:
            audio_bytes = mic_audio.getvalue()
            audio_signature = hashlib.md5(audio_bytes).hexdigest()
            if audio_signature != st.session_state.get("last_transcribed_audio_sig"):
                st.session_state.last_transcribed_audio_sig = audio_signature
                with st.spinner(t("transcribing_label")):
                    # Keep the transcript in the language actually detected
                    # from the recording. Do not translate it to Hindi.
                    transcribed_text, detected_lang = speech_audio_to_text(audio_bytes)
                if transcribed_text:
                    st.session_state.pending_transcript = transcribed_text
                    st.session_state.pending_transcript_lang = detected_lang
                    st.session_state.transcript_language = detected_lang
                    st.rerun()

    st.markdown(
        f"""
        <div style="font-size:0.82rem; color:var(--muted); margin:8px 0 16px;">
            {t("review_edit_notice")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1, 1.4])
    with b1:
        if st.button(t("back_btn"), use_container_width=True, key="story_back"):
            go("consent")
    with b2:
        if st.button(t("continue_safety"), type="primary", use_container_width=True, key="story_continue"):
            if not st.session_state.narrative.strip() and not st.session_state.get("show_mic_recorder", False):
                st.warning(t("story_empty_warn"))
            else:
                go("review")


def page_review():
    st.markdown(f'<div class="eyebrow">{t("consent_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('consent_title')}")
    st.markdown(
        f'<div class="subtitle">{t("consent_subtitle")}</div>',
        unsafe_allow_html=True,
    )
    render_steps("review")

    st.markdown(
        f"""
        <div class="saathi-card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
                <div style="width:40px; height:40px; border-radius:10px; background:#E8F0EB;
                    display:flex; align-items:center; justify-content:center; font-size:1.2rem;">📋</div>
                <div>
                    <div style="font-size:0.72rem; letter-spacing:0.08em; color:var(--muted); font-weight:600;">
                        {t("review_step_label")}
                    </div>
                    <div style="font-size:1.25rem; font-weight:700; color:var(--teal-dark); margin-top:2px;">
                        {t("review_card_heading")}
                    </div>
                </div>
            </div>
            <p style="color:var(--muted); font-size:0.9rem; margin-bottom:12px;">
                {t("review_card_sub")}
            </p>
            <div style="font-size:0.72rem; letter-spacing:0.06em; color:var(--muted); font-weight:600; margin-bottom:6px;">
                {t("transcript_label")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("audio_only_mode", False):
        st.markdown(
            '<div class="notice-box"><span style="font-size:1.1rem;">🎙️</span>'
            '<span>Your recording will be analyzed automatically. No written response is required.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="transcript-box">{st.session_state.narrative}</div>',
            unsafe_allow_html=True,
        )

        if st.button(t("edit_story_btn"), key="edit_story_review"):
            st.session_state.narrative_input = st.session_state.narrative
            go("story")

    st.markdown(
        f"""
        <div class="notice-box" style="background:#FDF6F0; border-color:#E8C4A8;">
            <span style="font-size:1.1rem;">🛡️</span>
            <span>{t("safety_sep_notice")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1, 1.4])
    with b1:
        if st.button(t("back_btn"), use_container_width=True, key="review_back"):
            go("story")
    with b2:
        if st.button(t("run_screening_btn"), type="primary", use_container_width=True, key="run_screening"):
            audio_path = None
            audio_file = st.session_state.audio_file
            audio_only = st.session_state.get("audio_only_mode", False)

            if audio_file is not None:
                suffix = "." + audio_file.name.split(".")[-1]
                audio_bytes = audio_file.getvalue()

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    audio_path = tmp.name

                # For uploaded recordings, speech content is transcribed internally
                # so the user never has to type anything. The transcript is used
                # only for scoring/safety detection and is not shown in the UI.
                if audio_only:
                    with st.spinner("⏳ Understanding the recording and calculating the SVI..."):
                        transcribed_text, detected_lang = speech_audio_to_text(audio_bytes)
                    if transcribed_text:
                        st.session_state.narrative = transcribed_text
                    else:
                        st.session_state.narrative = ""

            with st.spinner("⏳ Analyzing language patterns and calculating stress indices..."):
                time.sleep(0.3)
                report = assess_complainant(st.session_state.narrative, audio_path=audio_path)

            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

            st.session_state.report = report
            go("results")

    st.markdown(
        f"""
        <div class="meta-row">
            <span>{t("meta_private")}</span>
            <span>{t("meta_time")}</span>
            <span>🌐 {st.session_state.language}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _indicator_breakdown(narrative: str):
    """Display-only breakdown of lexicon-category hits — does NOT feed into SVI."""
    eng_text = to_english(narrative)
    combined = f"{narrative} {eng_text}".lower()
    combined_clean = re.sub(r"[^a-z\s]", " ", combined)

    def hits(category):
        kws = TRAUMA_LEXICON[category]["keywords"]
        return sum(1 for kw in kws if kw in combined_clean)

    rows = [
        (t("ind_emotional_distress"), hits("prolonged_distress"), len(TRAUMA_LEXICON["prolonged_distress"]["keywords"]), 20, "#2D6A4F"),
        (t("ind_fear_anxiety"), hits("fear_intimidation"), len(TRAUMA_LEXICON["fear_intimidation"]["keywords"]), 20, "#3A7D65"),
        (t("ind_intimidation_threat"), hits("violence_threat"), len(TRAUMA_LEXICON["violence_threat"]["keywords"]), 20, "#C1652F"),
        (t("ind_social_vulnerability"), hits("social_isolation"), len(TRAUMA_LEXICON["social_isolation"]["keywords"]), 15, "#5B8A72"),
        (t("ind_trauma_indicators"), hits("sexual_violence") + hits("suicidal_ideation"),
         len(TRAUMA_LEXICON["sexual_violence"]["keywords"]) + len(TRAUMA_LEXICON["suicidal_ideation"]["keywords"]), 15, "#6B9E82"),
        (t("ind_immediate_safety"), hits("displacement_loss"), len(TRAUMA_LEXICON["displacement_loss"]["keywords"]), 10, "#A63A3A"),
    ]

    out = []
    for label, hit_count, total_kw, max_val, color in rows:
        frac = min(hit_count / max(total_kw, 1), 1.0)
        value = round(max_val * (0.25 + 0.75 * frac)) if hit_count else round(max_val * 0.15)
        out.append((label, min(value, max_val), max_val, color))
    return out


def _immediate_danger_flag(narrative: str) -> bool:
    """Detect explicit immediate self-harm or violence language."""
    if not narrative or not narrative.strip():
        return False

    eng_text = to_english(narrative)
    combined = f"{narrative} {eng_text}".lower()
    combined = re.sub(r"[^a-z0-9\s]", " ", combined)
    combined = re.sub(r"\s+", " ", combined).strip()

    danger_phrases = [
        "suicide",
        "commit suicide",
        "want to commit suicide",
        "i want to commit suicide",
        "commit suicide right now",
        "suicide right now",
        "want to kill myself",
        "i want to kill myself",
        "kill myself right now",
        "going to kill myself",
        "i am going to kill myself",
        "i am suicidal",
        "i feel suicidal",
        "feeling suicidal",
        "suicidal thought",
        "suicidal thoughts",
        "want to die",
        "i want to die",
        "die right now",
        "planning to kill myself",
        "plan to kill myself",
        "plan to commit suicide",
        "end my life",
        "i want to end my life",
        "no reason to live",
        "kill myself",
        "kill me",
        "going to kill someone",
        "going to hurt someone",
        "threatened to kill",
        "threatened me",
        "murder me",
        "weapon",
        "gun",
        "marna chahta",
        "marna chahti",
        "marna hai",
        "jaan dena",
        "jaan deni",
        "mar jana",
    ]

    return any(phrase in combined for phrase in danger_phrases)


def page_results():
    if st.session_state.report is None:
        go("landing")
        return

    report = st.session_state.report
    risk = report["risk_category"]
    color = RISK_COLORS[risk]

    st.markdown(f'<div class="eyebrow">{t("results_eyebrow")}</div>', unsafe_allow_html=True)

    header_col, edit_col = st.columns([4, 1])
    with header_col:
        st.markdown(f"### {t('results_heading')}")
        st.markdown(
            f'<div class="subtitle">{t("results_subtitle")}</div>',
            unsafe_allow_html=True,
        )
    with edit_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button(t("edit_story_btn"), key="edit_from_results"):
            st.session_state.narrative_input = st.session_state.narrative
            go("story")

    left, right = st.columns([1.15, 1.4])

    with left:
        st.markdown(_circular_gauge(report["SVI"], risk, color), unsafe_allow_html=True)

    with right:
        danger = _immediate_danger_flag(st.session_state.narrative)
        high_or_critical = report["risk_category"] in ("High", "Critical")

        if danger:
            st.markdown(
                f"""
                <div class="notice-box warn" style="margin-top:0;">
                    <span style="font-size:1.2rem;">⚠️</span>
                    <span><b>{t("danger_detected_title")}</b><br>
                    {t("danger_detected_desc")}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif high_or_critical:
            st.markdown(
                f"""
                <div class="notice-box warn" style="margin-top:0;">
                    <span style="font-size:1.2rem;">⚠️</span>
                    <span><b>Elevated risk detected.</b><br>
                    The SVI is in the {risk} range. Prompt human review is recommended;
                    this prototype score is not a diagnosis.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="safety-ok" style="margin-top:0;">
                    <span style="font-size:1.2rem;">✅</span>
                    <span><b>{t("safety_ok_title")}</b><br>
                    {t("safety_ok_desc")}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        sig1, sig2 = st.columns(2)
        with sig1:
            st.markdown(
                f'<div class="signal-box"><b>{t("text_signal")}</b>{report["text_score"]} / 100</div>',
                unsafe_allow_html=True,
            )
        with sig2:
            voice_display = f'{report["voice_score"]} / 100' if report["voice_score"] is not None else t("not_provided")
            st.markdown(
                f'<div class="signal-box"><b>{t("voice_signal")}</b>{voice_display}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ai-card">
            <div class="ai-card-header">
                <div>
                    <div style="font-size:0.72rem; letter-spacing:0.08em; color:var(--muted); font-weight:600;">
                        {t("ai_detected")}
                    </div>
                    <div style="font-size:1.05rem; font-weight:700; color:var(--teal-dark);">
                        {t("indicators_surfaced")}
                    </div>
                </div>
                <span class="ai-tag">{t("proto_logic")}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    for label, value, max_val, bar_color in _indicator_breakdown(st.session_state.narrative):
        pct = int(100 * value / max_val)
        st.markdown(
            f"""
            <div class="indicator-row">
                <div class="indicator-label">
                    <span>{label}</span>
                    <span style="font-weight:600;">{value}/{max_val}</span>
                </div>
                <div class="indicator-track">
                    <div class="indicator-fill" style="width:{pct}%; background:{bar_color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="font-size:0.78rem; color:var(--muted); margin-top:12px; line-height:1.45;">
            {t("proto_caveat")}
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="saathi-card">
            <div style="font-size:0.72rem; letter-spacing:0.08em; color:var(--teal); font-weight:600; margin-bottom:6px;">
                {t("rec_steps_heading").upper()}
            </div>
            <div style="font-size:1.05rem; font-weight:700; color:var(--teal-dark); margin-bottom:12px;">
                {t("rec_steps_heading")}
            </div>
        """,
        unsafe_allow_html=True,
    )
    for action in report["recommended_actions"]:
        translated_action = translate_action(action)
        st.markdown(
            f'<div style="padding:6px 0; font-size:0.92rem; color:var(--teal-dark);">→ {translated_action}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1.5])
    with b1:
        if st.button(t("start_another"), use_container_width=True, key="another_checkin"):
            st.session_state.narrative = ""
            st.session_state.narrative_input = ""
            st.session_state.report = None
            st.session_state.audio_only_mode = False
            st.session_state.audio_file = None
            st.session_state.show_text_writer = False
            st.session_state.show_mic_recorder = False
            st.session_state.transcript_language = ""
            go("consent")
    with b2:
        if st.button(t("choose_pathway"), type="primary", use_container_width=True, key="go_pathways"):
            go("pathways")

    st.markdown(
        f'<div class="caveat">{t("results_caveat")}</div>',
        unsafe_allow_html=True,
    )


def page_pathways():
    report = st.session_state.report

    st.markdown(f'<div class="eyebrow">{t("pathways_eyebrow")}</div>', unsafe_allow_html=True)
    st.markdown(f"### {t('pathways_heading')}")
    st.markdown(
        f'<div class="subtitle">{t("pathways_subtitle")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="choice-pill" style="margin-bottom:1.4rem;">
            {t("choice_stays")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    pathways = [
        ("💬", t("pathway_counsel_title"), t("pathway_counsel_desc")),
        ("⚖️", t("pathway_legal_title"), t("pathway_legal_desc")),
        ("🩺", t("pathway_med_title"), t("pathway_med_desc")),
        ("🏠", t("pathway_shelter_title"), t("pathway_shelter_desc")),
    ]

    for i in range(0, len(pathways), 2):
        c1, c2 = st.columns(2)
        for col, (icon, title, desc) in zip((c1, c2), pathways[i:i + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="pathway-card">
                        <div class="pathway-icon">{icon}</div>
                        <b>{title}</b><br>
                        <span style="font-size:0.85rem; color:var(--muted); line-height:1.45;">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button(t("start_another"), use_container_width=True, key="path_another"):
            st.session_state.narrative = ""
            st.session_state.narrative_input = ""
            st.session_state.report = None
            st.session_state.audio_only_mode = False
            st.session_state.audio_file = None
            st.session_state.show_text_writer = False
            st.session_state.show_mic_recorder = False
            st.session_state.transcript_language = ""
            go("consent")
    with b2:
        if st.button(t("view_case_status"), use_container_width=True, key="view_status"):
            if report:
                status_risk = t(f"risk_{report['risk_category'].lower()}")
                st.info(f"{t('case_status_prefix')} **{status_risk}** · SVI {report['SVI']} / 100")
            else:
                st.info("No active case in this session.")

    st.markdown(
        f'<div class="footer-line">{t("footer_text")}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
render_topbar()

_router = {
    "landing": page_landing,
    "consent": page_consent,
    "story": page_story,
    "review": page_review,
    "results": page_results,
    "pathways": page_pathways,
}
_router.get(st.session_state.step, page_landing)()
