import streamlit as st
import pandas as pd
import numpy as np
import cv2
import base64
import html
import time
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="CineMind AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Accept either filename so a differently-named export still works without
# the user having to rename it by hand.
MOVIE_FILE_CANDIDATES = [
    DATA_DIR / "movie_database_final.csv",
    DATA_DIR / "movie_data_final.csv",
]

FACE_MODEL = BASE_DIR / "models" / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_CONFIG = BASE_DIR / "models" / "deploy.prototxt"
EMOTION_MODEL = BASE_DIR / "models" / "emotion-ferplus-8.onnx"

ASSET_DIR = BASE_DIR / "assets"
HERO = ASSET_DIR / "hero.png"
LOGO = ASSET_DIR / "logo.png"
POPCORN = ASSET_DIR / "popcorn.jpg"
REEL = ASSET_DIR / "film reel.jpg"  # optional; falls back gracefully if absent

EMOTIONS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

MOOD_MAP = {
    "happiness": ["Comedy", "Family", "Romance", "Animation"],
    "sadness": ["Drama", "Romance"],
    "anger": ["Action", "Thriller", "Crime"],
    "fear": ["Horror", "Thriller", "Mystery"],
    "surprise": ["Action", "Adventure", "Science Fiction"],
    "disgust": ["Thriller", "Crime", "Drama"],
    "contempt": ["Comedy", "Drama", "Crime"],
    "neutral": ["Drama", "Comedy", "Adventure", "Action"],
}


# =========================================================
# HELPERS
# =========================================================
def render_html(markup):
    """Render a multi-line block of raw HTML via st.markdown.

    Python source indentation ends up baked into triple-quoted strings.
    When a nested line inside one of these blocks has 4+ leading spaces
    (common once you're a couple of <div>s deep, especially after a
    blank line), Markdown's CommonMark parser reads it as an *indented
    code block* instead of raw HTML - so instead of your styled div,
    the user sees the literal HTML tags printed out in a code box.
    Stripping each line's leading whitespace before handing it to
    st.markdown avoids that misparse entirely.
    """
    flat = "\n".join(line.strip() for line in markup.strip("\n").split("\n"))
    st.markdown(flat, unsafe_allow_html=True)


def show_loading_screen(logo_path, seconds=2.2):
    """Show a full-screen splash using the app logo, once per browser
    session. Relies on the .loading-screen / .loader-* CSS classes
    defined in the global stylesheet, so this must run after the CSS
    block has been injected."""
    if st.session_state.get("splash_shown"):
        return

    logo_b64 = img64(logo_path)
    logo_markup = (
        f'<img src="data:image/png;base64,{logo_b64}" class="loader-logo">'
        if logo_b64 else
        '<div class="loader-logo-fallback">🎬</div>'
    )

    placeholder = st.empty()
    with placeholder:
        render_html(f"""
        <div class="loading-screen">
            <div class="loader-content">
                {logo_markup}
                <div class="loader-title">CineMind AI</div>
                <div class="loader-sub">Curating your cinematic experience...</div>
                <div class="loader-bar"><div class="loader-bar-fill"></div></div>
            </div>
        </div>
        """)
    time.sleep(seconds)
    placeholder.empty()
    st.session_state["splash_shown"] = True


def cinematic_loading(message, seconds=1.0, icon="reel"):
    """Show a brief inline cinematic loading moment (spinner + caption)
    while a search/recommendation/detection step "processes". Renders in
    a placeholder that is cleared right before the real result is drawn -
    mirrors the splash-screen pattern but scoped to a single action
    instead of the whole app, and kept short so it never feels like a
    delay was added just for show.
    """
    icon_markup = (
        '<div class="cine-loader-reel"></div>' if icon == "reel"
        else '<div class="cine-loader-scan"><div class="cine-loader-scan-line"></div></div>'
    )
    placeholder = st.empty()
    with placeholder:
        render_html(f"""
        <div class="cine-loader">
            {icon_markup}
            <div class="cine-loader-caption">{html.escape(message)}</div>
        </div>
        """)
    time.sleep(seconds)
    placeholder.empty()


def img64(path):
    if not path.exists():
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return ""


def poster_url(row):
    value = row.get("poster_url", "")
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_existing(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


@st.cache_data
def load_movies():
    movie_file = next((f for f in MOVIE_FILE_CANDIDATES if f.exists()), None)
    if movie_file is None:
        return pd.DataFrame()

    df = pd.read_csv(movie_file)

    # Some exports use slightly different column names (e.g. a cleaned
    # export with "genres_clean"/"cast_clean" instead of "genres"/"cast",
    # or "year" instead of a full "release_date"). Map whichever variant
    # is present onto the names the rest of the app expects.
    column_aliases = {
        "genres": ["genres", "genres_clean"],
        "cast": ["cast", "cast_clean"],
        "crew": ["crew", "crew_clean"],
        "keywords": ["keywords", "keywords_clean"],
        "release_date": ["release_date", "year"],
    }
    for target, candidates in column_aliases.items():
        if target in df.columns:
            continue
        source = first_existing(df, candidates)
        if source:
            df[target] = df[source]

    # Make sure common fields exist.
    for col in ["title", "overview", "genres", "keywords", "cast", "crew",
                "director", "vote_average", "vote_count", "popularity",
                "release_date", "poster_url"]:
        if col not in df.columns:
            df[col] = ""

    numeric_cols = ["vote_average", "vote_count", "popularity"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # release_date may have arrived as a bare year (e.g. from a "year"
    # column, possibly as a float like 2009.0) rather than a full date.
    # Normalize it to a plain year string so downstream code that reads
    # release_date[:4] still shows the right year.
    def to_year_string(x):
        if pd.isna(x) or str(x).strip() == "":
            return ""
        try:
            return str(int(float(x)))
        except ValueError:
            return str(x)

    df["release_date"] = df["release_date"].map(to_year_string)

    text_cols = ["title", "overview", "genres", "keywords", "cast", "crew",
                 "director", "poster_url"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    def clean_text(x):
        return " ".join(str(x).replace("|", " ").replace(",", " ").split())

    df["combined_features"] = (
        df["genres"].map(clean_text) + " " +
        df["keywords"].map(clean_text) + " " +
        df["overview"].map(clean_text) + " " +
        df["cast"].map(clean_text) + " " +
        df["director"].map(clean_text)
    ).str.lower()

    return df


@st.cache_resource
def build_recommender(df):
    if df.empty:
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=12000,
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(df["combined_features"])
    similarity = cosine_similarity(matrix)
    return vectorizer, similarity


def recommend_from_movie(df, similarity, title, n=8):
    if df.empty or similarity is None:
        return pd.DataFrame()

    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title.lower(), na=False)]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    # DataFrame index may not be continuous.
    pos = df.index.get_loc(idx)

    scores = list(enumerate(similarity[pos]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    rows = []
    for i, score in scores[1:]:
        row = df.iloc[i].copy()
        row["similarity_score"] = float(score)
        rows.append(row)
        if len(rows) >= n:
            break

    return pd.DataFrame(rows)


def recommend_from_mood(df, emotion, n=8):
    if df.empty:
        return pd.DataFrame()

    genres = MOOD_MAP.get(emotion, MOOD_MAP["neutral"])

    mask = pd.Series(False, index=df.index)
    for genre in genres:
        mask = mask | df["genres"].str.lower().str.contains(
            genre.lower(), na=False
        )

    result = df[mask].copy()

    if result.empty:
        result = df.copy()

    # Balanced ranking using rating, vote count and popularity.
    result["mood_score"] = (
        result["vote_average"] * 0.65 +
        np.log1p(result["vote_count"]) * 0.12 +
        np.log1p(result["popularity"]) * 0.23
    )
    return result.sort_values("mood_score", ascending=False).head(n)


FACE_CONF_THRESHOLD = 0.5


def missing_emotion_model_files():
    """Return the list of required model files that are not present on
    disk, so the UI can explain exactly what's missing instead of just
    failing silently on every camera capture."""
    return [
        p.name for p in (FACE_CONFIG, FACE_MODEL, EMOTION_MODEL)
        if not p.exists()
    ]


@st.cache_resource(show_spinner=False)
def load_emotion_models():
    """Load the face-detector and FER+ nets once per session instead of
    re-reading them from disk on every camera capture."""
    if not FACE_MODEL.exists() or not FACE_CONFIG.exists() or not EMOTION_MODEL.exists():
        return None, None

    try:
        face_net = cv2.dnn.readNetFromCaffe(str(FACE_CONFIG), str(FACE_MODEL))
        emotion_net = cv2.dnn.readNetFromONNX(str(EMOTION_MODEL))
        return face_net, emotion_net
    except cv2.error:
        return None, None


def detect_emotion(frame):
    face_net, emotion_net = load_emotion_models()
    if face_net is None or emotion_net is None:
        return None, {}

    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        best_conf = 0.0
        best_box = None

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_box = box.astype(int)
                best_conf = confidence

        if best_box is None or best_conf < FACE_CONF_THRESHOLD:
            return None, {}

        x1, y1, x2, y2 = best_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None, {}

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        blob = cv2.dnn.blobFromImage(
            gray,
            1.0,
            (64, 64),
            (0,),
            swapRB=False,
            crop=False
        )
        emotion_net.setInput(blob)
        output = emotion_net.forward().flatten()

        exp = np.exp(output - np.max(output))
        probs = exp / exp.sum()

        scores = {
            emotion: float(prob)
            for emotion, prob in zip(EMOTIONS, probs)
        }
        emotion = EMOTIONS[int(np.argmax(probs))]
        return emotion, scores

    except (cv2.error, ValueError, IndexError):
        return None, {}


# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root {
    --bg: #030712;
    --sidebar: #070d1c;
    --card: #0a1020;
    --card2: #0d1427;
    --pink: #ff4f70;
    --purple: #a855f7;
    --blue: #2563eb;
    --text: #f8fafc;
    --muted: #94a3b8;
    --border: rgba(148,163,184,.13);
}

/* =========================================================
   ANIMATION KEYFRAMES
   ========================================================= */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes logoPulse {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.08); }
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes barFill {
    from { width: 0%; }
    to   { width: 100%; }
}

@keyframes loaderFadeOut {
    to { opacity: 0; visibility: hidden; }
}

/* =========================================================
   LOADING SCREEN
   ========================================================= */
.loading-screen {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 999999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background:
        radial-gradient(circle at 50% 35%, rgba(168,85,247,.14), transparent 55%),
        var(--bg);
    animation: loaderFadeOut .6s ease 2.2s forwards;
}

.loader-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
}

.loader-logo {
    width: 84px;
    height: 84px;
    border-radius: 22px;
    object-fit: contain;
    box-shadow: 0 18px 50px rgba(168,85,247,.4);
    animation: logoZoomIn 1s cubic-bezier(.22,.8,.25,1) both,
               logoPulse 1.2s ease-in-out 1s infinite;
}

.loader-logo-fallback {
    width: 84px;
    height: 84px;
    border-radius: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
    background: linear-gradient(135deg, #ff4f70, #a855f7);
    box-shadow: 0 18px 50px rgba(168,85,247,.4);
    animation: logoZoomIn 1s cubic-bezier(.22,.8,.25,1) both,
               logoPulse 1.2s ease-in-out 1s infinite;
}

@keyframes logoZoomIn {
    0%   { transform: scale(.25); opacity: 0; }
    65%  { transform: scale(1.12); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}

.loader-title {
    font-family: 'Manrope', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #fff;
    animation: titleReveal 1.1s cubic-bezier(.22,.8,.25,1) .35s both;
    text-shadow: 0 0 34px rgba(255,79,112,.45), 0 0 70px rgba(168,85,247,.28);
}

@keyframes titleReveal {
    0%   { transform: scale(.55); opacity: 0; letter-spacing: 10px; }
    100% { transform: scale(1); opacity: 1; letter-spacing: -.5px; }
}

.loader-sub {
    color: #94a3b8;
    font-size: 13px;
    animation: fadeInUp .5s ease .95s both;
}

.loader-bar {
    width: 180px;
    height: 4px;
    border-radius: 4px;
    background: rgba(148,163,184,.15);
    overflow: hidden;
    margin-top: 6px;
}

.loader-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff4f70, #a855f7);
    animation: barFill 2s ease 1.1s forwards;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 75% 10%, rgba(168,85,247,.07), transparent 28%),
        radial-gradient(circle at 30% 70%, rgba(37,99,235,.05), transparent 28%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid rgba(148,163,184,.08);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1500px;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 8px 30px 8px;
}

.brand-icon {
    width: 43px;
    height: 43px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #ff4f70, #a855f7);
    box-shadow: 0 8px 28px rgba(168,85,247,.24);
    font-size: 22px;
}

.brand-name {
    color: #fff;
    font-size: 20px;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -.5px;
}

.brand-sub {
    color: #64748b;
    font-size: 10px;
    margin-top: 4px;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}

.nav-title {
    color: #475569;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin: 6px 10px 8px;
}

div[data-testid="stSidebar"] .stRadio > div {
    gap: 6px;
}

div[data-testid="stSidebar"] .stRadio label {
    border-radius: 12px;
    padding: 12px 14px;
    color: #94a3b8;
    font-size: 15.5px;
    font-weight: 600;
    transition: transform .18s ease, background .18s ease, color .18s ease;
}

div[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,.045);
    color: #fff;
    transform: translateX(3px);
}

.sidebar-note {
    position: fixed;
    bottom: 24px;
    left: 26px;
    color: #64748b;
    font-family: cursive;
    font-size: 14px;
}

.top-row {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 15px;
    margin-bottom: 16px;
}

.icon-pill, .user-pill {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(15,23,42,.75);
    border: 1px solid var(--border);
    color: #cbd5e1;
}

.hero {
    position: relative;
    min-height: 425px;
    border-radius: 25px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,.14);
    background: #0b1020;
    box-shadow: 0 25px 70px rgba(0,0,0,.34);
    display: flex;
    align-items: center;
    animation: fadeInUp .6s ease both;
}

.hero-bg {
    position: absolute;
    inset: 0;
    background-size: cover;
    background-position: center;
    opacity: .9;
    animation: kenBurns 22s ease-in-out infinite alternate;
}

.poster-wall {
    position: absolute;
    inset: 0;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 3px;
    opacity: .6;
    filter: saturate(.7) brightness(.72);
    animation: kenBurns 26s ease-in-out infinite alternate;
}

.poster-wall img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

@keyframes kenBurns {
    0%   { transform: scale(1) translate(0, 0); }
    100% { transform: scale(1.09) translate(-1%, -1%); }
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(5,8,22,.98) 0%, rgba(5,8,22,.91) 42%, rgba(5,8,22,.45) 72%, rgba(5,8,22,.58) 100%),
        linear-gradient(0deg, rgba(3,7,18,.45), transparent);
}

.hero-content {
    position: relative;
    z-index: 2;
    padding: 55px;
    width: 59%;
}

.eyebrow {
    color: #fb7185;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .8px;
    margin-bottom: 15px;
    opacity: 0;
    animation: fadeInUp .6s ease .15s both;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(42px, 4.1vw, 66px);
    line-height: 1.02;
    letter-spacing: -2.8px;
    margin: 0 0 22px;
    color: #fff;
    font-weight: 800;
    opacity: 0;
    animation: fadeInUp .65s ease .32s both;
}

.gradient-text {
    background: linear-gradient(90deg, #ff4f70, #c084fc, #ff4f70);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: gradientShift 4s ease-in-out infinite;
}

.hero p {
    color: #a7b1c4;
    font-size: 15px;
    line-height: 1.75;
    max-width: 590px;
    margin-bottom: 26px;
    opacity: 0;
    animation: fadeInUp .65s ease .5s both;
}

.hero-mini {
    color: #64748b;
    font-size: 12px;
    margin-top: 15px;
    opacity: 0;
    animation: fadeInUp .6s ease .68s both;
}

/* Pulls the "Explore Movies / Discover by Mood" button row (rendered
   right after the hero markup) up into the bottom of the hero card, so
   they read as part of the hero rather than a separate block below it. */
.hero-cta-anchor + div[data-testid="stHorizontalBlock"] {
    position: relative;
    z-index: 5;
    margin-top: -86px;
    margin-left: 55px;
    max-width: 560px;
    opacity: 0;
    animation: fadeInUp .6s ease .85s both;
}

.hero-cta-anchor + div[data-testid="stHorizontalBlock"] button {
    border-radius: 999px !important;
    font-weight: 700 !important;
    transition: transform .2s ease, box-shadow .2s ease !important;
}

.hero-cta-anchor + div[data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-of-type(1) button {
    background: linear-gradient(90deg, #ff4f70, #a855f7) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 12px 30px rgba(255,79,112,.35) !important;
}

.hero-cta-anchor + div[data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-of-type(2) button {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.28) !important;
    color: #fff !important;
}

.hero-cta-anchor + div[data-testid="stHorizontalBlock"] button:hover {
    transform: translateY(-2px);
}

@media (max-width: 900px) {
    .hero-cta-anchor + div[data-testid="stHorizontalBlock"] {
        margin-top: 10px;
        margin-left: 0;
        max-width: 100%;
    }
}

.stat-card {
    margin-top: 18px;
    border: 1px solid rgba(148,163,184,.12);
    background: rgba(10,16,32,.78);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 24px 12px;
    animation: fadeInUp .5s ease both;
    transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
}

.stat-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 40px rgba(0,0,0,.35);
    border-color: rgba(168,85,247,.35);
}

.stat {
    text-align: center;
    padding: 4px 18px;
}

.stat + .stat {
    border-left: 1px solid rgba(148,163,184,.1);
}

.stat-number {
    color: #fff;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -.7px;
}

.stat-label {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 5px;
}

.discovery {
    margin-top: 22px;
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 24px;
    min-height: 215px;
    background:
        radial-gradient(circle at 90% 35%, rgba(168,85,247,.12), transparent 28%),
        linear-gradient(135deg, #0b1122, #0a1020);
    padding: 30px;
    position: relative;
    overflow: hidden;
    animation: fadeInUp .5s ease .1s both;
}

.discovery-title {
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    color: #fff;
    margin-bottom: 5px;
}

.discovery-sub {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 20px;
}

.discovery-art {
    position: absolute;
    right: 35px;
    top: 18px;
    width: 230px;
    height: 175px;
    object-fit: cover;
    border-radius: 20px;
    opacity: .52;
    transform: rotate(-4deg);
    filter: saturate(.75);
    animation: floatArt 4.5s ease-in-out infinite;
}

@keyframes floatArt {
    0%, 100% { transform: rotate(-4deg) translateY(0); }
    50%      { transform: rotate(-4deg) translateY(-10px); }
}

.hand-note {
    position: absolute;
    right: 30px;
    bottom: 18px;
    color: #64748b;
    font-family: cursive;
    font-size: 15px;
    transform: rotate(-3deg);
}

.section-title {
    font-family: 'Manrope', sans-serif;
    color: #fff;
    font-size: 25px;
    font-weight: 800;
    margin: 4px 0 3px;
}

.section-sub {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 20px;
}

.movie-card {
    background: #0a1020;
    border: 1px solid rgba(148,163,184,.1);
    border-radius: 17px;
    overflow: hidden;
    height: 100%;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    animation: fadeInUp .45s ease both;
}

.movie-card:hover {
    transform: translateY(-6px);
    border-color: rgba(168,85,247,.4);
    box-shadow: 0 16px 40px rgba(168,85,247,.18);
}

.movie-poster {
    width: 100%;
    height: 260px;
    object-fit: cover;
    display: block;
    background: #111827;
}

.movie-body {
    padding: 14px;
    animation: fadeInUp .45s ease both;
}

.movie-title {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.movie-meta {
    color: #64748b;
    font-size: 11px;
    margin-top: 5px;
}

.movie-rating {
    color: #fbbf24;
    font-size: 12px;
    margin-top: 8px;
}

.feature-box {
    background: #0a1020;
    border: 1px solid rgba(148,163,184,.1);
    border-radius: 20px;
    padding: 23px;
    min-height: 145px;
    transition: border-color .25s ease, transform .25s ease;
    animation: fadeInUp .5s ease both;
}

.feature-box:hover {
    border-color: rgba(168,85,247,.35);
    transform: translateY(-3px);
}

.feature-icon {
    font-size: 25px;
    margin-bottom: 10px;
}

.feature-title {
    color: #fff;
    font-size: 15px;
    font-weight: 700;
}

.feature-text {
    color: #64748b;
    font-size: 12px;
    line-height: 1.55;
    margin-top: 7px;
}

.result-panel {
    background: #0a1020;
    border: 1px solid rgba(148,163,184,.1);
    border-radius: 20px;
    padding: 22px;
    animation: fadeInUp .5s ease both;
}

/* Poster & chart images get a subtle hover zoom wherever they appear */
[data-testid="stImage"] img {
    border-radius: 14px;
    transition: transform .3s ease, filter .3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.035);
    filter: brightness(1.06);
}

div.stButton > button {
    border-radius: 11px;
    min-height: 42px;
    font-weight: 700;
    border: 1px solid rgba(148,163,184,.15);
    background: rgba(15,23,42,.9);
    color: #f8fafc;
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease, color .18s ease;
}

div.stButton > button:hover {
    border-color: #a855f7;
    color: #fff;
    box-shadow: 0 7px 22px rgba(168,85,247,.14);
    transform: translateY(-2px);
}

div.stButton > button:active {
    transform: translateY(0);
}

div.stButton.primary > button {
    background: linear-gradient(90deg, #ff4f70, #a855f7);
    border: none;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: #0a1020 !important;
    color: #fff !important;
    border-color: rgba(148,163,184,.15) !important;
    border-radius: 11px !important;
}

[data-testid="stMetric"] {
    background: #0a1020;
    border: 1px solid rgba(148,163,184,.1);
    padding: 16px;
    border-radius: 15px;
}

@media (max-width: 900px) {
    .block-container { padding: 1rem; }
    .hero-content { width: 100%; padding: 35px; }
    .hero h1 { font-size: 43px; }
    .hero-bg { opacity: .35; }
    .discovery-art { opacity: .2; right: -25px; }
    .hand-note { display: none; }
}

/* =========================================================
   ACTION LOADERS (search / recommend / mood scan)
   ========================================================= */
@keyframes reelSpin {
    to { transform: rotate(360deg); }
}

@keyframes scanMove {
    0%   { top: 6%; opacity: 0; }
    12%  { opacity: 1; }
    88%  { opacity: 1; }
    100% { top: 88%; opacity: 0; }
}

@keyframes captionPulse {
    0%, 100% { opacity: .65; }
    50%      { opacity: 1; }
}

.cine-loader {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 46px 20px;
    animation: fadeIn .3s ease both;
}

.cine-loader-reel {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    border: 3px solid rgba(168,85,247,.18);
    border-top-color: #ff4f70;
    border-right-color: #a855f7;
    animation: reelSpin .9s linear infinite;
}

.cine-loader-scan {
    position: relative;
    width: 64px;
    height: 64px;
    border-radius: 16px;
    border: 2px solid rgba(168,85,247,.35);
    background: radial-gradient(circle, rgba(168,85,247,.12), transparent 70%);
    overflow: hidden;
}

.cine-loader-scan-line {
    position: absolute;
    left: 4%;
    width: 92%;
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, #ff4f70, #a855f7, transparent);
    box-shadow: 0 0 10px rgba(168,85,247,.8);
    animation: scanMove 1.4s ease-in-out infinite;
}

.cine-loader-caption {
    color: #cbd5e1;
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: .2px;
    animation: captionPulse 1.3s ease-in-out infinite;
}

/* Poster placeholder for rows with no poster_url in the dataset */
.poster-placeholder {
    width: 100%;
    height: 260px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
    color: rgba(248,250,252,.4);
    background:
        radial-gradient(circle at 30% 20%, rgba(255,79,112,.16), transparent 55%),
        radial-gradient(circle at 80% 80%, rgba(168,85,247,.16), transparent 55%),
        #0d1427;
    border: 1px solid rgba(148,163,184,.1);
    transition: transform .3s ease, filter .3s ease;
}

.movie-card:hover .poster-placeholder {
    transform: scale(1.02);
    filter: brightness(1.1);
}

/* Active sidebar nav item */
div[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(90deg, rgba(255,79,112,.16), rgba(168,85,247,.16));
    color: #fff;
    box-shadow: inset 0 0 0 1px rgba(168,85,247,.3);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SPLASH / LOADING SCREEN
# =========================================================
show_loading_screen(LOGO)


# =========================================================
# DATA
# =========================================================
df = load_movies()

if df.empty:
    st.error(
        "Movie database not found. Make sure one of these files exists:\n"
        "`data/movie_database_final.csv` or `data/movie_data_final.csv`"
    )
    st.stop()

_, similarity_matrix = build_recommender(df)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    # A button clicked earlier in this same run (e.g. the hero's
    # "Explore Movies" CTA) can't reassign nav_radio directly once the
    # radio widget below has been created - so it stashes the target
    # page in "pending_nav" instead. Applying it here, right before the
    # widget is instantiated, is what makes the redirect take effect.
    if "pending_nav" in st.session_state:
        st.session_state["nav_radio"] = st.session_state.pop("pending_nav")

    logo_b64 = img64(LOGO)

    if logo_b64:
        render_html(f"""
            <div class="brand">
                <img src="data:image/png;base64,{logo_b64}"
                     style="width:43px;height:43px;object-fit:contain;border-radius:12px;">
                <div>
                    <div class="brand-name">CineMind AI</div>
                    <div class="brand-sub">Movie Intelligence</div>
                </div>
            </div>
            """)
    else:
        render_html("""
            <div class="brand">
                <div class="brand-icon">🎬</div>
                <div>
                    <div class="brand-name">CineMind AI</div>
                    <div class="brand-sub">Movie Intelligence</div>
                </div>
            </div>
            """)

    st.markdown('<div class="nav-title">DISCOVER</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "🔎  Movie Search",
            "😊  Mood Recommendations",
            "📊  Analytics",
        ],
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown(
        '<div class="sidebar-note">Good Movies Better Mood ♡</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HOME
# =========================================================
if page == "🏠  Home":

    # Build a poster-wall backdrop from the app's own movie database
    # (same TMDB image links already used on the Search/Recommendations
    # cards) instead of the single static hero image, when posters are
    # available - kept stable per session so it doesn't reshuffle on
    # every click.
    if "hero_poster_wall" not in st.session_state:
        poster_pool = (
            df["poster_url"].dropna()
            .loc[lambda s: s.astype(str).str.startswith("http")]
            .tolist()
        )
        st.session_state["hero_poster_wall"] = (
            random.sample(poster_pool, min(10, len(poster_pool)))
            if poster_pool else []
        )
    wall_urls = st.session_state["hero_poster_wall"]

    if wall_urls:
        wall_imgs = "".join(f'<img src="{html.escape(u)}" loading="lazy">' for u in wall_urls)
        hero_bg_markup = f'<div class="poster-wall">{wall_imgs}</div>'
    else:
        hero_b64 = img64(HERO)
        hero_style = (
            f'background-image:url("data:image/png;base64,{hero_b64}");'
            if hero_b64 else ""
        )
        hero_bg_markup = f'<div class="hero-bg" style="{hero_style}"></div>'

    render_html(f"""
        <div class="top-row">
            <div class="icon-pill">☼</div>
            <div class="user-pill">◉</div>
        </div>

        <div class="hero">
            {hero_bg_markup}
            <div class="hero-overlay"></div>

            <div class="hero-content">
                <div class="eyebrow">WELCOME TO CINEMIND AI</div>
                <h1>Your mood knows<br> <span class="gradient-text">what to watch.</span></h1>
                <p>
                    Discover movies that match your taste, mood and personality.
                    CineMind AI combines machine learning with facial emotion
                    recognition to turn your next movie night into a perfect pick.
                </p>
                <div class="hero-mini">✦ AI-powered &nbsp; • &nbsp; 4,803+ movies &nbsp; • &nbsp; Personalized picks</div>
            </div>
        </div>
        <div class="hero-cta-anchor"></div>
        """)

    hcol1, hcol2, hcol_spacer = st.columns([1.3, 1.5, 3.7])
    with hcol1:
        if st.button("▶  Explore Movies", key="hero_explore_btn", use_container_width=True):
            st.session_state["pending_nav"] = "🔎  Movie Search"
            st.rerun()
    with hcol2:
        if st.button("😊  Discover by Mood", key="hero_mood_btn", use_container_width=True):
            st.session_state["pending_nav"] = "😊  Mood Recommendations"
            st.rerun()

    c1, c2, c3 = st.columns(3)

    with c1:
        render_html("""
            <div class="stat-card" style="animation-delay:.05s;">
                <div class="stat">
                    <div class="stat-number">4,803+</div>
                    <div class="stat-label">Movies in our database</div>
                </div>
            </div>
            """)

    with c2:
        render_html("""
            <div class="stat-card" style="animation-delay:.18s;">
                <div class="stat">
                    <div class="stat-number">Smart Picks</div>
                    <div class="stat-label">AI-powered recommendations</div>
                </div>
            </div>
            """)

    with c3:
        render_html("""
            <div class="stat-card" style="animation-delay:.31s;">
                <div class="stat">
                    <div class="stat-number">Mood-Based</div>
                    <div class="stat-label">Movies for every emotion</div>
                </div>
            </div>
            """)

    popcorn_b64 = img64(POPCORN)
    reel_b64 = img64(REEL)

    art = ""
    if popcorn_b64:
        art = f'<img class="discovery-art" src="data:image/jpeg;base64,{popcorn_b64}">'
    elif reel_b64:
        art = f'<img class="discovery-art" src="data:image/jpeg;base64,{reel_b64}">'

    render_html(f"""
        <div class="discovery">
            <div style="font-size:27px;margin-bottom:8px;">🍿</div>
            <div class="discovery-title">Not sure what to watch?</div>
            <div class="discovery-sub">
                Let CineMind find your next favorite movie.
            </div>
            {art}
            <div class="hand-note">Same you, new favorite movie ♡</div>
        </div>
        """)

    b1, b2, b3 = st.columns([1.1, 1.1, 3.5])

    with b1:
        if st.button("🎬  Find My Movie", use_container_width=True):
            st.session_state["home_action"] = "movie"

    with b2:
        if st.button("😊  What's My Mood?", use_container_width=True):
            st.session_state["home_action"] = "mood"

    action = st.session_state.get("home_action")

    if action == "movie":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Movie Pick</div>', unsafe_allow_html=True)
        selected = st.selectbox(
            "Choose a movie you already like",
            sorted(df["title"].dropna().unique()),
            key="home_movie"
        )

        if st.button("✨ Recommend Similar Movies", use_container_width=True):
            cinematic_loading("Finding your movie...", seconds=0.9)
            recs = recommend_from_movie(df, similarity_matrix, selected, 8)
            if not recs.empty:
                st.markdown(
                    '<div class="section-sub">Because you liked this, you may also like...</div>',
                    unsafe_allow_html=True
                )
                cols = st.columns(4)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % 4]:
                        p = poster_url(row)
                        if p:
                            st.image(p, use_container_width=True)
                        else:
                            render_html('<div class="poster-placeholder">🎬</div>')
                        render_html(f"""
                            <div class="movie-card">
                                <div class="movie-body">
                                    <div class="movie-title">{html.escape(row['title'])}</div>
                                    <div class="movie-rating">★ {row['vote_average']:.1f}</div>
                                </div>
                            </div>
                            """)

    elif action == "mood":
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Go to **Mood Recommendations** from the sidebar to use your camera and detect your current emotion.")


# =========================================================
# MOVIE SEARCH
# =========================================================
elif page == "🔎  Movie Search":
    st.markdown('<div class="section-title">Movie Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Search the CineMind database and explore movie details.</div>',
        unsafe_allow_html=True
    )

    query = st.text_input("Search movie", placeholder="Try: Inception, Avatar, Batman...")

    if query:
        if not st.session_state.get("search_started"):
            cinematic_loading("Searching the CineMind universe...", seconds=0.7)
            st.session_state["search_started"] = True
        results = df[
            df["title"].str.contains(query, case=False, na=False)
        ].sort_values(["vote_average", "popularity"], ascending=False).head(20)

        if results.empty:
            st.warning("No movie found.")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 4]:
                    p = poster_url(row)
                    if p:
                        st.image(p, use_container_width=True)
                    else:
                        render_html('<div class="poster-placeholder">🎬</div>')
                    render_html(f"""
                        <div class="movie-card">
                            <div class="movie-body">
                                <div class="movie-title">{html.escape(row['title'])}</div>
                                <div class="movie-meta">{html.escape(str(row['release_date'])[:4])}</div>
                                <div class="movie-rating">★ {row['vote_average']:.1f}</div>
                            </div>
                        </div>
                        """)

            st.markdown("<br>", unsafe_allow_html=True)
            chosen = st.selectbox("Open movie details", results["title"].tolist())

            movie = results[results["title"] == chosen].iloc[0]

            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            d1, d2 = st.columns([1, 2.5])

            with d1:
                p = poster_url(movie)
                if p:
                    st.image(p, use_container_width=True)
                else:
                    render_html('<div class="poster-placeholder" style="height:340px;font-size:52px;">🎬</div>')

            with d2:
                st.markdown(f"## {movie['title']}")
                st.write(movie["overview"] or "No overview available.")
                m1, m2, m3 = st.columns(3)
                m1.metric("Rating", f"{movie['vote_average']:.1f}")
                m2.metric("Votes", f"{int(movie['vote_count']):,}")
                m3.metric("Popularity", f"{movie['popularity']:.0f}")

                st.write("**Genres:**", movie["genres"] or "Not available")
                st.write("**Director:**", movie["director"] or "Not available")

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.session_state["search_started"] = False


# =========================================================
# MOVIE RECOMMENDATIONS
# =========================================================
elif page == "🎯  Movie Recommendations":
    st.markdown('<div class="section-title">AI Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Choose a movie and let TF-IDF + cosine similarity find movies with similar content.</div>',
        unsafe_allow_html=True
    )

    selected = st.selectbox(
        "Select a movie",
        sorted(df["title"].dropna().unique()),
        key="recommend_movie"
    )

    if st.button("✨ Generate Recommendations", use_container_width=True):
        cinematic_loading("Curating your next watch...", seconds=1.0)
        recs = recommend_from_movie(df, similarity_matrix, selected, 8)

        if recs.empty:
            st.warning("Could not generate recommendations.")
        else:
            st.success(f"Recommendations based on **{selected}**")

            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 4]:
                    p = poster_url(row)
                    if p:
                        st.image(p, use_container_width=True)
                    else:
                        render_html('<div class="poster-placeholder">🎬</div>')
                    render_html(f"""
                        <div class="movie-card">
                            <div class="movie-body">
                                <div class="movie-title">{html.escape(row['title'])}</div>
                                <div class="movie-meta">{html.escape(str(row['genres'])[:45])}</div>
                                <div class="movie-rating">
                                    ★ {row['vote_average']:.1f}
                                    &nbsp; • &nbsp; Match {row['similarity_score']*100:.0f}%
                                </div>
                            </div>
                        </div>
                        """)


# =========================================================
# MOOD RECOMMENDATIONS
# =========================================================
elif page == "😊  Mood Recommendations":
    st.markdown('<div class="section-title">Mood Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Use your webcam to detect an emotion, or choose one manually.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        render_html("""
            <div class="feature-box">
                <div class="feature-icon">📷</div>
                <div class="feature-title">Facial Emotion Recognition</div>
                <div class="feature-text">
                    The FER+ model detects the strongest facial emotion and
                    maps it to suitable movie genres.
                </div>
            </div>
            """)

        missing_files = missing_emotion_model_files()
        if missing_files:
            st.warning(
                "📷 Camera-based mood detection is disabled because these "
                "model files are missing from `models/`: **"
                + ", ".join(missing_files) + "**.\n\n"
                "Run `python models/download_models.py` once, then restart "
                "the app to enable it. In the meantime, use **Choose Your "
                "Mood** on the right - it works without the camera."
            )

        camera = st.camera_input(
            "Take a photo for mood detection", disabled=bool(missing_files)
        )

        if camera is not None:
            bytes_data = camera.getvalue()
            arr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if not st.session_state.get("mood_scanned_key") == bytes_data[:32]:
                cinematic_loading("Reading your mood...", seconds=1.1, icon="scan")
                st.session_state["mood_scanned_key"] = bytes_data[:32]

            emotion, scores = detect_emotion(frame)

            if emotion:
                st.success(f"Detected mood: **{emotion.title()}**")

                top_scores = sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )[:5]

                chart_df = pd.DataFrame(
                    top_scores, columns=["Emotion", "Probability"]
                )
                fig = px.bar(
                    chart_df,
                    x="Probability",
                    y="Emotion",
                    orientation="h",
                    title="Emotion Confidence"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.session_state["detected_emotion"] = emotion
            else:
                st.warning(
                    "No clear face/emotion detected. Try better lighting "
                    "and keep your face visible."
                )

    with right:
        render_html("""
            <div class="feature-box">
                <div class="feature-icon">🎭</div>
                <div class="feature-title">Choose Your Mood</div>
                <div class="feature-text">
                    You can also select an emotion manually if you do not
                    want to use the camera.
                </div>
            </div>
            """)

        selected_emotion = st.selectbox(
            "Emotion",
            [e.title() for e in EMOTIONS],
            key="manual_emotion"
        )

        emotion_value = selected_emotion.lower()

        if st.button("🎬 Find Movies For This Mood", use_container_width=True):
            cinematic_loading("Finding the perfect vibe...", seconds=0.9)
            recs = recommend_from_mood(df, emotion_value, 8)

            st.markdown(
                f"### Movies for a **{selected_emotion}** mood"
            )

            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 4]:
                    p = poster_url(row)
                    if p:
                        st.image(p, use_container_width=True)
                    else:
                        render_html('<div class="poster-placeholder">🎬</div>')
                    render_html(f"""
                        <div class="movie-card">
                            <div class="movie-body">
                                <div class="movie-title">{html.escape(row['title'])}</div>
                                <div class="movie-rating">★ {row['vote_average']:.1f}</div>
                            </div>
                        </div>
                        """)

    detected = st.session_state.get("detected_emotion")
    if detected:
        st.markdown("---")
        st.markdown(
            f"### 🤖 AI pick for your detected mood: **{detected.title()}**"
        )

        recs = recommend_from_mood(df, detected, 4)
        cols = st.columns(4)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i]:
                p = poster_url(row)
                if p:
                    st.image(p, use_container_width=True)
                else:
                    render_html('<div class="poster-placeholder">🎬</div>')
                render_html(f"""
                    <div class="movie-card">
                        <div class="movie-body">
                            <div class="movie-title">{html.escape(row['title'])}</div>
                            <div class="movie-rating">★ {row['vote_average']:.1f}</div>
                        </div>
                    </div>
                    """)


# =========================================================
# ANALYTICS
# =========================================================
elif page == "📊  Analytics":
    st.markdown('<div class="section-title">Movie Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Explore ratings, popularity and genre distribution in the dataset.</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)
    a.metric("Movies", f"{len(df):,}")
    b.metric("Average Rating", f"{df['vote_average'].mean():.2f}")
    c.metric("Highest Rating", f"{df['vote_average'].max():.1f}")
    d.metric("Avg. Popularity", f"{df['popularity'].mean():.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        genre_counts = {}
        for value in df["genres"]:
            for genre in str(value).replace("|", ",").split(","):
                genre = genre.strip()
                if genre:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

        genre_df = (
            pd.DataFrame(
                list(genre_counts.items()),
                columns=["Genre", "Movies"]
            )
            .sort_values("Movies", ascending=False)
            .head(12)
        )

        fig = px.bar(
            genre_df,
            x="Movies",
            y="Genre",
            orientation="h",
            title="Top Genres"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rating_df = df[df["vote_average"] > 0].copy()

        fig = px.histogram(
            rating_df,
            x="vote_average",
            nbins=20,
            title="Rating Distribution"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⭐ Highest Rated Movies")

    top = (
        df[df["vote_count"] >= 100]
        .sort_values("vote_average", ascending=False)
        .head(10)
    )

    st.dataframe(
        top[["title", "vote_average", "vote_count", "popularity", "genres"]],
        use_container_width=True,
        hide_index=True
    )
