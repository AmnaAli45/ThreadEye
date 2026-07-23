"""
app.py
-------
ThreadEye - Fabric Defect Detection
A demo web app: upload a fabric image, the model inspects it for defects,
and results are shown on a styled "inspection light table" interface.

Usage:
    streamlit run app.py
"""

import time
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# SETTINGS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "threadeye_v1.pt"
DEFAULT_CONFIDENCE = 0.10


# PAGE CONFIG

st.set_page_config(
    page_title="ThreadEye — Fabric Inspection",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS — "inspection light table" visual identity

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #1B1F2A;
    --panel: #242938;
    --panel-light: #2E3446;
    --accent: #C98A3E;
    --accent-dim: #8a5f2c;
    --text: #EDEBE4;
    --text-dim: #9A9FAE;
    --defect: #C15B4A;
    --clear: #6FA88A;
    --thread: repeating-linear-gradient(90deg, #3A4055 0px, #3A4055 2px, transparent 2px, transparent 6px);
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Hide default Streamlit chrome for a cleaner look */
#MainMenu, footer, header {visibility: hidden;}

/* ---- Header banner ---- */
.te-header {
    padding: 2.2rem 2.5rem 1.6rem 2.5rem;
    border-bottom: 1px solid #3A4055;
    margin-bottom: 2rem;
}
.te-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.te-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: var(--text);
    letter-spacing: -0.01em;
    margin: 0;
}
.te-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-dim);
    font-size: 1rem;
    margin-top: 0.5rem;
    max-width: 640px;
}

/* ---- Weave divider (signature structural device) ---- */
.te-weave {
    height: 3px;
    background: var(--thread);
    margin: 1.6rem 0;
    opacity: 0.8;
}

/* ---- Panels / cards ---- */
.te-panel {
    background: var(--panel);
    border: 1px solid #363C50;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
}

/* ---- Readout style for confidence numbers ---- */
.te-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-dim);
}
.te-readout-value {
    color: var(--accent);
    font-weight: 500;
}

/* ---- Status badges ---- */
.te-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.25rem 0.7rem;
    border-radius: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.te-badge-defect {
    background: rgba(193, 91, 74, 0.15);
    color: var(--defect);
    border: 1px solid rgba(193, 91, 74, 0.4);
}
.te-badge-clear {
    background: rgba(111, 168, 138, 0.15);
    color: var(--clear);
    border: 1px solid rgba(111, 168, 138, 0.4);
}

/* ---- File uploader restyle ---- */
[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1.5px dashed #454C63;
    border-radius: 6px;
    padding: 1rem;
}
[data-testid="stFileUploader"] label {
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ---- Buttons ---- */
.stButton>button {
    background: var(--accent);
    color: #1B1F2A;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 4px;
    padding: 0.55rem 1.4rem;
    letter-spacing: 0.02em;
}
.stButton>button:hover {
    background: #DDA050;
    color: #1B1F2A;
}

/* ---- Slider ---- */
[data-testid="stSlider"] label {
    color: var(--text-dim) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #171A24;
    border-right: 1px solid #3A4055;
}

/* ---- Footer note ---- */
.te-footer {
    margin-top: 3rem;
    padding: 1.2rem 0;
    border-top: 1px solid #3A4055;
    color: var(--text-dim);
    font-size: 0.82rem;
    line-height: 1.6;
}
.te-footer b { color: var(--text); }
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown("""
<div class="te-header">
<div class="te-eyebrow">Automated Quality Control · Prototype v0.1</div>
<div class="te-title">🧵 ThreadEye</div>
<div class="te-subtitle">
AI-assisted fabric inspection for textile mills. Upload a fabric sample
image below and ThreadEye will scan it for visible defects — broken
yarn, holes, stains, and weaving irregularities.
</div>
</div>
""", unsafe_allow_html=True)


# SIDEBAR — controls + model info

with st.sidebar:
    st.markdown('<div class="te-eyebrow">Inspection Settings</div>', unsafe_allow_html=True)
    confidence = st.slider(
        "Sensitivity (confidence threshold)",
        min_value=0.05, max_value=0.90, value=DEFAULT_CONFIDENCE, step=0.05,
        help="Lower values catch more potential defects but may include false alarms."
    )

    st.markdown('<div class="te-weave"></div>', unsafe_allow_html=True)
    st.markdown('<div class="te-eyebrow">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="te-readout">
Architecture: <span class="te-readout-value">YOLOv8n-seg</span><br>
Trained on: <span class="te-readout-value">AITEX dataset</span><br>
Status: <span class="te-readout-value">Early prototype</span>
</div>
""", unsafe_allow_html=True)


# MODEL LOADING (cached so it only loads once)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return YOLO(str(MODEL_PATH))


model = load_model()

if model is None:
    st.error(f"Model not found at {MODEL_PATH}. Train the model first (see train.py).")
    st.stop()


# MAIN CONTENT — upload + inspect

col_upload, col_result = st.columns([1, 1.3], gap="large")

with col_upload:
    st.markdown('<div class="te-eyebrow">Load Sample</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a fabric image here",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Sample loaded", width="stretch")

with col_result:
    st.markdown('<div class="te-eyebrow">Inspection Result</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
<div class="te-panel">
<span class="te-readout">Awaiting sample — load a fabric image on the left to begin inspection.</span>
</div>
""", unsafe_allow_html=True)
    else:
        with st.spinner("Scanning fabric for defects..."):
            time.sleep(0.4)  # brief pause so the scanning moment registers
            results = model.predict(source=image, conf=confidence, verbose=False)

        result = results[0]
        annotated = result.plot()  # numpy array (BGR) with boxes drawn
        annotated_rgb = annotated[:, :, ::-1]  # convert BGR -> RGB

        st.image(annotated_rgb, caption="Annotated result", width="stretch")

        num_detections = len(result.boxes) if result.boxes is not None else 0

        if num_detections == 0:
            st.markdown("""
<div class="te-panel">
<span class="te-badge te-badge-clear">Clear</span>
<p class="te-readout" style="margin-top:0.8rem;">No defects detected above the current sensitivity threshold.</p>
</div>
""", unsafe_allow_html=True)
        else:
            # IMPORTANT: build each row as a single unindented line.
            # Multi-line / indented HTML inside st.markdown() gets treated
            # as a Markdown code block instead of being rendered as HTML.
            rows = ""
            for i, box in enumerate(result.boxes):
                conf_val = float(box.conf[0])
                rows += (
                    '<div style="display:flex; justify-content:space-between; '
                    'padding:0.4rem 0; border-bottom:1px solid #363C50;">'
                    f'<span class="te-readout">Defect {i+1}</span>'
                    f'<span class="te-readout-value">{conf_val*100:.1f}% confidence</span>'
                    '</div>'
                )

            panel_html = (
                '<div class="te-panel">'
                f'<span class="te-badge te-badge-defect">{num_detections} Defect(s) Found</span>'
                f'<div style="margin-top:0.9rem;">{rows}</div>'
                '</div>'
            )
            st.markdown(panel_html, unsafe_allow_html=True)


# FOOTER — honest disclaimer

st.markdown("""
<div class="te-weave"></div>
<div class="te-footer">
<b>About this prototype:</b> ThreadEye is trained on a small public research
dataset (AITEX, 244 images) as a proof of concept. Detection accuracy is
still limited — some defects may be missed, and occasional false alarms
can occur. Accuracy will improve significantly once the model is
fine-tuned on real fabric samples from your mill.
</div>
""", unsafe_allow_html=True)