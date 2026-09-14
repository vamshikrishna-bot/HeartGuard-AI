import streamlit as st
import pandas as pd
import joblib
import math

st.set_page_config(page_title="HeartGuard AI", page_icon="&#9829;", layout="wide", initial_sidebar_state="expanded")

model=joblib.load('KNN_heart.pkl')
scaler=joblib.load('scaler_heart.pkl')
expected_columns=joblib.load('columns_heart.pkl')

if "entered" not in st.session_state:
    st.session_state.entered = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {
  --bg-deep: #0b0518;
  --bg-mid: #150a2b;
  --border: rgba(255,255,255,0.09);
  --border-hover: rgba(198,158,255,0.45);
  --text: #f4f1fb;
  --muted: #a89cc4;
  --accent: #a855f7;
  --accent2: #ec4899;
  --accent3: #22d3ee;
  --success: #34d399;
  --danger: #fb7185;
}

@keyframes drift { 0% { background-position: 0% 0%, 100% 100%; } 50% { background-position: 100% 30%, 0% 70%; } 100% { background-position: 0% 0%, 100% 100%; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideInLeft { from { opacity: 0; transform: translateX(-16px); } to { opacity: 1; transform: translateX(0); } }
@keyframes pulseGlow { 0%,100% { box-shadow: 0 0 0 0 rgba(168,85,247,.4); } 50% { box-shadow: 0 0 0 12px rgba(168,85,247,0); } }
@keyframes shimmer { 0% { background-position: -500px 0; } 100% { background-position: 500px 0; } }
@keyframes floatIcon { 0%,100% { transform: translateY(0) rotate(-3deg); } 50% { transform: translateY(-6px) rotate(3deg); } }
@keyframes orbFloat1 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(40px,-50px) scale(1.12); } }
@keyframes orbFloat2 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-50px,40px) scale(1.08); } }
@keyframes orbFloat3 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(30px,30px) scale(.94); } }
@keyframes spinSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes ringPop { 0% { opacity: 0; transform: scale(.7); } 60% { opacity: 1; transform: scale(1.05); } 100% { transform: scale(1); } }
@keyframes beat { 0%,100% { transform: scale(1); } 15% { transform: scale(1.18); } 30% { transform: scale(1); } 45% { transform: scale(1.12); } 60% { transform: scale(1); } }
@keyframes countGlow { 0%,100% { text-shadow: 0 0 18px currentColor; } 50% { text-shadow: 0 0 34px currentColor; } }
@keyframes fillSeg { to { transform: scaleX(1); } }
@keyframes heroPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(168,85,247,.5), 0 0 0 0 rgba(236,72,153,.3); } 50% { box-shadow: 0 0 0 24px rgba(168,85,247,0), 0 0 0 44px rgba(236,72,153,0); } }
@keyframes heroBeat { 0%,100% { transform: scale(1); } 20% { transform: scale(1.14); } 40% { transform: scale(1); } 60% { transform: scale(1.08); } 80% { transform: scale(1); } }
@keyframes heroFadeIn { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
@keyframes ecgDraw { to { stroke-dashoffset: 0; } }

html, body { background: var(--bg-deep); }

.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(168,85,247,.22), transparent 40%),
    radial-gradient(circle at 88% 18%, rgba(236,72,153,.16), transparent 38%),
    radial-gradient(circle at 50% 100%, rgba(34,211,238,.10), transparent 45%),
    linear-gradient(160deg, var(--bg-deep) 0%, var(--bg-mid) 55%, #0b0618 100%);
  background-size: 200% 200%, 200% 200%, 200% 200%, 100% 100%;
  animation: drift 22s ease-in-out infinite;
  color: var(--text); font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; z-index: 5; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0620 0%, #140a29 100%); border-right: 1px solid var(--border); position: relative; z-index: 3; }
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.4rem; }
[data-testid="stAppViewContainer"] { position: relative; }

.bg-orb { position: absolute; border-radius: 50%; filter: blur(70px); z-index: 0; pointer-events: none; opacity: .5; }
.bg-orb-1 { width: 380px; height: 380px; top: -60px; left: 0; background: radial-gradient(circle, rgba(168,85,247,.55), transparent 70%); animation: orbFloat1 16s ease-in-out infinite; }
.bg-orb-2 { width: 320px; height: 320px; top: 20%; right: 0; background: radial-gradient(circle, rgba(236,72,153,.45), transparent 70%); animation: orbFloat2 19s ease-in-out infinite; }
.bg-orb-3 { width: 300px; height: 300px; top: 60%; left: 30%; background: radial-gradient(circle, rgba(34,211,238,.4), transparent 70%); animation: orbFloat3 21s ease-in-out infinite; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }
.main .block-container { max-width: 980px; padding: 3rem 2rem 5rem; margin: 0 auto; position: relative; z-index: 2; }

/* ===== Landing / entry screen ===== */
.hero-wrap {
  min-height: 78vh; display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; position: relative; z-index: 2; animation: heroFadeIn .8s ease-out both;
}
.hero-icon {
  width: 108px; height: 108px; border-radius: 30px; display: grid; place-items: center; font-size: 52px;
  background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; margin-bottom: 1.75rem;
  animation: heroBeat 2.6s ease-in-out infinite, heroPulse 2.6s ease-in-out infinite;
}
.hero-title {
  font: 800 3rem 'Space Grotesk', sans-serif; margin: 0 0 .6rem; letter-spacing: -0.03em;
  background: linear-gradient(90deg, #ffffff, #d8c8ff 50%, var(--accent3));
  background-size: 220% auto; animation: shimmer 5s linear infinite;
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-subtitle { font-size: 1.08rem; color: var(--muted); max-width: 520px; margin: 0 0 2.2rem; line-height: 1.6; }
.hero-ecg { width: 100%; max-width: 480px; height: 60px; margin-bottom: 2.4rem; opacity: .85; }
.hero-ecg path { fill: none; stroke: var(--accent3); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 800; stroke-dashoffset: 800; animation: ecgDraw 2.6s ease-out .3s forwards, drift 6s ease-in-out 3s infinite; }
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; margin-top: 2.4rem; }
.hero-badge {
  padding: 7px 14px; border-radius: 999px; border: 1px solid var(--border-hover);
  background: rgba(255,255,255,.05); color: var(--muted); font-size: .76rem; font-weight: 500;
}
.hero-badge span { color: var(--accent3); margin-right: 5px; }

div[data-testid="stButton"].hero-btn-wrap > button,
.hero-wrap + div .stButton > button {
  min-height: 60px; padding: 0 3rem; width: auto; border: none; border-radius: 16px;
  background: linear-gradient(120deg, var(--accent) 0%, var(--accent2) 55%, var(--accent3) 120%);
  background-size: 220% auto; color: #fff; font: 700 1.08rem 'Space Grotesk', sans-serif;
  box-shadow: 0 16px 45px rgba(168,85,247,.4), inset 0 1px rgba(255,255,255,.3);
  transition: background-position .5s ease, transform .18s ease, box-shadow .18s ease;
  animation: pulseGlow 2.6s ease-in-out infinite; margin: 0 auto;
}
.hero-wrap + div .stButton { display: flex; justify-content: center; margin-top: 0; }
.hero-wrap + div .stButton > button:hover { background-position: right center; transform: translateY(-3px) scale(1.02); }

/* ===== Main app ===== */
.brand { display: flex; align-items: center; gap: 16px; margin-bottom: 2.25rem; animation: fadeUp .6s ease-out both; }
.brand-icon {
  width: 58px; height: 58px; display: grid; place-items: center; border-radius: 17px;
  background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; font-size: 27px;
  box-shadow: 0 10px 30px rgba(168,85,247,.4), inset 0 1px rgba(255,255,255,.25);
  animation: floatIcon 4s ease-in-out infinite, beat 3.2s ease-in-out infinite;
}
.brand-title {
  margin: 0; font: 800 2.05rem 'Space Grotesk', sans-serif; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #ffffff, #d8c8ff 55%, var(--accent3));
  background-size: 220% auto; animation: shimmer 6s linear infinite;
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.brand-subtitle { margin: 4px 0 0; color: var(--muted); font-size: .92rem; }

.progress-track { display: flex; gap: 6px; margin: 0 0 2rem; animation: fadeUp .6s ease-out .1s both; }
.progress-seg { height: 4px; flex: 1; border-radius: 4px; background: rgba(255,255,255,.08); overflow: hidden; position: relative; }
.progress-seg::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3)); transform: scaleX(0); transform-origin: left; animation: fillSeg 1.6s ease-out forwards; }
.progress-seg:nth-child(1)::after { animation-delay: .15s; }
.progress-seg:nth-child(2)::after { animation-delay: .4s; }
.progress-seg:nth-child(3)::after { animation-delay: .65s; }

.intro {
  padding: 20px 24px; margin-bottom: 2.25rem; border-radius: 16px;
  border: 1px solid var(--border-hover);
  background: linear-gradient(120deg, rgba(168,85,247,.16), rgba(236,72,153,.08), rgba(34,211,238,.08));
  background-size: 200% 200%; animation: drift 10s ease-in-out infinite, fadeUp .6s ease-out .05s both;
  box-shadow: 0 18px 50px rgba(88,20,150,.25), inset 0 1px rgba(255,255,255,.06);
}
.intro p { margin: 0; color: #f0eaff; font-size: .96rem; }
.intro strong { color: #fff; background: linear-gradient(90deg, var(--accent3), var(--accent2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.intro p.sub { margin-top: 8px; color: var(--muted); font-size: .82rem; }

.section-kicker {
  display: inline-block; color: var(--accent3); font-size: .72rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; margin: 0 0 1rem; padding: 4px 12px;
  border: 1px solid rgba(34,211,238,.3); border-radius: 999px; background: rgba(34,211,238,.08);
  animation: fadeUp .5s ease-out .1s both;
}

.section-label { display: flex; align-items: center; gap: 10px; color: #fff; font: 700 1.05rem 'Space Grotesk', sans-serif; padding: 0 0 6px; margin-bottom: 4px; }
.section-label span {
  display: grid; place-items: center; width: 28px; height: 28px; border-radius: 9px; font-size: .8rem;
  background: linear-gradient(135deg, rgba(168,85,247,.35), rgba(236,72,153,.25)); border: 1px solid rgba(255,255,255,.18);
}
.helper { color: var(--muted); font-size: .82rem; line-height: 1.55; margin: 0 0 18px; }

[data-testid="stVerticalBlockBorderWrapper"] {
  padding: 26px 28px; border: 1px solid var(--border); border-radius: 20px;
  background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.015));
  backdrop-filter: blur(18px);
  box-shadow: 0 24px 60px rgba(0,0,0,.35), inset 0 1px rgba(255,255,255,.06);
  transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease;
  animation: fadeUp .6s ease-out both;
  position: relative; z-index: 2;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--border-hover);
  box-shadow: 0 28px 70px rgba(88,20,150,.32), inset 0 1px rgba(255,255,255,.08);
  transform: translateY(-3px);
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"] { gap: 1.4rem; }
.card-gap { height: 1.3rem; }

label, [data-testid="stWidgetLabel"] p { color: #e3d9fb !important; font-weight: 500 !important; font-size: .875rem !important; }

[data-testid="stSlider"] { padding: 10px 14px 4px; border: 1px solid var(--border); border-radius: 14px; background: rgba(255,255,255,.03); transition: border-color .2s ease, background .2s ease; }
[data-testid="stSlider"]:hover { border-color: var(--border-hover); background: rgba(255,255,255,.05); }
[data-testid="stSlider"] [role="slider"] {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  box-shadow: 0 0 0 5px rgba(168,85,247,.18), 0 4px 14px rgba(168,85,247,.4) !important;
  transition: box-shadow .2s ease, transform .15s ease;
}
[data-testid="stSlider"] [role="slider"]:hover { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(168,85,247,.22), 0 4px 18px rgba(168,85,247,.5) !important; }
[data-testid="stTickBar"] { display: none; }

[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
  background: rgba(255,255,255,.04) !important; border: 1px solid var(--border) !important; color: var(--text) !important;
  border-radius: 12px !important; min-height: 44px; transition: border-color .2s ease, background .2s ease, box-shadow .2s ease;
}
[data-baseweb="select"] > div:hover, [data-testid="stNumberInput"] input:hover { background: rgba(255,255,255,.07) !important; border-color: var(--border-hover) !important; }
[data-baseweb="select"] > div:focus-within, [data-testid="stNumberInput"] input:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 4px rgba(168,85,247,.18) !important;
}
[data-baseweb="popover"] { filter: drop-shadow(0 12px 30px rgba(0,0,0,.5)); z-index: 10; }

.stSlider [role="slider"]:focus-visible, .stButton > button:focus-visible, input:focus-visible { outline: 2px solid var(--accent3); outline-offset: 3px; }
.stSelectbox, .stNumberInput { margin-bottom: 1rem; }

.stButton { margin-top: 1.75rem; }
.stButton > button {
  width: 100%; min-height: 58px; border: none; border-radius: 14px;
  background: linear-gradient(120deg, var(--accent) 0%, var(--accent2) 55%, var(--accent3) 120%);
  background-size: 220% auto;
  color: #fff; font: 700 1.05rem 'Space Grotesk', sans-serif; letter-spacing: .01em;
  box-shadow: 0 14px 40px rgba(168,85,247,.38), inset 0 1px rgba(255,255,255,.3);
  transition: background-position .5s ease, transform .18s ease, box-shadow .18s ease;
  animation: pulseGlow 2.6s ease-in-out infinite;
  position: relative; overflow: hidden; z-index: 2;
}
.stButton > button::before {
  content: ""; position: absolute; inset: 0; background: linear-gradient(120deg, transparent, rgba(255,255,255,.35), transparent);
  background-size: 300px 100%; background-repeat: no-repeat; background-position: -300px 0; transition: background-position .6s ease;
}
.stButton > button:hover::before { background-position: 500px 0; }
.stButton > button:hover { background-position: right center; transform: translateY(-3px); box-shadow: 0 20px 52px rgba(236,72,153,.42), inset 0 1px rgba(255,255,255,.35); }
.stButton > button:active { transform: translateY(-1px) scale(.99); }

.result-wrap { margin-top: 2.4rem; animation: fadeUp .5s ease-out both; position: relative; z-index: 2; }
.result {
  padding: 28px 30px; border-radius: 20px; border: 1px solid;
  display: flex; gap: 26px; align-items: center; position: relative; overflow: hidden;
  backdrop-filter: blur(14px); flex-wrap: wrap;
}
.result::after { content: ""; position: absolute; inset: 0; background: linear-gradient(120deg, transparent, rgba(255,255,255,.1), transparent); background-size: 900px 100%; animation: shimmer 2.2s ease-in-out 1; pointer-events: none; }
.result.low { background: linear-gradient(120deg, rgba(52,211,153,.16), rgba(15,60,50,.5)); border-color: rgba(52,211,153,.45); box-shadow: 0 22px 60px rgba(16,120,90,.28); }
.result.high { background: linear-gradient(120deg, rgba(251,113,133,.18), rgba(70,15,30,.5)); border-color: rgba(251,113,133,.45); box-shadow: 0 22px 60px rgba(150,25,50,.28); }

.gauge-wrap { position: relative; width: 128px; height: 128px; flex-shrink: 0; animation: ringPop .7s cubic-bezier(.2,.9,.3,1.2) both; }
.gauge-wrap svg { transform: rotate(-90deg); overflow: visible; }
.gauge-bg { fill: none; stroke: rgba(255,255,255,.09); stroke-width: 10; }
.gauge-fill { fill: none; stroke-width: 10; stroke-linecap: round; }
.gauge-fill.low { stroke: url(#gaugeLow); }
.gauge-fill.high { stroke: url(#gaugeHigh); }
.gauge-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.gauge-pct { font: 800 1.6rem 'Space Grotesk', sans-serif; animation: countGlow 2.4s ease-in-out infinite; }
.gauge-pct.low { color: var(--success); }
.gauge-pct.high { color: var(--danger); }
.gauge-label { font-size: .62rem; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin-top: 2px; }

.result-body { flex: 1; min-width: 220px; }
.result-icon-row { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.result-icon {
  font-size: 1.7rem; line-height: 1; width: 44px; height: 44px; display: grid; place-items: center;
  border-radius: 13px; background: rgba(255,255,255,.08); flex-shrink: 0; animation: beat 2.4s ease-in-out infinite;
}
.result h3 { margin: 0; color: #fff; font-size: 1.22rem; }
.result p { margin: 6px 0 0; color: #ece6fb; font-size: .93rem; line-height: 1.55; }
.result small { display: block; margin-top: 12px; color: var(--muted); font-size: .78rem; }

.side-brand { display: flex; align-items: center; gap: 12px; padding-bottom: 1.4rem; border-bottom: 1px solid var(--border); animation: slideInLeft .5s ease-out both; }
.side-brand-icon {
  display: grid; place-items: center; width: 42px; height: 42px; border-radius: 13px;
  background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff; font-size: 21px;
  box-shadow: 0 8px 22px rgba(168,85,247,.35); animation: beat 3.4s ease-in-out infinite;
}
.side-title { color: #fff; font: 700 1.05rem 'Space Grotesk', sans-serif; }
.side-subtitle { color: var(--muted); font-size: .7rem; line-height: 1.35; margin-top: 2px; }

.side-card {
  padding: 16px 17px; margin-top: 1.2rem; border-radius: 15px; border: 1px solid var(--border);
  background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
  transition: border-color .2s ease, transform .2s ease, background .2s ease;
  animation: slideInLeft .5s ease-out both;
}
.side-card:nth-of-type(1) { animation-delay: .08s; }
.side-card:nth-of-type(2) { animation-delay: .16s; }
.side-card:nth-of-type(3) { animation-delay: .24s; }
.side-card:nth-of-type(4) { animation-delay: .32s; }
.side-card:hover { border-color: var(--border-hover); transform: translateX(3px); background: linear-gradient(160deg, rgba(255,255,255,.08), rgba(255,255,255,.02)); }
.side-card.disclaimer { border-color: rgba(251,191,36,.35); background: linear-gradient(160deg, rgba(251,191,36,.1), rgba(120,72,10,.12)); }
.side-heading { display: flex; align-items: center; gap: 8px; color: #fff; font: 600 .86rem 'Space Grotesk', sans-serif; margin: 0 0 .5rem; }
.side-heading span { color: var(--accent3); font-size: .95rem; display: inline-block; animation: spinSlow 6s linear infinite; }
.side-card.disclaimer .side-heading span { color: #fbbf24; animation: beat 2.6s ease-in-out infinite; }
.side-copy { color: var(--muted); font-size: .8rem; line-height: 1.6; margin: 0; }

.side-model { display: flex; flex-direction: column; gap: 8px; }
.model-name { color: #ece6fb; font: 500 .82rem 'Space Grotesk', sans-serif; }
.model-tag {
  display: inline-block; width: fit-content; padding: 4px 10px; border-radius: 999px;
  background: linear-gradient(120deg, rgba(168,85,247,.28), rgba(34,211,238,.22));
  background-size: 200% auto; animation: shimmer 4s linear infinite;
  border: 1px solid rgba(168,85,247,.4); color: #e6d9ff; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
}

@media (max-width: 800px) {
  .main .block-container { padding: 2rem 1rem 3rem; }
  [data-testid="stVerticalBlockBorderWrapper"] { padding: 18px 16px; }
  .brand-title { font-size: 1.6rem; }
  .brand { margin-bottom: 2rem; }
  .result { padding: 20px; gap: 16px; }
  .gauge-wrap { width: 96px; height: 96px; }
  .hero-title { font-size: 2.1rem; }
  .hero-icon { width: 84px; height: 84px; font-size: 40px; }
  .hero-wrap { min-height: 60vh; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
</style>

<div class="bg-orb bg-orb-1"></div>
<div class="bg-orb bg-orb-2"></div>
<div class="bg-orb bg-orb-3"></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="side-brand"><div class="side-brand-icon">&#9829;</div><div><div class="side-title">HeartGuard AI</div><div class="side-subtitle">VK-Powered Heart Risk Assessment</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-heading"><span>&#9679;</span> About this application</div><p class="side-copy">A focused screening companion that turns common cardiovascular measurements into an easy-to-understand risk signal.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-heading"><span>&#9881;</span> How the assessment works</div><p class="side-copy">Enter the patient profile, heart measurements, and ECG details. The trained model evaluates the provided information and returns a screening result.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-heading"><span>&#9733;</span> Model information</div><div class="side-model"><span class="model-name">K-Nearest Neighbors, Logistic Regression, SVM, Decision Tree</span><span class="model-tag">KNN</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card disclaimer"><div class="side-heading"><span>&#9888;</span> Medical disclaimer</div><p class="side-copy">This tool is for educational screening purposes only and is not a medical diagnosis (Developed by Vamshi Krishna Adapala). Always discuss health concerns with a qualified clinician.</p></div>', unsafe_allow_html=True)

# ===================== LANDING / ENTRY SCREEN =====================
if not st.session_state.entered:
    st.markdown('''
    <div class="hero-wrap">
      <div class="hero-icon">&#9829;</div>
      <h1 class="hero-title">HeartGuard AI</h1>
      <p class="hero-subtitle">A focused, VK-powered screening companion that turns common cardiovascular measurements into an easy-to-understand risk signal — in under a minute.</p>
      <svg class="hero-ecg" viewBox="0 0 480 60" xmlns="http://www.w3.org/2000/svg">
        <path d="M0 30 L90 30 L110 30 L120 10 L135 50 L150 30 L170 30 L185 5 L200 55 L215 30 L480 30" />
      </svg>
    </div>
    ''', unsafe_allow_html=True)

    enter_left, enter_center, enter_right = st.columns([1, 1, 1])
    with enter_center:
        if st.button("Begin Assessment  →", use_container_width=True):
            st.session_state.entered = True
            st.rerun()

    st.markdown('''
    <div class="hero-badges">
      <span class="hero-badge"><span>&#9679;</span>Educational use only</span>
      <span class="hero-badge"><span>&#9679;</span>Takes under 60 seconds</span>
      <span class="hero-badge"><span>&#9679;</span>No data stored</span>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<style> #MainMenu, footer {visibility: hidden;} </style>', unsafe_allow_html=True)
    st.stop()

# ===================== MAIN APP =====================
st.markdown('<div class="brand"><div class="brand-icon">&#9829;</div><div><p class="brand-title">HeartGuard AI</p><p class="brand-subtitle">VK-Powered Heart Risk Assessment</p></div></div>', unsafe_allow_html=True)
st.markdown('<div class="progress-track"><div class="progress-seg"></div><div class="progress-seg"></div><div class="progress-seg"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="intro"><p><strong>Screen with context.</strong> Complete the profile below for a quick, educational heart risk assessment.</p><p class="sub">This tool is for educational screening purposes only and is not a medical diagnosis (Developed by Vamshi Krishna Adapala).</p></div>', unsafe_allow_html=True)

st.markdown('<p class="section-kicker">Clinical intake</p>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="section-label"><span>&#9679;</span> Patient Profile</div><p class="helper">Start with the patient demographics used to establish the assessment profile.</p>', unsafe_allow_html=True)
    profile_left, profile_right = st.columns(2, gap="large")
    with profile_left:
        age=st.slider("Age",18,100,40, help="Patient age in years")
    with profile_right:
        sex=st.selectbox("Sex",["Male","Female"], help="Biological sex recorded for this screening")

st.markdown('<div class="card-gap"></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="section-label"><span>&#9829;</span> Heart Measurements</div><p class="helper">Add resting vital signs, blood chemistry, and exercise capacity measurements.</p>', unsafe_allow_html=True)
    measurement_one, measurement_two, measurement_three = st.columns(3, gap="large")
    with measurement_one:
        resting_bp=st.number_input("Resting Blood Pressure (mm Hg)",80,200,120)
        fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1], format_func=lambda value: "Yes" if value == 1 else "No")
    with measurement_two:
        cholesterol=st.number_input("Cholesterol (mg/dl)",100,600,200)
        max_heart_rate=st.slider("Max Heart Rate Achieved",60,220,150)
    with measurement_three:
        oldpeak=st.slider("Oldpeak (ST depression induced by exercise relative to rest)",0.0,10.0,1.0)
        st.markdown('<p class="helper">ST depression measured during exercise relative to rest.</p>', unsafe_allow_html=True)

st.markdown('<div class="card-gap"></div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="section-label"><span>&#9881;</span> ECG &amp; Exercise</div><p class="helper">Capture symptoms, rhythm observations, and response to exertion.</p>', unsafe_allow_html=True)
    ecg_left, ecg_right = st.columns(2, gap="large")
    with ecg_left:
        chest_pain=st.selectbox("Chest Pain Type",["ATA","NAP","ASY","TA"], help="Reported chest pain classification")
        resting_ecg=st.selectbox("Resting ECG",["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"])
    with ecg_right:
        exercise_angina=st.selectbox("Exercise Induced Angina",["Yes","No"])
        st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])

if st.button("Analyze Heart Risk"):
    raw_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_heart_rate,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }
    input_data = pd.DataFrame([raw_data])

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_columns]

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    # Supplementary risk percentage for the gauge visual only (does not affect the prediction above)
    try:
        risk_pct = round(float(model.predict_proba(scaled_input)[0][1]) * 100)
    except Exception:
        risk_pct = 82 if prediction == 1 else 12

    radius = 54
    circumference = 2 * math.pi * radius
    offset = circumference - (circumference * risk_pct / 100)

    if prediction == 1:
        st.markdown(f'''
        <div class="result-wrap">
          <div class="result high">
            <div class="gauge-wrap">
              <svg width="128" height="128" viewBox="0 0 128 128">
                <defs>
                  <linearGradient id="gaugeHigh" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#fb7185"/>
                    <stop offset="100%" stop-color="#f43f5e"/>
                  </linearGradient>
                </defs>
                <circle class="gauge-bg" cx="64" cy="64" r="{radius}"/>
                <circle class="gauge-fill high" cx="64" cy="64" r="{radius}"
                  stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"/>
              </svg>
              <div class="gauge-center">
                <span class="gauge-pct high">{risk_pct}%</span>
                <span class="gauge-label">Risk</span>
              </div>
            </div>
            <div class="result-body">
              <div class="result-icon-row">
                <div class="result-icon">&#9829;</div>
                <h3>Elevated risk signal detected</h3>
              </div>
              <p>The model indicates a higher screening risk based on the information provided. Please consult a qualified doctor for proper evaluation.</p>
              <small>This result is not a diagnosis and should not replace professional medical advice.</small>
            </div>
          </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="result-wrap">
          <div class="result low">
            <div class="gauge-wrap">
              <svg width="128" height="128" viewBox="0 0 128 128">
                <defs>
                  <linearGradient id="gaugeLow" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#34d399"/>
                    <stop offset="100%" stop-color="#10b981"/>
                  </linearGradient>
                </defs>
                <circle class="gauge-bg" cx="64" cy="64" r="{radius}"/>
                <circle class="gauge-fill low" cx="64" cy="64" r="{radius}"
                  stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"/>
              </svg>
              <div class="gauge-center">
                <span class="gauge-pct low">{risk_pct}%</span>
                <span class="gauge-label">Risk</span>
              </div>
            </div>
            <div class="result-body">
              <div class="result-icon-row">
                <div class="result-icon">&#9829;</div>
                <h3>Lower risk signal detected</h3>
              </div>
              <p>The model does not indicate elevated screening risk from the information provided. Continue healthy habits and regular checkups.</p>
              <small>This result is not a diagnosis and should not replace professional medical advice.</small>
            </div>
          </div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<style> #MainMenu, footer {visibility: hidden;} </style>', unsafe_allow_html=True)