import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HeartGuard AI", page_icon="&#9829;", layout="wide", initial_sidebar_state="expanded")

model=joblib.load('KNN_heart.pkl')
scaler=joblib.load('scaler_heart.pkl')
expected_columns=joblib.load('columns_heart.pkl')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root { --navy: #071a2b; --panel: #0d2639; --panel-light: #12344a; --cyan: #61e3e0; --muted: #9ab2c2; --line: #21475d; }
.stApp { background: radial-gradient(circle at 9% 3%, rgba(50, 133, 151, .16), transparent 27rem), linear-gradient(135deg, #061522 0%, #092237 55%, #061925 100%); color: #eaf7fa; font-family: 'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #061522 0%, #081d2d 58%, #09263a 100%); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding: 1.8rem 1.35rem 2rem; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
.main .block-container { max-width: 1280px; padding: 3.25rem 4rem 5rem; }
.brand { display: flex; align-items: center; gap: 16px; margin-bottom: 2.5rem; }
.brand-icon { width: 58px; height: 58px; display: grid; place-items: center; border: 1px solid rgba(133, 244, 237, .55); border-radius: 18px; background: linear-gradient(145deg, #63e6df, #1e98ad); color: #062032; font-size: 30px; box-shadow: 0 8px 30px rgba(57, 211, 216, .22); }
.brand-title { margin: 0; color: #f5ffff; font: 700 2rem 'Space Grotesk', sans-serif; }
.brand-subtitle { margin: 3px 0 0; color: var(--muted); font-size: .94rem; }
.intro { padding: 20px 24px; border: 1px solid rgba(97, 227, 224, .25); border-radius: 14px; background: linear-gradient(100deg, rgba(28, 107, 125, .32), rgba(13, 38, 57, .35)); box-shadow: 0 14px 42px rgba(0, 0, 0, .14), inset 0 1px rgba(255, 255, 255, .05); margin-bottom: 2.3rem; }
.intro strong { color: var(--cyan); }
.intro p { margin: 0; color: #c8dce3; }
.section-label { display: flex; align-items: center; gap: 10px; color: #f2ffff; font: 600 1.08rem 'Space Grotesk', sans-serif; padding: 18px 0 9px; border-bottom: 1px solid var(--line); margin-bottom: 10px; }
.section-label span { display: grid; place-items: center; width: 25px; height: 25px; border: 1px solid rgba(97, 227, 224, .35); border-radius: 8px; color: var(--cyan); font-size: .75rem; background: rgba(97, 227, 224, .1); }
.helper { color: var(--muted); font-size: .82rem; line-height: 1.45; margin: 0 0 16px; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 10px 20px 16px; border: 1px solid rgba(97, 227, 224, .18); border-radius: 16px; background: linear-gradient(145deg, rgba(18, 52, 74, .48), rgba(7, 26, 43, .34)); box-shadow: 0 16px 38px rgba(0, 0, 0, .14), inset 0 1px rgba(255, 255, 255, .04); }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"] { gap: 1rem; }
.card-gap { height: 1.1rem; }
.section-kicker { color: var(--cyan); font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: 1.1rem 0 0; }
label, [data-testid="stWidgetLabel"] p { color: #c9dce4 !important; font-weight: 500 !important; }
[data-testid="stSlider"] { padding: 8px 13px 2px; border: 1px solid rgba(40, 83, 106, .72); border-radius: 12px; background: rgba(13, 38, 57, .42); }
[data-testid="stSlider"] [role="slider"] { background: var(--cyan); box-shadow: 0 0 0 4px rgba(97, 227, 224, .12); }
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input { background: rgba(16, 44, 64, .86); border: 1px solid #28536a; color: #f1ffff; border-radius: 10px; min-height: 42px; transition: border-color .2s ease, background .2s ease, box-shadow .2s ease; }
[data-baseweb="select"] > div:hover, [data-testid="stNumberInput"] input:focus { background: #14364b; border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(97, 227, 224, .1); }
[data-baseweb="select"] > div:focus-within { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(97, 227, 224, .1); }
.stSlider [role="slider"]:focus-visible, .stButton > button:focus-visible, input:focus-visible { outline: 3px solid rgba(244, 194, 116, .9); outline-offset: 3px; }
.stSelectbox, .stNumberInput { margin-bottom: 1rem; }
.stButton { margin-top: 1.5rem; }
.stButton > button { width: 100%; min-height: 58px; border: 1px solid rgba(144, 248, 238, .5); border-radius: 12px; background: linear-gradient(100deg, #55dedb, #2ba5bd); color: #062032; font: 700 1rem 'DM Sans', sans-serif; box-shadow: 0 10px 28px rgba(43, 165, 189, .24); transition: transform .2s ease, box-shadow .2s ease, filter .2s ease; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(43, 165, 189, .38); color: #041522; }
.stButton > button:active { transform: translateY(0); filter: brightness(.96); }
.result { margin-top: 2.4rem; padding: 26px 30px; border-radius: 16px; border: 1px solid; display: flex; gap: 16px; align-items: flex-start; box-shadow: 0 18px 45px rgba(0, 0, 0, .2), inset 0 1px rgba(255, 255, 255, .08); animation: result-in .35s ease-out both; }
.result.low { background: linear-gradient(110deg, rgba(25, 137, 117, .35), rgba(10, 57, 65, .7)); border-color: #36c9ad; }
.result.high { background: linear-gradient(110deg, rgba(153, 47, 62, .42), rgba(68, 25, 41, .75)); border-color: #f17983; }
.result-icon { font-size: 2rem; line-height: 1; }
.result h3 { margin: 0 0 5px; color: #fff; }
.result p { margin: 0; color: #d1e5e7; }
.result small { display: block; margin-top: 12px; color: #b5cdd0; }
.result h3 { font-size: 1.2rem; }
@keyframes result-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.side-brand { display: flex; align-items: center; gap: 11px; padding-bottom: 1.25rem; border-bottom: 1px solid rgba(97, 227, 224, .18); }
.side-brand-icon { display: grid; place-items: center; width: 39px; height: 39px; border-radius: 12px; background: linear-gradient(145deg, #63e6df, #1e98ad); color: #062032; font-size: 20px; box-shadow: 0 7px 20px rgba(57, 211, 216, .18); }
.side-title { color: #f2ffff; font: 700 1.08rem 'Space Grotesk', sans-serif; }
.side-subtitle { color: var(--muted); font-size: .7rem; line-height: 1.35; margin-top: 2px; }
.side-card { padding: 14px 15px; margin-top: 1.15rem; border: 1px solid rgba(97, 227, 224, .16); border-radius: 13px; background: rgba(18, 52, 74, .42); box-shadow: inset 0 1px rgba(255, 255, 255, .04); }
.side-card.disclaimer { border-color: rgba(242, 194, 116, .3); background: rgba(92, 69, 37, .18); }
.side-heading { display: flex; align-items: center; gap: 8px; color: #f2ffff; font: 600 .86rem 'Space Grotesk', sans-serif; margin: 0 0 .45rem; }
.side-heading span { color: var(--cyan); font-size: .9rem; }
.side-card.disclaimer .side-heading span { color: #f2c274; }
.side-copy { color: var(--muted); font-size: .79rem; line-height: 1.55; margin: 0; }
.side-model { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.model-name { color: #e7fbfb; font: 600 .82rem 'Space Grotesk', sans-serif; }
.model-tag { padding: 4px 8px; border: 1px solid rgba(97, 227, 224, .3); border-radius: 6px; color: var(--cyan); font-size: .68rem; font-weight: 700; letter-spacing: .05em; }
@media (max-width: 800px) { .main .block-container { padding: 2rem 1rem 3rem; } [data-testid="stVerticalBlockBorderWrapper"] { padding: 8px 13px 13px; } .brand-title { font-size: 1.6rem; } .brand { margin-bottom: 2rem; } .result { padding: 20px; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="side-brand"><div class="side-brand-icon">&#9829;</div><div><div class="side-title">HeartGuard AI</div><div class="side-subtitle">VK-Powered Heart Risk Assessment</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-heading"><span>&#9679;</span> About this application</div><p class="side-copy">A focused screening companion that turns common cardiovascular measurements into an easy-to-understand risk signal.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-heading"><span>&#9881;</span> How the assessment works</div><p class="side-copy">Enter the patient profile, heart measurements, and ECG details. The trained model evaluates the provided information and returns a screening result.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card"><div class="side-heading"><span>&#9733;</span> Model information</div><div class="side-model"><span class="model-name">K-Nearest Neighbors, Logistic Regression, SVM, Decision Tree </span><span class="model-tag">KNN</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card disclaimer"><div class="side-heading"><span>&#9888;</span> Medical disclaimer</div><p class="side-copy">This tool is for educational screening purposes only and is not a medical diagnosis (Developed by Vamshi Krishna Adapala). Always discuss health concerns with a qualified clinician.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="brand"><div class="brand-icon">&#9829;</div><div><p class="brand-title">HeartGuard AI</p><p class="brand-subtitle">VK-Powered Heart Risk Assessment</p></div></div>', unsafe_allow_html=True)
st.markdown('<div class="intro"><p><strong>Screen with context.</strong> Complete the profile below for a quick, educational heart risk assessment.</p><p style="margin-top: 7px; font-size: .82rem;">This tool is for educational screening purposes only and is not a medical diagnosis (Developed by Vamshi Krishna Adapala).</p></div>', unsafe_allow_html=True)

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
    if prediction == 1:
        st.markdown('<div class="result high"><div class="result-icon">&#9829;</div><div><h3>Elevated risk signal detected</h3><p>The model indicates a higher screening risk based on the information provided. Please consult a qualified doctor for proper evaluation.</p><small>This result is not a diagnosis and should not replace professional medical advice.</small></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result low"><div class="result-icon">&#9829;</div><div><h3>Lower risk signal detected</h3><p>The model does not indicate elevated screening risk from the information provided. Continue healthy habits and regular checkups.</p><small>This result is not a diagnosis and should not replace professional medical advice.</small></div></div>', unsafe_allow_html=True)

st.markdown('<style> #MainMenu, footer {visibility: hidden;} </style>', unsafe_allow_html=True)
      