from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
FEATURES = ['study_hours', 'attendance', 'previous_marks', 'sleep_hours']
st.set_page_config(page_title='EduPredict | Student Success AI', page_icon='🎓', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif;}
.stApp{background:radial-gradient(ellipse at 80% 0%,#182c50 0%,#0b1224 44%,#080d1b 100%);color:#eef4ff;}
h1,h2,h3{font-family:'Outfit',sans-serif!important;letter-spacing:-.025em;}
[data-testid="stSidebar"]{background:#0c1730;border-right:1px solid #243657;}
[data-testid="stSidebar"] *{color:#e8f1ff;}
.block-container{padding-top:2rem;max-width:1240px;}
.hero{padding:34px 38px;border-radius:24px;background:linear-gradient(115deg,#214eaa,#5b3ab5 70%,#8a44ba);box-shadow:0 18px 45px #02081666;margin-bottom:25px;}
.eyebrow{font-size:12px;letter-spacing:2px;font-weight:800;color:#c9e5ff;}
.hero h1{font-size:clamp(32px,4vw,52px);line-height:1.1;color:white;margin:12px 0;}
.hero p{color:#e5eaff;font-size:16px;max-width:660px;margin:0;}
.tile{background:#121f38;border:1px solid #2b3d5b;padding:23px;border-radius:19px;min-height:122px;}
.tile-label{font-size:13px;color:#9db1d4;font-weight:600;}
.tile-value{font-family:'Outfit';font-size:31px;font-weight:700;color:#fff;margin-top:8px;}
.section-label{font-weight:800;font-size:19px;color:#f0f5ff;margin:10px 0 16px;}
.small-note{font-size:13px;color:#aebfdd;line-height:1.7;}
[data-testid="stMetric"]{background:#13223e;border:1px solid #324768;border-radius:18px;padding:18px;}
[data-testid="stMetricValue"]{color:#d6ecff;}
.stButton>button[kind="primary"]{border-radius:12px;background:linear-gradient(100deg,#4889ff,#9265ff);border:0;color:white;font-weight:800;padding:12px 20px;}
[data-testid="stForm"]{border:1px solid #30415e;background:#101c33;border-radius:20px;padding:20px;}
footer{visibility:hidden;}
@media(max-width:700px){.hero{padding:23px}.block-container{padding:1rem;}}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')

@st.cache_data
def load_info():
    return json.loads((ROOT / 'model_info.json').read_text(encoding='utf-8'))

st.sidebar.markdown('## 🎓 EduPredict')
st.sidebar.caption('STUDENT SUCCESS INTELLIGENCE')
st.sidebar.divider()
page = st.sidebar.radio('Navigation', ['🏠 Dashboard', '📊 Model insights', 'ℹ️ About'], label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.info('🧪 Demo model trained on synthetic data. Predictions are illustrative, not academic advice.')
st.sidebar.caption('Internship Task 05 • Streamlit + scikit-learn')

try:
    model = load_model()
    info = load_info()
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError('Model feature order does not match app inputs.')
except FileNotFoundError:
    st.error('Model files are missing. Run `python train_model.py` and restart the app.')
    st.stop()
except (ValueError, KeyError, OSError) as exc:
    st.error(f'Unable to initialize the prediction model: {exc}')
    st.stop()

if page == '🏠 Dashboard':
    st.markdown('''<div class="hero"><div class="eyebrow">✦ AI-POWERED STUDENT ANALYTICS</div><h1>Understand your potential.<br>Plan your progress.</h1><p>Explore a machine learning demo that estimates exam outcomes from learning habits and academic inputs.</p></div>''', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="tile"><div class="tile-label">Model type</div><div class="tile-value">Random Forest</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="tile"><div class="tile-label">Holdout accuracy*</div><div class="tile-value">{info["accuracy"]:.1%}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="tile"><div class="tile-label">Training samples</div><div class="tile-value">{info["samples"]:,}</div></div>', unsafe_allow_html=True)
    st.caption('*Measured on held-out synthetic demonstration records; not evidence of real-world predictive performance.')
    st.markdown('<div class="section-label">🔮 Try a prediction</div>', unsafe_allow_html=True)
    left,right = st.columns([1.15, .85], gap='large')
    with left:
        with st.form('predict_form'):
            st.markdown('#### Student profile')
            st.caption('Adjust the values below to simulate different learning scenarios.')
            study_hours = st.slider('📚 Daily study hours', 0.0, 12.0, 4.0, .5)
            attendance = st.slider('📅 Attendance (%)', 0, 100, 80)
            previous_marks = st.slider('📝 Previous exam score (%)', 0, 100, 65)
            sleep_hours = st.slider('🌙 Daily sleep hours', 0.0, 12.0, 7.0, .5)
            submitted = st.form_submit_button('✨ Predict performance', type='primary', use_container_width=True)
    with right:
        st.markdown('#### Prediction studio')
        if submitted:
            try:
                row = pd.DataFrame([[study_hours,attendance,previous_marks,sleep_hours]], columns=FEATURES)
                if not ((row['study_hours'].between(0,12) & row['attendance'].between(0,100) & row['previous_marks'].between(0,100) & row['sleep_hours'].between(0,12)).all()):
                    raise ValueError('Input is outside the supported range.')
                pred = int(model.predict(row)[0])
                probabilities = model.predict_proba(row)[0]
                pass_probability = float(probabilities[list(model.classes_).index(1)])
                if pred == 1: st.success('🌟 Model result: Likely pass')
                else: st.warning('📈 Model result: Needs improvement')
                st.metric('Estimated pass probability', f'{pass_probability:.1%}')
                st.progress(pass_probability)
                st.caption('Estimated probability from this synthetic-data model; not a guarantee.')
                st.markdown('**Probability breakdown**')
                st.bar_chart(pd.DataFrame({'Probability':[1-pass_probability,pass_probability]},index=['Needs improvement','Likely pass']), horizontal=True)
            except (ValueError, KeyError, IndexError) as exc:
                st.error(f'Could not calculate prediction: {exc}')
            except Exception:
                st.error('Prediction failed. Please check the model and inputs.')
        else:
            st.info('👈 Fill out the profile and click **Predict performance** to see your result.')
            st.markdown('**What you will see**')
            st.markdown('✔ Predicted category  \n✔ Probability percentage  \n✔ Visual probability chart')
    st.divider()
    st.markdown('<div class="small-note">🔒 No student names, emails, or personal identifiers are collected. All predictions run in the app session.</div>', unsafe_allow_html=True)
elif page == '📊 Model insights':
    st.title('📊 Model insights')
    st.write('How the demonstration classifier was built and evaluated.')
    a,b = st.columns(2)
    a.metric('Holdout accuracy',f'{info["accuracy"]:.1%}')
    b.metric('Synthetic samples',f'{info["samples"]:,}')
    st.subheader('Feature importance')
    importance = pd.DataFrame({'Feature':['Study hours','Attendance','Previous marks','Sleep hours'], 'Importance':model.feature_importances_}).sort_values('Importance')
    st.bar_chart(importance.set_index('Feature'), horizontal=True)
    st.caption('Feature importance indicates model reliance, not causation.')
    st.warning('Training records are synthetic and generated from a designed scoring rule. Evaluation on held-out synthetic records cannot establish reliability for real students.')
else:
    st.title('ℹ️ About EduPredict')
    st.write('An end-to-end Streamlit and scikit-learn classification demo for InternCircle Data Science & AI Project 05.')
    st.markdown('**Pipeline:** Generate demo data → train Random Forest → save with joblib → load with Streamlit cache → validate inputs → show prediction and probability.')
    st.markdown('**Stack:** Python, pandas, scikit-learn, joblib, Streamlit.')
    st.info('For an authentic educational prediction tool, replace the synthetic dataset with a suitable consented, representative real dataset and re-evaluate model fairness and calibration.')
