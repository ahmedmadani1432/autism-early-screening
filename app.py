# # app.py  -- Streamlit ASD screening demo (simple, interpretable)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import shap
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
# import matplotlib.pyplot as plt
# import base64
# import io
# import joblib
# model = joblib.load("model.pkl")

# st.set_page_config(layout="wide", page_title="ASD Early-Screening (Demo)")

# # ---------- helpers ----------
# @st.cache_data
# def load_sample_data():
#     # you should replace this with reading your CSV dataset
#     # minimal example: 5 features representing questionnaire items
#     df = pd.DataFrame({
#         "q1":[1,3,2,4],
#         "q2":[0,1,1,0],
#         "q3":[2,2,3,1],
#         "age":[24,30,18,10],
#         "gender":[0,1,0,0],   # 0=male,1=female
#         "label":[0,1,0,1]
#     })
#     return df

# @st.cache_data
# def train_or_load_model(df):
#     X = df[["q1","q2","q3","age","gender"]]
#     y = df["label"]
#     # small interpretable model
#     lr = LogisticRegression(max_iter=1000)
#     lr.fit(X, y)
#     clf = CalibratedClassifierCV(lr, cv="prefit")
#     # calibrate with the same training set (ok for demo; for real work use holdout)
#     clf.fit(X, y)
#     return clf, X

# def shap_html_plot(explainer, X_row):
#     # create force_plot and return HTML
#     shap_values = explainer.shap_values(X_row)
#     # use matplotlib fallback: bar plot of abs shap
#     sv = np.abs(shap_values).reshape(-1)
#     feat = X_row.columns.tolist()
#     fig, ax = plt.subplots(figsize=(6,2.5))
#     ax.barh(feat, sv[::-1])
#     ax.set_xlabel("abs(SHAP value)")
#     st.pyplot(fig)

# # ---------- App UI ----------
# st.title("ASD Early-Screening — Clinically interpretable demo")

# # layout: left form, right results/explainability
# col1, col2 = st.columns([1,1.3])

# with col1:
#     st.header("Patient input (questionnaire)")
#     df = load_sample_data()
#     # Input form
#     with st.form("patient_form"):
#         q1 = st.slider("Q1: social interaction score (0-4)", 0, 4, 2)
#         q2 = st.selectbox("Q2: repetitive behavior (0=No,1=Yes)", [0,1], index=0)
#         q3 = st.slider("Q3: unusual speech (0-4)", 0, 4, 2)
#         age = st.number_input("Age (years)", 1, 120, 24)
#         gender = st.selectbox("Gender", ["Male","Female"])
#         submitted = st.form_submit_button("Evaluate")
#     # sample quick load
#     if st.button("Load sample patient"):
#         sample = df.iloc[0]
#         q1,q2,q3,age,gender = int(sample.q1), int(sample.q2), int(sample.q3), int(sample.age), ("Male" if sample.gender==0 else "Female")
#         st.experimental_rerun()

# with col2:
#     st.header("Result")
#     # train/load model
#     # model, Xtrain = train_or_load_model(df)

#     # map gender
#     gnum = 0 if gender=="Male" else 1
#     X_row = pd.DataFrame([[q1,q2,q3,age,gnum]], columns=["q1","q2","q3","age","gender"])
#     if 'submitted' not in locals():
#         st.info("Fill the questionnaire and click Evaluate.")
#     else:
#         prob = model.predict_proba(X_row)[0,1]
#         # risk band
#         if prob < 0.3:
#             band = "Low"
#             rec = "No immediate action. Monitor and rescreen."
#         elif prob < 0.7:
#             band = "Medium"
#             rec = "Consider clinical follow-up and more screening."
#         else:
#             band = "High"
#             rec = "Recommend referral to specialist."
#         st.metric("Predicted ASD risk", f"{prob:.2f}", delta=band)
#         st.write("Recommendation:", rec)
#         # show model coefficients (interpretable)
#         if hasattr(model.base_estimator, "coef_"):
#             coef = model.base_estimator.coef_[0]
#             feat = X_row.columns.tolist()
#             coef_df = pd.DataFrame({"feature":feat, "coef":coef})
#             st.subheader("Model (Logistic Regression) coefficients")
#             st.table(coef_df.style.format({"coef":"{:.3f}"}))
#         # SHAP explainability (global + local)
#         st.subheader("Local explanation (SHAP-like)")
#         explainer = shap.LinearExplainer(model.base_estimator, Xtrain, feature_perturbation="interventional")
#         shap_html_plot(explainer, X_row)

#         # Downloadable report (CSV)
#         out = X_row.copy()
#         out["pred_prob"]=prob
#         csv = out.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         st.markdown(f"[Download report](data:file/csv;base64,{b64})")

# st.markdown("---")
# st.caption("Demo: screening aid — not a diagnostic device. Replace demo model with trained model and proper validation for deployment.")

# 2nd phase
# import streamlit as st
# import pandas as pd
# import joblib

# # Load trained model
# model = joblib.load("model.pkl")

# st.set_page_config(page_title="ASD Early Screening", layout="wide")

# st.title("ASD Early-Screening — Clinically Interpretable Demo")
# st.markdown("Autism early screening tool based on questionnaire responses.")

# col1, col2 = st.columns([1,1])

# # ---------------- INPUT PANEL ----------------
# with col1:
#     st.header("Child Screening Questionnaire")

#     with st.form("asd_form"):

#         st.subheader("Answer the following questions")

#         a1 = st.selectbox("A1: Does the child respond when their name is called?", [0,1])
#         a2 = st.selectbox("A2: Does the child make eye contact?", [0,1])
#         a3 = st.selectbox("A3: Does the child point to objects to show interest?", [0,1])
#         a4 = st.selectbox("A4: Does the child imitate actions or gestures?", [0,1])
#         a5 = st.selectbox("A5: Does the child engage in pretend play?", [0,1])
#         a6 = st.selectbox("A6: Does the child follow where you are looking?", [0,1])
#         a7 = st.selectbox("A7: Does the child show interest in other children?", [0,1])
#         a8 = st.selectbox("A8: Does the child bring objects to show parents?", [0,1])
#         a9 = st.selectbox("A9: Does the child respond to emotional expressions?", [0,1])
#         a10 = st.selectbox("A10: Does the child look at faces when interacting?", [0,1])

#         submitted = st.form_submit_button("Evaluate ASD Risk")

# # ---------------- RESULT PANEL ----------------
# with col2:
#     st.header("Screening Result")

#     if submitted:

#         # Create input dataframe
#         X_row = pd.DataFrame(
#             [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]],
#             columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
#         )

#         # Predict
#         prediction = model.predict(X_row)[0]
#         probability = model.predict_proba(X_row)[0][1]

#         st.metric("ASD Risk Probability", f"{probability:.2f}")

#         # Risk interpretation
#         if probability < 0.30:
#             st.success("Low ASD Risk")
#             recommendation = "Continue monitoring developmental milestones."
#         elif probability < 0.70:
#             st.warning("Moderate ASD Risk")
#             recommendation = "Consider further developmental screening."
#         else:
#             st.error("High ASD Risk")
#             recommendation = "Clinical evaluation by a specialist recommended."

#         st.subheader("Recommendation")
#         st.write(recommendation)

#         # Display input summary
#         st.subheader("Input Summary")
#         st.dataframe(X_row)

#         # Download report
#         report = X_row.copy()
#         report["ASD_Risk_Probability"] = probability
#         csv = report.to_csv(index=False)

#         st.download_button(
#             label="Download Screening Report",
#             data=csv,
#             file_name="asd_screening_report.csv",
#             mime="text/csv"
#         )

#     else:
#         st.info("Fill the questionnaire and click 'Evaluate ASD Risk'.")

# st.markdown("---")
# st.caption("This tool is intended for research and screening purposes only and is not a clinical diagnosis.")


# 3rd phase
# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load("model.pkl")

# st.set_page_config(page_title="Autism Early Screening", layout="wide")

# st.title("🧠 Autism Early Screening System")
# st.caption("Clinically Interpretable ML Screening Tool")

# col1, col2 = st.columns([1,1])

# # ---------------- INPUT ----------------
# with col1:

#     st.subheader("Child Behaviour Questionnaire")

#     with st.form("screening_form"):

#         a1 = st.selectbox("Responds to name", [0,1])
#         a2 = st.selectbox("Maintains eye contact", [0,1])
#         a3 = st.selectbox("Points to objects", [0,1])
#         a4 = st.selectbox("Imitates actions", [0,1])
#         a5 = st.selectbox("Engages in pretend play", [0,1])
#         a6 = st.selectbox("Follows gaze", [0,1])
#         a7 = st.selectbox("Shows interest in peers", [0,1])
#         a8 = st.selectbox("Brings objects to parents", [0,1])
#         a9 = st.selectbox("Responds to emotions", [0,1])
#         a10 = st.selectbox("Looks at faces", [0,1])

#         submitted = st.form_submit_button("Evaluate ASD Risk")

# # ---------------- RESULT ----------------
# with col2:

#     st.subheader("Screening Result")

#     if submitted:

#         X = pd.DataFrame([[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]],
#         columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10'])

#         prob = model.predict_proba(X)[0][1]

#         st.metric("ASD Probability", f"{prob:.2f}")

#         if prob < 0.30:
#             st.success("Low ASD Risk")
#         elif prob < 0.70:
#             st.warning("Moderate ASD Risk")
#         else:
#             st.error("High ASD Risk")

#         st.progress(prob)

#         st.write("Recommendation")

#         if prob < 0.30:
#             st.write("Continue developmental monitoring.")
#         elif prob < 0.70:
#             st.write("Further screening recommended.")
#         else:
#             st.write("Consult a developmental specialist.")

# st.markdown("---")
# st.caption("Screening support tool — not a clinical diagnosis.")


# 4th phase (final)
# import streamlit as st
# import pandas as pd
# import joblib
# import shap
# import matplotlib.pyplot as plt

# # Load model
# model = joblib.load("model.pkl")

# st.set_page_config(page_title="Autism Early Screening System", layout="wide")

# st.title("🧠 Autism Early Screening Assistant")
# st.caption("Machine Learning + Explainable AI for Early Autism Screening")

# # Load dataset for analytics
# @st.cache_data
# def load_dataset():
#     return pd.read_csv("dataset.csv")

# data = load_dataset()

# # Tabs
# tab1, tab2, tab3 = st.tabs(["Screening Tool", "Explainable AI", "Dataset Insights"])

# # -------------------------------------------------------
# # TAB 1 — SCREENING TOOL
# # -------------------------------------------------------

# with tab1:

#     col1, col2 = st.columns([1,1])

#     with col1:

#         st.subheader("Child Behaviour Questionnaire")

#         with st.form("screening_form"):

#             a1 = st.selectbox("Responds to name", [0,1])
#             a2 = st.selectbox("Maintains eye contact", [0,1])
#             a3 = st.selectbox("Points to objects", [0,1])
#             a4 = st.selectbox("Imitates actions", [0,1])
#             a5 = st.selectbox("Engages in pretend play", [0,1])
#             a6 = st.selectbox("Follows gaze", [0,1])
#             a7 = st.selectbox("Shows interest in peers", [0,1])
#             a8 = st.selectbox("Brings objects to parents", [0,1])
#             a9 = st.selectbox("Responds to emotions", [0,1])
#             a10 = st.selectbox("Looks at faces", [0,1])

#             submitted = st.form_submit_button("Evaluate ASD Risk")

#     with col2:

#         st.subheader("Screening Result")

#         if submitted:

#             X = pd.DataFrame(
#                 [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]],
#                 columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
#             )

#             probability = model.predict_proba(X)[0][1]

#             st.metric("ASD Risk Probability", f"{probability:.2f}")

#             if probability < 0.30:
#                 st.success("Low ASD Risk")
#                 recommendation = "Continue monitoring developmental milestones."
#             elif probability < 0.70:
#                 st.warning("Moderate ASD Risk")
#                 recommendation = "Further developmental screening recommended."
#             else:
#                 st.error("High ASD Risk")
#                 recommendation = "Clinical evaluation by a specialist recommended."

#             st.progress(probability)

#             st.subheader("Recommendation")
#             st.write(recommendation)

#             # Save last prediction for XAI tab
#             st.session_state["last_input"] = X

#         else:
#             st.info("Complete questionnaire to see screening result.")

# # -------------------------------------------------------
# # TAB 2 — EXPLAINABLE AI
# # -------------------------------------------------------

# with tab2:

#     st.subheader("Explainable AI — Prediction Factors")

#     if "last_input" in st.session_state:

#         X = st.session_state["last_input"]

#         # background dataset
#         X_background = data[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']]

#         explainer = shap.LinearExplainer(model, X_background)
#         shap_values = explainer.shap_values(X)

#         fig, ax = plt.subplots()
#         shap.bar_plot(shap_values[0], feature_names=X.columns)
#         st.pyplot(fig)

#         st.write("This chart shows which behaviours contributed most to the prediction.")

#     else:

#         st.info("Run the screening first to generate explanations.")

# # -------------------------------------------------------
# # TAB 3 — DATASET INSIGHTS
# # -------------------------------------------------------

# with tab3:

#     st.subheader("Dataset Overview")

#     st.write("Dataset size:", data.shape)

#     st.write("Preview of dataset")
#     st.dataframe(data.head())

#     st.subheader("ASD vs Non-ASD Distribution")

#     class_counts = data['Class/ASD Traits '].value_counts()

#     st.bar_chart(class_counts)

#     st.subheader("Question Response Frequency")

#     question_cols = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
#     question_means = data[question_cols].mean()

#     st.bar_chart(question_means)

# st.markdown("---")
# st.caption("This tool is intended for research and screening purposes only and is not a clinical diagnosis.")

# PHASE 5 (final final)
# import streamlit as st
# import pandas as pd
# import joblib
# import shap
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# import plotly.graph_objects as go

# # ---------------- LOAD MODEL ----------------
# model = joblib.load("model.pkl")

# st.set_page_config(page_title="Autism Early Screening Assistant", layout="wide")

# st.title("🧠 Autism Early Screening Assistant")
# st.caption("Machine Learning + Explainable AI for Early Autism Detection")

# # ---------------- LOAD DATA ----------------
# @st.cache_data
# def load_dataset():
#     return pd.read_csv("dataset.csv")

# data = load_dataset()

# # ---------------- TABS ----------------
# tab1, tab2, tab3 = st.tabs([
#     "Screening Tool",
#     "Explainable AI",
#     "Dataset Insights"
# ])

# # =====================================================
# # TAB 1 — SCREENING TOOL
# # =====================================================

# with tab1:

#     col1, col2 = st.columns([1,1])

#     with col1:

#         st.subheader("Child Behaviour Questionnaire")

#         with st.form("screening_form"):

#             a1 = st.selectbox("Responds to name", [0,1])
#             a2 = st.selectbox("Maintains eye contact", [0,1])
#             a3 = st.selectbox("Points to objects", [0,1])
#             a4 = st.selectbox("Imitates actions", [0,1])
#             a5 = st.selectbox("Engages in pretend play", [0,1])
#             a6 = st.selectbox("Follows gaze", [0,1])
#             a7 = st.selectbox("Shows interest in peers", [0,1])
#             a8 = st.selectbox("Brings objects to parents", [0,1])
#             a9 = st.selectbox("Responds to emotions", [0,1])
#             a10 = st.selectbox("Looks at faces", [0,1])

#             submitted = st.form_submit_button("Evaluate ASD Risk")

#     with col2:

#         st.subheader("Screening Result")

#         # if submitted:

#         #     X = pd.DataFrame(
#         #         [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]],
#         #         columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
#         #     )
          
#         #     probability = model.predict_proba(X)[0][1]
#         if submitted:

#             X = pd.DataFrame(
#                 [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]],
#                 columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
#             )

#             probability = model.predict_proba(X)[0][1]

#             # store values globally for other tabs
#             st.session_state["last_input"] = X
#             st.session_state["probability"] = probability
#             st.session_state["prediction_done"] = True

#             # st.metric("ASD Probability", f"{probability:.2f}")
#             st.metric("ASD Probability", f"{probability:.3f}")

#             # -------- Risk Interpretation --------
#             if probability < 0.30:
#                 st.success("Low ASD Risk")
#                 recommendation = "Continue monitoring developmental milestones."
#             elif probability < 0.70:
#                 st.warning("Moderate ASD Risk")
#                 recommendation = "Further developmental screening recommended."
#             else:
#                 st.error("High ASD Risk")
#                 recommendation = "Consult developmental specialist."

#             # -------- Risk Meter --------
#             st.subheader("Risk Level")
#             st.progress(probability)

#             st.write("Recommendation:")
#             st.write(recommendation)
#             # ---------------- BEHAVIOR RADAR CHART ----------------

#             # st.subheader("Behavior Profile")

#             # values = X.iloc[0].values.tolist()

#             # labels = [
#             # "Responds to name",
#             # "Eye contact",
#             # "Pointing",
#             # "Imitation",
#             # "Pretend play",
#             # "Follow gaze",
#             # "Peer interest",
#             # "Show objects",
#             # "Emotion response",
#             # "Face attention"
#             # ]

#             # fig = go.Figure()

#             # fig.add_trace(go.Scatterpolar(
#             #     r=values,
#             #     theta=labels,
#             #     fill='toself',
#             #     name='Child Behavior Profile'
#             # ))

#             # fig.update_layout(
#             #     polar=dict(
#             #         radialaxis=dict(
#             #             visible=True,
#             #             range=[0,1]
#             #         )),
#             #     showlegend=False
#             # )

#             # st.plotly_chart(fig)

#             # ---------------- BEHAVIOR RADAR CHART ----------------

#             # st.subheader("Behavior Profile (1 = Normal, 0 = Concern)")
#             st.subheader("Behavior Profile (1 = Concern, 0 = Typical Behavior)")

#             values = [
#                 int(a1), int(a2), int(a3), int(a4), int(a5),
#                 int(a6), int(a7), int(a8), int(a9), int(a10)
#             ]

#             labels = [
#                 "Responds to name",
#                 "Eye contact",
#                 "Pointing",
#                 "Imitation",
#                 "Pretend play",
#                 "Follow gaze",
#                 "Peer interest",
#                 "Show objects",
#                 "Emotion response",
#                 "Face attention"
#             ]

#             fig = go.Figure()

#             fig.add_trace(go.Scatterpolar(
#                 r=values,
#                 theta=labels,
#                 fill='toself',
#                 name='Behavior Profile'
#             ))

#             fig.update_layout(
#                 polar=dict(
#                     radialaxis=dict(
#                         visible=True,
#                         range=[0,1]
#                     )),
#                 showlegend=False
#             )

#             st.plotly_chart(fig)

#             st.caption("Behavior Score Interpretation: 0 = Typical Behavior, 1 = Potential Concern")

#             # ---------------- AI RECOMMENDATION ENGINE ----------------

#             # st.subheader("AI Behavioral Recommendations")

#             # recommendations = []

#             # if a2 == 0:
#             #     recommendations.append("Encourage eye-contact engagement during play.")

#             # if a3 == 0:
#             #     recommendations.append("Practice pointing and joint attention exercises.")

#             # if a7 == 0:
#             #     recommendations.append("Introduce structured peer interaction activities.")

#             # if a4 == 0:
#             #     recommendations.append("Use imitation games to improve social learning.")

#             # if a9 == 0:
#             #     recommendations.append("Work on recognizing and responding to emotions.")

#             # if recommendations:

#             #     for r in recommendations:
#             #         st.write("•", r)

#             # else:
#             #     st.success("No major behavioural concerns detected.")

#             # st.session_state["last_input"] = X
#             # st.session_state["probability"] = probability

#             # ---------------- AI RECOMMENDATION ENGINE ----------------

#             st.subheader("AI Behavioral Recommendations")

#             recommendations = []

#             # Only trigger recommendations if risk is moderate or high
#             if probability >= 0.30:

#                 if a2 == 1:
#                     recommendations.append("Encourage eye-contact engagement during play.")

#                 if a3 == 1:
#                     recommendations.append("Practice pointing and joint attention exercises.")

#                 if a7 == 1:
#                     recommendations.append("Introduce structured peer interaction activities.")

#                 if a4 == 1:
#                     recommendations.append("Use imitation games to improve social learning.")

#                 if a9 == 1:
#                     recommendations.append("Work on recognizing and responding to emotions.")

#             if recommendations:

#                 for r in recommendations:
#                     st.write("•", r)

#             else:
#                 st.success("No behavioral intervention required based on current screening.")

# # =====================================================
# # TAB 2 — EXPLAINABLE AI
# # =====================================================

# with tab2:

#     st.subheader("Explainable AI — Prediction Factors")

#     # if "last_input" in st.session_state:
#     if st.session_state.get("prediction_done", False):

#         X = st.session_state["last_input"]

#         X_background = data[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']]

#         explainer = shap.LinearExplainer(model, X_background)

#         shap_values = explainer.shap_values(X)

#         fig, ax = plt.subplots()
#         shap.bar_plot(shap_values[0], feature_names=X.columns)
#         st.pyplot(fig)

#         st.write(
#             "The chart shows which behavioural responses influenced the ASD prediction."
#         )

#     else:

#         st.info("Run the screening tool first to generate explanations.")

# # =====================================================
# # TAB 3 — DATASET INSIGHTS
# # =====================================================

# with tab3:

#     st.subheader("Dataset Overview")

#     st.write("Dataset shape:", data.shape)

#     st.dataframe(data.head())

#     st.subheader("ASD Distribution")

#     class_counts = data['Class/ASD Traits '].value_counts()

#     st.bar_chart(class_counts)

#     st.subheader("Question Response Frequency")

#     questions = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']

#     st.bar_chart(data[questions].mean())

# # =====================================================
# # PDF REPORT GENERATOR
# # =====================================================

# if "probability" in st.session_state:

#     st.markdown("---")

#     st.subheader("Download Clinical Screening Report")

#     if st.button("Generate PDF Report"):

#         prob = st.session_state["probability"]

#         pdf = FPDF()
#         pdf.add_page()

#         pdf.set_font("Arial", size=16)
#         pdf.cell(200,10,"Autism Screening Report", ln=True)

#         pdf.set_font("Arial", size=12)

#         pdf.cell(200,10,f"ASD Probability: {prob:.2f}", ln=True)

#         if prob < 0.30:
#             result = "Low Risk"
#         elif prob < 0.70:
#             result = "Moderate Risk"
#         else:
#             result = "High Risk"

#         pdf.cell(200,10,f"Risk Category: {result}", ln=True)

#         pdf.cell(200,10,"Note: This tool is a screening aid and not a medical diagnosis.", ln=True)

#         pdf.output("screening_report.pdf")

#         with open("screening_report.pdf","rb") as file:

#             st.download_button(
#                 label="Download PDF",
#                 data=file,
#                 file_name="screening_report.pdf",
#                 mime="application/pdf"
#             )

# st.markdown("---")
# st.caption("This tool is intended for research and screening purposes only.")


# import streamlit as st
# import pandas as pd
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# # ---------------- LOAD MODEL ----------------
# model = joblib.load("model.pkl")

# st.set_page_config(page_title="Autism Early Screening Assistant", layout="wide")

# st.title("🧠 Autism Early Screening Assistant")
# st.caption("Machine Learning + Explainable AI for Early Autism Detection")

# # ---------------- LOAD DATA ----------------
# @st.cache_data
# def load_dataset():
#     return pd.read_csv("dataset.csv")

# data = load_dataset()

# # ---------------- TABS ----------------
# tab1, tab2, tab3 = st.tabs([
#     "Screening Tool",
#     "Explainable AI",
#     "Dataset Insights"
# ])

# # =====================================================
# # TAB 1 — SCREENING TOOL
# # =====================================================

# with tab1:

#     col1, col2 = st.columns([1,1])

#     with col1:

#         st.subheader("Child Behaviour Questionnaire")

#         a1 = st.selectbox("Responds to name", [0,1])
#         a2 = st.selectbox("Maintains eye contact", [0,1])
#         a3 = st.selectbox("Points to objects", [0,1])
#         a4 = st.selectbox("Imitates actions", [0,1])
#         a5 = st.selectbox("Engages in pretend play", [0,1])
#         a6 = st.selectbox("Follows gaze", [0,1])
#         a7 = st.selectbox("Shows interest in peers", [0,1])
#         a8 = st.selectbox("Brings objects to parents", [0,1])
#         a9 = st.selectbox("Responds to emotions", [0,1])
#         a10 = st.selectbox("Looks at faces", [0,1])

#         evaluate = st.button("Evaluate ASD Risk")

#     with col2:

#         st.subheader("Screening Result")

#         if evaluate:

#             X = pd.DataFrame(
#                 [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]],
#                 columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
#             )

#             probability = model.predict_proba(X)[0][1]

#             # store prediction
#             st.session_state["prediction_done"] = True
#             st.session_state["last_input"] = X
#             st.session_state["probability"] = probability

#         if st.session_state.get("prediction_done", False):

#             probability = st.session_state["probability"]

#             st.metric("ASD Probability", f"{probability:.3f}")

#             if probability < 0.30:
#                 st.success("Low ASD Risk")
#                 recommendation = "Continue monitoring developmental milestones."
#             elif probability < 0.70:
#                 st.warning("Moderate ASD Risk")
#                 recommendation = "Further developmental screening recommended."
#             else:
#                 st.error("High ASD Risk")
#                 recommendation = "Consult developmental specialist."

#             st.subheader("Risk Level")
#             st.progress(probability)

#             st.write("Recommendation:")
#             st.write(recommendation)

#             # ---------------- AI BEHAVIORAL RECOMMENDATIONS ----------------

#             st.subheader("AI Behavioral Recommendations")

#             recommendations = []

#             if a2 == 1:
#                 recommendations.append("Encourage eye-contact engagement during play.")

#             if a3 == 1:
#                 recommendations.append("Practice pointing and joint attention exercises.")

#             if a7 == 1:
#                 recommendations.append("Introduce structured peer interaction activities.")

#             if a4 == 1:
#                 recommendations.append("Use imitation games to improve social learning.")

#             if a9 == 1:
#                 recommendations.append("Work on recognizing and responding to emotions.")

#             if recommendations:
#                 for r in recommendations:
#                     st.write("•", r)
#             else:
#                 st.success("No behavioral intervention required based on current screening.")

#             # ---------------- RADAR CHART ----------------

#             st.subheader("Behavior Profile (1 = Concern, 0 = Typical Behavior)")

#             values = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]

#             labels = [
#                 "Responds to name",
#                 "Eye contact",
#                 "Pointing",
#                 "Imitation",
#                 "Pretend play",
#                 "Follow gaze",
#                 "Peer interest",
#                 "Show objects",
#                 "Emotion response",
#                 "Face attention"
#             ]

#             fig = go.Figure()

#             fig.add_trace(go.Scatterpolar(
#                 r=values,
#                 theta=labels,
#                 fill='toself'
#             ))

#             fig.update_layout(
#                 polar=dict(
#                     radialaxis=dict(
#                         visible=True,
#                         range=[0,1]
#                     )
#                 ),
#                 showlegend=False
#             )

#             st.plotly_chart(fig)

#             # ---------------- PDF REPORT ----------------

#         st.markdown("---")
#         st.subheader("Download Clinical Screening Report")

#         if st.button("Generate PDF Report"):

#             from fpdf import FPDF

#             prob = st.session_state["probability"]

#             pdf = FPDF()
#             pdf.add_page()

#             pdf.set_font("Arial", size=16)
#             pdf.cell(200,10,"Autism Screening Report", ln=True)

#             pdf.set_font("Arial", size=12)
#             pdf.cell(200,10,f"ASD Probability: {prob:.3f}", ln=True)

#             if prob < 0.30:
#                 result = "Low Risk"
#             elif prob < 0.70:
#                 result = "Moderate Risk"
#             else:
#                 result = "High Risk"

#             pdf.cell(200,10,f"Risk Category: {result}", ln=True)

#             pdf.cell(200,10,"Note: This tool is a screening aid and not a medical diagnosis.", ln=True)

#             pdf.output("screening_report.pdf")

#             with open("screening_report.pdf","rb") as file:

#                 st.download_button(
#                     label="Download PDF",
#                     data=file,
#                     file_name="screening_report.pdf",
#                     mime="application/pdf"
#                 )

# # =====================================================
# # TAB 2 — EXPLAINABLE AI
# # =====================================================

# with tab2:

#     st.subheader("Explainable AI — Prediction Factors")

#     if st.session_state.get("prediction_done", False):

#         X = st.session_state["last_input"]

#         X_background = data[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']]

#         explainer = shap.LinearExplainer(model, X_background)

#         shap_values = explainer.shap_values(X)

#         fig, ax = plt.subplots()
#         shap.bar_plot(shap_values[0], feature_names=X.columns)
#         st.pyplot(fig)

#     else:

#         st.info("Run the screening tool first to generate explanations.")

# # =====================================================
# # TAB 3 — DATASET INSIGHTS
# # =====================================================

# with tab3:

#     st.subheader("Dataset Overview")

#     st.write("Dataset shape:", data.shape)

#     st.dataframe(data.head())

#     st.subheader("ASD Distribution")

#     class_counts = data['Class/ASD Traits '].value_counts()

#     st.bar_chart(class_counts)

# st.markdown("---")
# st.caption("This tool is intended for research and screening purposes only.")


import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

st.set_page_config(page_title="Autism Early Screening Assistant", layout="wide")

st.title("🧠 Autism Early Screening Assistant")
st.caption("ML + Explainable AI for Early Autism Detection")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_dataset():
    return pd.read_csv("dataset.csv")

data = load_dataset()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "Screening Tool",
    "Explainable AI",
    "Dataset Insights"
])

# =====================================================
# TAB 1 — SCREENING TOOL
# =====================================================

with tab1:

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("Patient Details")

        name = st.text_input("Child Name")
        age = st.number_input("Age (months)", min_value=1, max_value=60)

        st.subheader("Behaviour Questionnaire (1 = Concern, 0 = Normal)")

        a1 = st.selectbox("Responds to name", [0,1])
        a2 = st.selectbox("Maintains eye contact", [0,1])
        a3 = st.selectbox("Points to objects", [0,1])
        a4 = st.selectbox("Imitates actions", [0,1])
        a5 = st.selectbox("Engages in pretend play", [0,1])
        a6 = st.selectbox("Follows gaze", [0,1])
        a7 = st.selectbox("Shows interest in peers", [0,1])
        a8 = st.selectbox("Brings objects to parents", [0,1])
        a9 = st.selectbox("Responds to emotions", [0,1])
        a10 = st.selectbox("Looks at faces", [0,1])

        evaluate = st.button("Evaluate ASD Risk")

    with col2:

        st.subheader("Screening Result")

        if evaluate:

            X = pd.DataFrame(
                [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]],
                columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']
            )

            probability = model.predict_proba(X)[0][1]

            st.session_state["prediction_done"] = True
            st.session_state["last_input"] = X
            st.session_state["probability"] = probability
            st.session_state["name"] = name
            st.session_state["age"] = age

        if st.session_state.get("prediction_done", False):

            probability = st.session_state["probability"]

            st.metric("ASD Probability", f"{probability:.3f}")

            if probability < 0.30:
                st.success("Low ASD Risk")
                recommendation = "Continue monitoring developmental milestones."
            elif probability < 0.70:
                st.warning("Moderate ASD Risk")
                recommendation = "Further developmental screening recommended."
            else:
                st.error("High ASD Risk")
                recommendation = "Consult developmental specialist."

            st.progress(probability)

            st.write("Recommendation:")
            st.write(recommendation)

            # ---------------- RADAR CHART ----------------
            st.subheader("Behavior Profile")

            values = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]

            labels = [
                "Responds to name","Eye contact","Pointing","Imitation",
                "Pretend play","Follow gaze","Peer interest","Show objects",
                "Emotion response","Face attention"
            ]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself'
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=False
            )

            st.plotly_chart(fig)

            st.caption("1 = Concern, 0 = Typical Behavior")

            # ---------------- AI RECOMMENDATIONS ----------------
            st.subheader("AI Behavioral Recommendations")

            recs = []

            if a2 == 1: recs.append("Encourage eye-contact engagement.")
            if a3 == 1: recs.append("Practice pointing and joint attention.")
            if a7 == 1: recs.append("Introduce peer interaction activities.")
            if a4 == 1: recs.append("Use imitation-based play exercises.")
            if a9 == 1: recs.append("Work on emotional recognition.")

            if recs:
                for r in recs:
                    st.write("•", r)
            else:
                st.success("No major behavioral concerns detected.")

            # ---------------- PDF REPORT ----------------
            st.markdown("---")
            st.subheader("Download Clinical Report")

            if st.button("Generate PDF Report"):

                pdf = FPDF()
                pdf.add_page()

                pdf.set_font("Arial", size=16)
                pdf.cell(200,10,"Autism Screening Report", ln=True)

                pdf.set_font("Arial", size=12)
                pdf.cell(200,10,f"Name: {st.session_state['name']}", ln=True)
                pdf.cell(200,10,f"Age (months): {st.session_state['age']}", ln=True)
                pdf.cell(200,10,f"ASD Probability: {probability:.3f}", ln=True)

                if probability < 0.30:
                    result = "Low Risk"
                elif probability < 0.70:
                    result = "Moderate Risk"
                else:
                    result = "High Risk"

                pdf.cell(200,10,f"Risk Category: {result}", ln=True)

                pdf.cell(200,10,"Note: This is a screening tool, not diagnosis.", ln=True)

                filename = f"{st.session_state['name']}_ASD_Report.pdf"
                pdf.output(filename)

                with open(filename,"rb") as f:
                    st.download_button("Download PDF", f, file_name=filename)

# =====================================================
# TAB 2 — EXPLAINABLE AI
# =====================================================

with tab2:

    st.subheader("Explainable AI")

    if st.session_state.get("prediction_done", False):

        X = st.session_state["last_input"]

        X_background = data[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']]

        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots()
        shap.bar_plot(shap_values[0], feature_names=X.columns)
        st.pyplot(fig)

    else:
        st.info("Run screening first.")

# =====================================================
# TAB 3 — DATASET INSIGHTS
# =====================================================

with tab3:

    st.subheader("Dataset Overview")
    st.write(data.shape)
    st.dataframe(data.head())

    st.subheader("ASD Distribution")
    st.bar_chart(data['Class/ASD Traits '].value_counts())

st.markdown("---")
st.caption("For research use only. Not a clinical diagnosis tool.")
st.subheader("Model Comparison")

# Precomputed results (replace with your actual results)
model_results = {
    "Logistic Regression": 0.92,
    "Random Forest": 0.95,
    "SVM": 0.91
}

df_models = pd.DataFrame(list(model_results.items()), columns=["Model","Accuracy"])

st.bar_chart(df_models.set_index("Model"))

best_model = max(model_results, key=model_results.get)

st.success(f"Best Performing Model: {best_model}")