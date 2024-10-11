import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]
import Utils_nocturnal_hrv_prediction_summary as Utils

pd.options.mode.chained_assignment = None
np.random.seed(14)

# Streamlit app setup
st.title("Nocturnal HRV Prediction & Suggestions")
st.sidebar.header("Select User")

oura_database = pd.read_csv('database/oura_data/oura_database.csv')
model_stats = pd.read_csv("ML_Models/Model_stats.csv", index_col= 0).reset_index().rename(columns={'index': 'user'})

st.sidebar.header("Select User")
email = st.sidebar.selectbox("Select User", model_stats.email.values.tolist())
user = email.split('@')[0].split('.')[0]
user_model = joblib.load(f'ML_Models/xgboost_{user}.joblib')
user_data = oura_database[(oura_database.email == email)]
X = user_data[user_model.feature_names_in_.tolist()]
duration_cols = [i for i in user_model.feature_names_in_.tolist() if '_time' in i or '_duration' in i]

date = st.selectbox("Select Date", user_data[['day', 'nocturnal_hrv']].sort_values(by= "day", ascending= False).dropna().day.values.tolist())

explainer = shap.TreeExplainer(user_model)
user_data_day = user_data[user_data.day == date]
X_day = user_data_day[user_model.feature_names_in_.tolist()]
shap_values = explainer.shap_values(X_day)
shap_df = pd.DataFrame({'Feature': X_day.columns.to_list(), 'SHAP Value': shap_values[0].tolist()}).sort_values(by=['SHAP Value'], ascending= True)
shap_df['Weights'] = np.abs(shap_df['SHAP Value'])
shap_df['Feature_values'] = [Utils.convert_seconds_to_hhmm(X_day[col].values[0]) if col in duration_cols else f"{X_day[col].values[0]}" for col in shap_df.Feature ]
shap_df.sort_values(by= "Weights", ascending= False, inplace= True)
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values, X_day, matplotlib=True)

st.write(f"Nocturnal HRV: {user_data_day.nocturnal_hrv.values[0]}")
st.write("Features that have the highest impact on the selected day's nocturnal HRV are:")
for col in shap_df.head(5).Feature:
    st.write(col.replace('_', " ").title())

suggestions = Utils.make_suggestions(shap_df, user_data, X_day, user_model, duration_cols)
st.subheader("Suggestions to Improve Nocturnal HRV")
st.write("AI recommendations for improving Nocturnal HRV: ")

recommendations = Utils.generate_recommendations(suggestions)
i = 0
for recommendation in recommendations:
    i+= 1
    st.write(f"{i}. {recommendation}")
nocturnal_hrv_prediction_summary = Utils.create_nocturnal_hrv_prediction_summary(shap_df, recommendations, 6)

Utils.plot_funnel_chart(nocturnal_hrv_prediction_summary)
print(nocturnal_hrv_prediction_summary)
