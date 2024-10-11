import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]

pd.options.mode.chained_assignment = None
np.random.seed(14)

def convert_seconds_to_hhmm(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{int(hours):02}:{int(minutes):02}"

def make_suggestions(shap_df, user_data, X_day, user_model):
    suggestions = []
    for feature_name in shap_df.head(5).Feature:
        feature_value = X_day[feature_name].values[0]
        simulated_values = {}
        simulated_values['p75_value'] = user_data[feature_name].describe()['75%']
        simulated_values['p50_value'] = user_data[feature_name].describe()['50%']
        simulated_values['p25_value'] = user_data[feature_name].describe()['25%']
        p_value = 0
        flag = 0
        for i in [75, 50, 25]:
            X_day_simulated = X_day.copy()
            X_day_simulated[feature_name] = simulated_values[f'p{i}_value']
            predicted_hrv = user_model.predict(X_day)[0]
            simulated_hrv = user_model.predict(X_day_simulated)[0]
            improvement = np.round(simulated_hrv - predicted_hrv, 0)
            if improvement > flag:
                flag = improvement
                p_value = i
                target_value = simulated_values[f'p{p_value}_value']
        if flag > 0:
            if feature_value > target_value:
                direction = "reducing"
            else: direction = "increasing"
            if feature_name in duration_cols:
                feature_value = convert_seconds_to_hhmm(feature_value)
                target_value = convert_seconds_to_hhmm(target_value)
            else:
                feature_value = f'{feature_value:.0f}'
                target_value = f'{target_value:.0f}'
            suggestion = f"Try {direction} your {feature_name} from {feature_value} to {target_value} to improve the Nocturnal HRV by {flag:.0f} bpm."
            suggestions.append(suggestion)
    return suggestions

def openai_response(prompt, model = "gpt-4o"):
    client = OpenAI(api_key= openai_api_key.replace('"',''))
    response = client.chat.completions.create(
        model= model,
        messages=[ {"role": "user", "content": f"{prompt}"}]
        )
    return response.choices[0].message.content.strip()

def create_nocturnal_hrv_prediction_summary(shap_df, feature_count):
    temp_shap_df = shap_df.head(feature_count)
    features = [i.replace('_', ' ').title() for i in temp_shap_df.Feature.tolist()[::-1]]
    weights = temp_shap_df.Weights.tolist()[::-1]
    contributions = [(value / sum(weights)) * 100 for value in weights]
    values = temp_shap_df.Feature_values.tolist()[::-1]
    nocturnal_hrv_prediction_summary = {
    "features": features, 
    "weights": weights,
    "contributions": contributions, 
    "values": values,
    "recommendation": recommendations
    }
    return nocturnal_hrv_prediction_summary

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
shap_df['Feature_values'] = [convert_seconds_to_hhmm(X_day[col].values[0]) if col in duration_cols else f"{X_day[col].values[0]}" for col in shap_df.Feature ]
shap_df.sort_values(by= "Weights", ascending= False, inplace= True)
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values, X_day, matplotlib=True)

st.write(f"Nocturnal HRV: {user_data_day.nocturnal_hrv.values[0]}")
st.write("Features that have the highest impact on the selected day's nocturnal HRV are:")
for col in shap_df.head(5).Feature:
    st.write(col.replace('_', " ").title())

suggestions = make_suggestions(shap_df, user_data, X_day, user_model)
st.subheader("Suggestions to Improve Nocturnal HRV")
i = 0
recommendations = []
st.write("AI recommendations for improving Nocturnal HRV: ")
for suggestion in suggestions:
    i += 1
    prompt = f"To increase my nocturnal HRV during sleep, I was given the following piece of advice: {suggestion}. Can you suggest in a 2-3 sentences some ways I can achieve this through easy changes in my behaviour. Mention the impact it can have on nocturnal hrv by changing the current value to the suggested value. Structure it like a recommendation starting with something like you can. Also call out the improvement/ reduction that needs to be made in the metric"
    recommendation = openai_response(prompt)
    recommendations.append(recommendation)
    st.write(f"{i}. {recommendation}")

nocturnal_hrv_prediction_summary = create_nocturnal_hrv_prediction_summary(shap_df, 6)
# Create a bar chart
fig = go.Figure(go.Funnel(
    y=nocturnal_hrv_prediction_summary['features'],  # Feature names on the y-axis
    x=nocturnal_hrv_prediction_summary['weights'],  # SHAP values on the x-axis
    textinfo="text",  # Display value and percentage of the initial value
    text=[f"{percent:.0f}%" for percent in nocturnal_hrv_prediction_summary['contributions']],
    marker={"color": "blue"}  # You can customize the color here
))
# Update layout for better readability
fig.update_layout(
    title="Feature Contributions Funnel (Top 6 SHAP Values)",
    funnelmode="stack"  # Stacks the funnel sections
)
# Show the plot
st.plotly_chart(fig)

print(nocturnal_hrv_prediction_summary)
