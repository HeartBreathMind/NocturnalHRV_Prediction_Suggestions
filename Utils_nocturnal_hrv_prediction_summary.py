import pandas as pd
import numpy as np
import shap
import joblib
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]

def convert_seconds_to_hhmm(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{int(hours):02}:{int(minutes):02}"

def make_suggestions(shap_df, user_data, X_day, user_model, duration_cols):
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

def create_nocturnal_hrv_prediction_summary(shap_df, recommendations, feature_count):
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

def plot_funnel_chart(nocturnal_hrv_prediction_summary):
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

def generate_recommendations(suggestions):
    recommendations = []
    for suggestion in suggestions:
        prompt = f"To increase my nocturnal HRV during sleep, I was given the following piece of advice: {suggestion}. Can you suggest in a 2-3 sentences some ways I can achieve this through easy changes in my behaviour. Mention the impact it can have on nocturnal hrv by changing the current value to the suggested value. Structure it like a recommendation starting with something like you can. Also call out the improvement/ reduction that needs to be made in the metric"
        recommendation = openai_response(prompt)
        recommendations.append(recommendation)
    return recommendations
