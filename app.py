# ===========================
# Streamlit Predictive Maintenance App
# With Color Zones + Scatter Plot
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Dataset (Historical)
# ---------------------------
data = {
    'Temperature': [77, 80, 90, 70, 85, 95, 65, 79, 92, 88],
    'Vibration': [0.5, 0.7, 0.9, 0.4, 0.8, 1.0, 0.3, 0.6, 0.95, 0.85],
    'Pressure': [30, 35, 40, 24, 38, 42, 22, 33, 41, 39],
    'MaintenanceNeeded': [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# ---------------------------
# 2. Train AI Model
# ---------------------------
X = df[['Temperature', 'Vibration', 'Pressure']]
y = df['MaintenanceNeeded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# 3. Risk Scoring (Color Zones)
# ---------------------------
def risk_score(temp, vib, pres):
    temp_score = min(temp / 84, 1)
    vib_score = min(vib / 0.8, 1)
    pres_score = min(pres / 32, 1)
    return (temp_score + vib_score + pres_score) / 3

def risk_zone(score):
    if score <= 0.66:
        return "SAFE", "green"
    elif score <= 0.78:
        return "WARNING", "orange"
    else:
        return "DANGER", "red"

# ---------------------------
# 4. Decision System
# ---------------------------
def decision_system(temp, vib, pres):
    rule_trigger = (temp >= 80) or (vib >= 0.6) or (pres >= 29)

    new_data = pd.DataFrame({
        'Temperature': [temp],
        'Vibration': [vib],
        'Pressure': [pres]
    })

    probability = model.predict_proba(new_data)[0][1]
    ai_trigger = probability >= 0.5

    final_decision = rule_trigger or ai_trigger
    return final_decision, probability, rule_trigger

# ---------------------------
# 5. Scatter Plot
# ---------------------------
def plot_scatter(temp, vib):
    fig, ax = plt.subplots()

    ax.scatter(
        df['Temperature'],
        df['Vibration'],
        c=df['MaintenanceNeeded'],
        cmap='coolwarm',
        s=80,
        label='Historical'
    )

    ax.scatter(temp, vib, color='black', marker='X', s=150, label='Current')

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Vibration")
    ax.set_title("Temperature vs Vibration Scatter Plot")
    ax.legend()

    st.pyplot(fig)

# ---------------------------
# 6. Risk Zone Bar
# ---------------------------
def plot_risk_bar(score):
    fig, ax = plt.subplots()

    ax.barh([0], [1], color='lightgrey')
    ax.barh([0], [score], color=risk_zone(score)[1])

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title("Overall Machine Risk Level")

    st.pyplot(fig)

# ---------------------------
# 7. Streamlit UI
# ---------------------------
st.title("🔧 Predictive Maintenance System")

temp = st.number_input("Temperature", value=80.0)
vib = st.number_input("Vibration", value=0.5)
pres = st.number_input("Pressure", value=30.0)

if st.button("Check Maintenance"):
    decision, prob, rule_flag = decision_system(temp, vib, pres)

    score = risk_score(temp, vib, pres)
    zone, color = risk_zone(score)

    st.subheader("📊 Results")
    st.write(f"**Maintenance Probability:** {prob:.2f}")
    st.write(f"**Risk Zone:** :{color}[{zone}]")

    if rule_flag:
        st.warning("Rule-based threshold triggered")

    if decision:
        st.error("Maintenance Needed")
    else:
        st.success("No Maintenance Needed")

    st.subheader("📈 Risk Visualization")
    plot_risk_bar(score)

    st.subheader("🔍 Scatter Analysis")
    plot_scatter(temp, vib)
