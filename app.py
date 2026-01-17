# ===========================
# Streamlit Predictive Maintenance App
# ===========================

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Create dataset
# ---------------------------
data = {
    'Temperature': [77, 80, 90, 70, 85, 95, 65, 79, 92, 88],
    'Vibration': [0.5, 0.7, 0.9, 0.4, 0.8, 1.0, 0.3, 0.6, 0.95, 0.85],
    'Pressure': [30, 35, 40, 24, 38, 42, 22, 33, 41, 39],
    'MaintenanceNeeded': [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# ---------------------------
# 2. Train model
# ---------------------------
X = df[['Temperature', 'Vibration', 'Pressure']]
y = df['MaintenanceNeeded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# 3. Rule-based + AI logic
# ---------------------------
def decision_system(temp, vib, pres):
    # Rule-based thresholds
    rule_maintenance = False
    if temp >= 85 or vib >= 0.8 or pres >= 42:
        rule_maintenance = True

    # AI prediction
    new_data = pd.DataFrame({
        'Temperature': [80],
        'Vibration': [0.5],
        'Pressure': [30]
    })
    probability = model.predict_proba(new_data)[0][1]
    ai_maintenance = probability >= 0.5

    # Final decision
    final_decision = rule_maintenance or ai_maintenance

    return final_decision, probability, rule_maintenance

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("Predictive Maintenance System")

temp = st.number_input("Temperature", value=80.0)
vib = st.number_input("Vibration", value=0.5)
pres = st.number_input("Pressure", value=30.0)

if st.button("Check Maintenance"):
    final_decision, prob, rule_flag = decision_system(temp, vib, pres)

    st.write(f"🔍 **Maintenance Probability:** {prob:.2f}")

    if rule_flag:
        st.write("⚠️ **Rule Triggered: Extreme condition detected**")

    if final_decision:
        st.write("✅ **Decision: Maintenance Needed**")
    else:
        st.write("✅ **Decision: No Maintenance Needed**")
