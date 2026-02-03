# main.py simplified concept
import streamlit as st
from modules.respiratory import engine as respiratory

# Map the modules
diseases = {
    "Respiratory": respiratory,
    # "Heart": heart, ... (Teammates add their modules here)
}

selected = st.selectbox("Select Disease", list(diseases.keys()))
module = diseases[selected]

# 1. Automatically build the UI based on module's inputs
user_inputs = {}
for item in module.get_inputs():
    user_inputs[item['name']] = st.slider(item['label'], item['min'], item['max'])

# 2. Run the logic
if st.button("Diagnose"):
    results = module.run_inference(user_inputs)
    st.write(f"Risk: {results['risk_level']} ({results['risk_percentage']}%)")


if __name__ == "__main__":
    print("Welcome to the Medical Diagnosis System!")
    # TODO: Implement main menu logic
