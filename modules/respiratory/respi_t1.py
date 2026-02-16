import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Respiratory Detection ",
    ##page_icon="🫁",
    layout="wide"
)


KNOWLEDGE_BASE = {
    "symptoms_list": [
        "A cough that lasts more than three weeks",
        "Chills",
        "Fever",
        "Loss of appetite and unintentional weight loss",
        "Night sweats",
        "coughing",
        "shortness of breath",
        "tight feeling in the chest",
        "wheezing"
    ],
    "diseases": {
        "Asthma": {
            "weights": {
                "shortness of breath": 0.45,
                "wheezing": 0.25,
                "tight feeling in the chest": 0.15,
                "coughing": 0.15
            },
            "age_profile": {"mean": 25.5, "std": 18.0},
            "treatment": "Omalizumab / Mepolizumab"
        },
        "Tuberculosis": {
            "weights": {
                "Night sweats": 0.20,
                "Chills": 0.20,
                "Fever": 0.20,
                "A cough that lasts more than three weeks": 0.20,
                "Loss of appetite and unintentional weight loss": 0.20
            },
            "age_profile": {"mean": 45.2, "std": 22.0},
            "treatment": "Pyrazinamide / Ethambutol"
        },
        "Pneumonia": {
            "weights": {
                "Fever": 0.35,
                "coughing": 0.30,
                "Chills": 0.20,
                "shortness of breath": 0.15
            },
            "age_profile": {"mean": 52.0, "std": 20.0},
            "treatment": "Antibiotics / Oxygen Therapy"
        }
    }
}


def calculate_gaussian_membership(x, mean, std):
    """Calculates how well the age fits the disease profile."""
    if std == 0: return 1.0
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    return np.exp(exponent)

def fuzzy_inference(user_age, user_symptoms, kb):
    """
    1. Fuzzify Symptoms (Labels -> Values)
    2. Fuzzify Age (Gaussian membership)
    3. Rule Evaluation & Normalization
    """
    SEVERITY_MAP = {"None": 0.0, "Low": 0.33, "Medium": 0.66, "High": 1.0}
    results = []
    
    for disease_name, data in kb["diseases"].items():
        # Symptom Match
        raw_symptom_score = 0.0
        max_possible_weight = 0.0
        for symptom, weight in data["weights"].items():
            user_val = SEVERITY_MAP[user_symptoms.get(symptom, "None")]
            raw_symptom_score += (weight * user_val)
            max_possible_weight += weight
            
        symptom_match = raw_symptom_score / max_possible_weight if max_possible_weight > 0 else 0
            
        
        age_match = calculate_gaussian_membership(user_age, data["age_profile"]["mean"], data["age_profile"]["std"])
        

        final_strength = symptom_match * (0.7 + (0.3 * age_match)) if symptom_match > 0 else 0
            
        results.append({
            "Disease": disease_name,
            "Probability": final_strength,
            "Treatment": data["treatment"],
            "Details": {"Symptom Match": symptom_match, "Age Match": age_match}
        })
        

    total_score = sum(r["Probability"] for r in results)
    for r in results:
        r["Percentage"] = (r["Probability"] / total_score * 100) if total_score > 0 else 0
        
    return sorted(results, key=lambda x: x["Percentage"], reverse=True)


def main():
    st.title("🫁 RespiDetect AI: Pre-trained Fuzzy System")
    st.info("System is using the clinical knowledge base extracted from your dataset.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Patient Data")
        age = st.slider("Age", 1, 100, 30)
        sex = st.radio("Sex", ["Male", "Female"])
        
        st.subheader("Symptom Severity")
        user_symptoms = {}
        with st.expander("Checklist", expanded=True):
            for symptom in KNOWLEDGE_BASE["symptoms_list"]:
                user_symptoms[symptom] = st.select_slider(
                    label=f"{symptom.capitalize()}",
                    options=["None", "Low", "Medium", "High"],
                    key=f"in_{symptom}"
                )
    
    with col2:
        st.header("Analysis")
        if st.button("Generate Diagnosis", type="primary", use_container_width=True):
            results = fuzzy_inference(age, user_symptoms, KNOWLEDGE_BASE)
            active = [r for r in results if r["Percentage"] > 0]
            
            if not active:
                st.warning("Please provide symptom details to see a diagnosis.")
            else:
                top = active[0]
                st.success(f"### Likely Condition: {top['Disease']}")
                st.metric("Confidence Level", f"{top['Percentage']:.1f}%")
                st.info(f"**Suggested Treatment:** {top['Treatment']}")
                
                st.divider()
                st.subheader("Full Probability Distribution")
                for r in active:
                    st.write(f"**{r['Disease']}** ({r['Percentage']:.1f}%)")
                    st.progress(int(r['Percentage']))

if __name__ == "__main__":
    main()