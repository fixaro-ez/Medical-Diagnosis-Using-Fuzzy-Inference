import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="RespiDetect AI",
    page_icon="🫁",
    layout="wide"
)


@st.cache_data
def load_and_train_model(csv_file):
    """
    Parses the CSV to build a fuzzy knowledge base:
    1. Calculates symptom weights (frequency of symptom per disease).
    2. Calculates age statistics (mean/std) for fuzzy age matching.
    3. Maps treatments to diseases.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Clean column names and string values
    df.columns = [c.strip() for c in df.columns]
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    # Handle Age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    knowledge_base = {
        "symptoms_list": sorted(df['Symptoms'].unique().tolist()),
        "diseases": {}
    }

    unique_diseases = df['Disease'].unique()

    for disease in unique_diseases:
        subset = df[df['Disease'] == disease]
        total_cases = len(subset)
        
        # A. Symptom Weights
        # Weight = (Count of Symptom in Disease) / (Total Cases of Disease)
        symptom_counts = subset['Symptoms'].value_counts()
        weights = (symptom_counts / total_cases).to_dict()
        
        # B. Age Profile (Gaussian parameters)
        valid_ages = subset['Age'].dropna()
        if len(valid_ages) > 1:
            age_stats = {
                "mean": float(valid_ages.mean()),
                "std": float(valid_ages.std())
            }
        elif len(valid_ages) == 1:
            age_stats = {"mean": float(valid_ages.iloc[0]), "std": 5.0} # Fallback std
        else:
            # Fallback if no age data exists for this disease
            age_stats = {"mean": 40.0, "std": 20.0}

        # C. Treatment (Most common)
        try:
            treatment = subset['Treatment'].mode()[0]
        except:
            treatment = "Consult a Doctor"

        knowledge_base["diseases"][disease] = {
            "weights": weights,
            "age_profile": age_stats,
            "treatment": treatment
        }
        
    return knowledge_base



def calculate_gaussian_membership(x, mean, std):
    """
    Returns a value between 0.0 and 1.0 indicating how close 'x' (user age)
    is to the 'mean' (disease typical age).
    """
    if std == 0: return 1.0
    exponent = -((x - mean) ** 2) / (2 * (std ** 2))
    return np.exp(exponent)

def fuzzy_inference(user_age, user_symptoms, kb):
    """
    Core Logic:
    1. Fuzzify Symptoms (Low=0.33, Med=0.66, High=1.0)
    2. Fuzzify Age (Gaussian Match)
    3. Inference: Score = Symptom_Match * Age_Booster
    """
    SEVERITY_MAP = {"None": 0.0, "Low": 0.33, "Medium": 0.66, "High": 1.0}
    
    results = []
    
    for disease_name, data in kb["diseases"].items():
        # Step A: Calculate Symptom Match Score
        raw_symptom_score = 0.0
        max_possible_weight = 0.0
        
        disease_weights = data["weights"]
        
        for symptom, weight in disease_weights.items():
            user_severity_label = user_symptoms.get(symptom, "None")
            user_val = SEVERITY_MAP[user_severity_label]
            
            raw_symptom_score += (weight * user_val)
            max_possible_weight += weight
            
        
        symptom_match = 0.0
        if max_possible_weight > 0:
            symptom_match = raw_symptom_score / max_possible_weight
            
      
        age_match = calculate_gaussian_membership(
            user_age, 
            data["age_profile"]["mean"], 
            data["age_profile"]["std"]
        )
        
       
        final_strength = 0.0
        if symptom_match > 0:
            final_strength = symptom_match * (0.7 + (0.3 * age_match))
            
        results.append({
            "Disease": disease_name,
            "Probability": final_strength,
            "Treatment": data["treatment"],
            "Details": {
                "Symptom Match": symptom_match,
                "Age Match": age_match
            }
        })
        

    total_score = sum(r["Probability"] for r in results)
    
    final_results = []
    for r in results:
        percentage = (r["Probability"] / total_score * 100) if total_score > 0 else 0
        final_results.append({
            **r,
            "Percentage": percentage
        })
        
  
    final_results.sort(key=lambda x: x["Percentage"], reverse=True)
    return final_results


def main():
    st.title("Respiratory Detection")
    st.markdown("""
    This system uses **Fuzzy Logic** to diagnose respiratory diseases based on your dataset.
    It learns patterns (symptom weights and age profiles) directly from the CSV file.
    """)
    

    with st.sidebar:
        st.header("1. Data Source")
        uploaded_file = st.file_uploader("Upload 'respiratory symptoms.csv'", type=["csv"])
        
       
        if not uploaded_file:
            st.warning("Please upload the dataset to begin training the Fuzzy System.")
            
            try:
                
                kb = load_and_train_model("respiratory symptoms and treatment.csv")
                st.success("Found local dataset! Model trained.")
            except:
                kb = None
        else:
            kb = load_and_train_model(uploaded_file)
            st.success("Model trained successfully!")

    if kb:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("2. Patient Profile")
            age = st.slider("Patient Age", 1, 100, 30)
            sex = st.radio("Patient Sex", ["Male", "Female"])
            
            st.subheader("Symptoms")
            st.info("Select severity for symptoms you are experiencing.")
            
            user_symptoms = {}
            
            # Group symptoms or just list them
            # To save space, we use a multiselect to pick WHICH symptoms, then ask severity
            # OR we just list common ones. Given the dataset is small, let's list all.
            
            with st.expander("Select Symptoms", expanded=True):
                for symptom in kb["symptoms_list"]:
                    severity = st.select_slider(
                        label=f"{symptom}",
                        options=["None", "Low", "Medium", "High"],
                        value="None",
                        key=symptom
                    )
                    user_symptoms[symptom] = severity
        
        with col2:
            st.header("3. Diagnosis Results")
            
            if st.button("Run Fuzzy Analysis", type="primary", use_container_width=True):
                results = fuzzy_inference(age, user_symptoms, kb)
                
                # Filter out zero probability
                active_results = [r for r in results if r["Percentage"] > 0]
                
                if not active_results:
                    st.warning("No disease matches found based on current symptoms.")
                else:
                    top_match = active_results[0]
                    
                    # Highlight Top Result
                    st.success(f"### Most Likely: **{top_match['Disease']}** ({top_match['Percentage']:.1f}%)")
                    st.write(f"**Recommended Treatment:** {top_match['Treatment']}")
                    
                    st.divider()
                    st.subheader("Detailed Breakdown")
                    
                    for r in active_results:
                        with st.container():
                            c_name, c_bar, c_val = st.columns([2, 4, 1])
                            c_name.write(f"**{r['Disease']}**")
                            c_bar.progress(int(r['Percentage']))
                            c_val.write(f"{r['Percentage']:.1f}%")
                            
                            with st.expander("Logic Details"):
                                st.write(f"- Symptom Match Score: {r['Details']['Symptom Match']:.2f} (0-1)")
                                st.write(f"- Age Profile Match: {r['Details']['Age Match']:.2f} (0-1)")
                                st.caption("The final score is a fuzzy combination of symptom severity matches weighted by historical data, boosted by how well the patient's age fits the disease profile.")

    else:
        st.info("👈 Waiting for dataset upload...")

if __name__ == "__main__":
    main()