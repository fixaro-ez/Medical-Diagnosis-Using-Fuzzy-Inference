"""
BLUEPRINT FOR DISEASE MODULES
Every teammate must implement these two functions into their specific engine.py
"""

def get_inputs():
    """
    Returns a list of dictionaries defining the symptoms needed.
    AI Rules: Only use 'name', 'label', 'min', and 'max'.
    """
    return [
        {"name": "symptom_1", "label": "Degree of Cough", "min": 0, "max": 10},
        {"name": "symptom_2", "label": "Fever Temperature", "min": 35, "max": 42},
    ]

def run_inference(user_data):
    """
    Logic: Process the user_data (dict) through Fuzzy Logic.
    Output: Must return a dict with 'risk' and 'message'.
    """
    # 1. AI: Initialize your Antecedents and Consequents here
    # 2. AI: Apply your fuzzy rules
    
    # Mock result format
    result = {
        "risk_percentage": 75.5,
        "risk_level": "High",
        "recommendation": "Consult a specialist immediately."
    }
    return result