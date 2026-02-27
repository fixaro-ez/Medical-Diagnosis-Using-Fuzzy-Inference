"""
Respiratory module fuzzy logic engine.
Diagnoses Asthma, Tuberculosis, or Pneumonia using a fuzzy inference approach
with symptom-severity scoring and Gaussian age profiling.

Implements get_inputs() and run_inference(user_data) per template contract.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import skfuzzy as fuzz


# ── Clinical knowledge base (extracted from dataset) ───────────────────
KNOWLEDGE_BASE: Dict = {
    "diseases": {
        "Asthma": {
            "weights": {
                "shortness_of_breath": 0.45,
                "wheezing": 0.25,
                "chest_tightness": 0.15,
                "coughing": 0.15,
            },
            "age_profile": {"mean": 25.5, "std": 18.0},
            "treatment": "Omalizumab / Mepolizumab – use bronchodilators as needed.",
        },
        "Tuberculosis": {
            "weights": {
                "night_sweats": 0.20,
                "chills": 0.20,
                "fever": 0.20,
                "persistent_cough": 0.20,
                "weight_loss": 0.20,
            },
            "age_profile": {"mean": 45.2, "std": 22.0},
            "treatment": "Pyrazinamide / Ethambutol – complete full antibiotic regimen.",
        },
        "Pneumonia": {
            "weights": {
                "fever": 0.35,
                "coughing": 0.30,
                "chills": 0.20,
                "shortness_of_breath": 0.15,
            },
            "age_profile": {"mean": 52.0, "std": 20.0},
            "treatment": "Antibiotics / Oxygen Therapy – seek medical care promptly.",
        },
    },
}


def get_inputs() -> List[Dict]:
    """Symptom-severity sliders plus patient age, grouped with conditional toggles."""
    return [
        {
            "type": "selectbox", "name": "age",
            "label": "Age range", "unit": "years",
            "help": "Patient age range.",
            "options": ["1-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"],
        },
        {
            "type": "toggle",
            "name": "has_cough",
            "label": "Does the patient have a cough?",
            "help": "Whether the patient currently has a cough.",
            "children": [
                {
                    "type": "slider", "name": "coughing",
                    "label": "Coughing bouts per day", "unit": "bouts",
                    "min": 0, "max": 100,
                    "help": "Number of coughing fits per day.",
                },
                {
                    "type": "slider", "name": "persistent_cough",
                    "label": "Weeks with persistent cough", "unit": "weeks",
                    "min": 0, "max": 52,
                    "help": "Number of weeks the cough has persisted.",
                },
            ],
        },
        {
            "type": "toggle",
            "name": "has_breathing_issues",
            "label": "Does the patient have breathing difficulties?",
            "help": "Whether the patient experiences shortness of breath, wheezing, or chest tightness.",
            "children": [
                {
                    "type": "slider", "name": "shortness_of_breath",
                    "label": "Breathlessness episodes per day", "unit": "episodes",
                    "min": 0, "max": 20,
                    "help": "Number of times experiencing difficulty breathing per day.",
                },
                {
                    "type": "slider", "name": "wheezing",
                    "label": "Wheezing episodes per day", "unit": "episodes",
                    "min": 0, "max": 20,
                    "help": "Number of whistling sound episodes while breathing per day.",
                },
                {
                    "type": "slider", "name": "chest_tightness",
                    "label": "Chest tightness episodes per day", "unit": "episodes",
                    "min": 0, "max": 20,
                    "help": "Number of chest tightness or pressure episodes per day.",
                },
            ],
        },
        {
            "type": "toggle",
            "name": "has_fever",
            "label": "Does the patient have a fever?",
            "help": "Whether the patient has an elevated temperature.",
            "children": [
                {
                    "type": "slider", "name": "fever",
                    "label": "Fever temperature", "unit": "°C",
                    "min": 36.0, "max": 41.0,
                    "help": "Current body temperature in Celsius.",
                },
                {
                    "type": "slider", "name": "chills",
                    "label": "Chills episodes per day", "unit": "episodes",
                    "min": 0, "max": 10,
                    "help": "Number of chills / shivering episodes per day.",
                },
            ],
        },
        {
            "type": "toggle",
            "name": "has_systemic_symptoms",
            "label": "Night sweats or unexplained weight loss?",
            "help": "Whether the patient has night sweats or unexplained weight loss.",
            "children": [
                {
                    "type": "slider", "name": "night_sweats",
                    "label": "Night sweats episodes per week", "unit": "episodes",
                    "min": 0, "max": 7,
                    "help": "Number of night sweats episodes per week.",
                },
                {
                    "type": "slider", "name": "weight_loss",
                    "label": "Unintentional weight loss", "unit": "kg",
                    "min": 0, "max": 30,
                    "help": "Unexplained weight loss in kg over the last month.",
                },
            ],
        },
    ]


# ── Fuzzy helpers ───────────────────────────────────────────────────────

_SEVERITY_UNIVERSE = np.linspace(0, 10, 101)
_NONE_MF = fuzz.trapmf(_SEVERITY_UNIVERSE, [0, 0, 0.5, 2])
_LOW_MF = fuzz.trimf(_SEVERITY_UNIVERSE, [1, 3, 5])
_MED_MF = fuzz.trimf(_SEVERITY_UNIVERSE, [4, 6, 8])
_HIGH_MF = fuzz.trapmf(_SEVERITY_UNIVERSE, [7, 8.5, 10, 10])

_SEVERITY_VALUES = {"None": 0.0, "Low": 0.33, "Medium": 0.66, "High": 1.0}


def _defuzzify_severity(value: float, max_val: float) -> float:
    """Convert a raw value to a 0-1 severity via fuzzy membership."""
    universe = np.linspace(0, max_val, 101)
    none_mf = fuzz.trapmf(universe, [0, 0, max_val*0.05, max_val*0.2])
    low_mf = fuzz.trimf(universe, [max_val*0.1, max_val*0.3, max_val*0.5])
    med_mf = fuzz.trimf(universe, [max_val*0.4, max_val*0.6, max_val*0.8])
    high_mf = fuzz.trapmf(universe, [max_val*0.7, max_val*0.85, max_val, max_val])

    memberships = {
        "None": float(fuzz.interp_membership(universe, none_mf, value)),
        "Low": float(fuzz.interp_membership(universe, low_mf, value)),
        "Medium": float(fuzz.interp_membership(universe, med_mf, value)),
        "High": float(fuzz.interp_membership(universe, high_mf, value)),
    }
    total = sum(memberships.values())
    if total == 0:
        return 0.0
    return sum(memberships[k] * _SEVERITY_VALUES[k] for k in memberships) / total


def _severity_label(value: float) -> str:
    """Return the dominant severity label for a 0-10 slider value."""
    memberships = {
        "none": float(fuzz.interp_membership(_SEVERITY_UNIVERSE, _NONE_MF, value)),
        "low": float(fuzz.interp_membership(_SEVERITY_UNIVERSE, _LOW_MF, value)),
        "medium": float(fuzz.interp_membership(_SEVERITY_UNIVERSE, _MED_MF, value)),
        "high": float(fuzz.interp_membership(_SEVERITY_UNIVERSE, _HIGH_MF, value)),
    }
    return max(memberships, key=memberships.get)


def _gaussian_age_membership(x: float, mean: float, std: float) -> float:
    """Gaussian membership function for age profiling."""
    if std == 0:
        return 1.0
    return float(np.exp(-((x - mean) ** 2) / (2 * std ** 2)))


# ── Core inference ──────────────────────────────────────────────────────

def _fuzzy_disease_inference(age: float, symptom_scores: Dict[str, float]) -> List[Dict]:
    """Run disease-specific fuzzy inference for Asthma, TB, and Pneumonia."""
    results: List[Dict] = []
    
    max_vals = {
        "coughing": 100.0,
        "persistent_cough": 52.0,
        "shortness_of_breath": 20.0,
        "wheezing": 20.0,
        "chest_tightness": 20.0,
        "fever": 41.0,
        "chills": 10.0,
        "night_sweats": 7.0,
        "weight_loss": 30.0,
    }

    for disease_name, data in KNOWLEDGE_BASE["diseases"].items():
        raw_score = 0.0
        max_weight = 0.0
        matched_symptoms: List[str] = []

        for symptom, weight in data["weights"].items():
            val = symptom_scores.get(symptom, 0.0)
            if symptom == "fever":
                val = max(0.0, val - 36.0)
                max_val = 5.0
            else:
                max_val = max_vals.get(symptom, 10.0)
            sev = _defuzzify_severity(val, max_val)
            raw_score += weight * sev
            max_weight += weight
            if sev > 0.1:
                matched_symptoms.append(symptom.replace("_", " "))

        symptom_match = raw_score / max_weight if max_weight > 0 else 0.0

        age_match = _gaussian_age_membership(
            age, data["age_profile"]["mean"], data["age_profile"]["std"]
        )

        # Symptom is primary driver (70 %), age profile boosts (30 %)
        strength = symptom_match * (0.7 + 0.3 * age_match) if symptom_match > 0 else 0.0

        results.append({
            "disease": disease_name,
            "strength": strength,
            "treatment": data["treatment"],
            "symptom_match": symptom_match,
            "age_match": age_match,
            "matched_symptoms": matched_symptoms,
        })

    # Normalize to percentages
    total = sum(r["strength"] for r in results)
    for r in results:
        r["percentage"] = (r["strength"] / total * 100.0) if total > 0 else 0.0

    results.sort(key=lambda x: x["percentage"], reverse=True)
    return results


def _risk_level(score: float) -> str:
    if score < 34:
        return "Low"
    if score < 67:
        return "Medium"
    return "High"


# ── Public API ──────────────────────────────────────────────────────────

def run_inference(user_data: Dict) -> Dict:
    """Process patient data through fuzzy inference. Returns standardised result dict."""
    age = float(user_data.get("age", 30))

    symptom_names = [
        "coughing", "persistent_cough", "shortness_of_breath", "wheezing",
        "chest_tightness", "fever", "chills", "night_sweats", "weight_loss",
    ]
    symptom_scores = {name: float(user_data.get(name, 0)) for name in symptom_names}

    # Early exit when no symptoms are reported
    if all(v == 0 for v in symptom_scores.values()):
        return {
            "risk_percentage": 0.0,
            "risk_level": "Low",
            "recommendation": "No symptoms reported. Continue regular health monitoring.",
            "reasoning": "No respiratory symptoms were indicated.",
            "rule_trace": [],
            "plain_summary": "No respiratory risk pattern detected because no symptoms were entered.",
        }

    results = _fuzzy_disease_inference(age, symptom_scores)
    top = results[0]

    # Overall respiratory risk = top disease absolute strength (0-100)
    risk_pct = round(min(top["strength"] * 100.0, 100.0), 1)
    level = _risk_level(risk_pct)

    # Build human-readable reasoning
    matched = top["matched_symptoms"]
    symptom_text = ", ".join(matched) if matched else "mild symptoms"
    disease_summary = ", ".join(
        f"{r['disease']} {r['percentage']:.1f}%" for r in results if r["percentage"] > 0
    )
    reasoning = (
        f"Based on {symptom_text}, the most likely condition is {top['disease']} "
        f"({top['percentage']:.1f}%). Full distribution: {disease_summary}."
    )

    # Recommendation includes treatment and runner-up if close
    recommendation = top["treatment"]
    if len(results) > 1 and results[1]["percentage"] > 25:
        recommendation += (
            f" Also consider {results[1]['disease']} ({results[1]['percentage']:.1f}%)."
        )

    all_scores = {r["disease"]: round(r["percentage"], 1) for r in results}

    top_weights = KNOWLEDGE_BASE["diseases"][top["disease"]]["weights"]
    driver_rows = []
    
    max_vals = {
        "coughing": 100.0,
        "persistent_cough": 52.0,
        "shortness_of_breath": 20.0,
        "wheezing": 20.0,
        "chest_tightness": 20.0,
        "fever": 41.0,
        "chills": 10.0,
        "night_sweats": 7.0,
        "weight_loss": 30.0,
    }

    for symptom, weight in top_weights.items():
        val = symptom_scores.get(symptom, 0.0)
        if symptom == "fever":
            val = max(0.0, val - 36.0)
            max_val = 5.0
        else:
            max_val = max_vals.get(symptom, 10.0)
        severity = _defuzzify_severity(val, max_val)
        strength = float(weight * severity)
        if strength > 0:
            driver_rows.append(
                {
                    "rule": f"IF {symptom.replace('_', ' ')} is elevated THEN {top['disease']} probability increases",
                    "strength": round(min(1.0, strength), 2),
                }
            )
    driver_rows.sort(key=lambda item: item["strength"], reverse=True)
    rule_trace = driver_rows[:4]

    plain_summary = (
        f"Respiratory assessment suggests {top['disease']} as the leading condition "
        f"with overall {level.lower()} risk at {risk_pct:.1f}%."
    )

    return {
        "risk_percentage": risk_pct,
        "risk_level": level,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "all_scores": all_scores,
        "rule_trace": rule_trace,
        "plain_summary": plain_summary,
    }
