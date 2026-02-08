"""
Diabetes module fuzzy logic engine.
Implements get_inputs() and run_inference(user_data) per .cursorrules contract.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_inputs() -> List[Dict]:
    """
    Returns a list of dictionaries defining the symptoms needed.
    Must follow the UI input schema contract from .cursorrules.
    """
    return [
        {
            "type": "slider",
            "name": "glucose",
            "label": "Fasting glucose",
            "unit": "mg/dL",
            "min": 70,
            "max": 250,
            "help": "Blood glucose after an overnight fast.",
        },
        {
            "type": "slider",
            "name": "bmi",
            "label": "Body mass index",
            "unit": "kg/m²",
            "min": 15,
            "max": 45,
            "help": "Body mass index (BMI).",
        },
    ]


def _build_fis(glucose_min: float, glucose_max: float, bmi_min: float, bmi_max: float):
    glucose = ctrl.Antecedent(np.linspace(glucose_min, glucose_max, 181), "glucose")
    bmi = ctrl.Antecedent(np.linspace(bmi_min, bmi_max, 151), "bmi")
    risk = ctrl.Consequent(np.linspace(0, 100, 101), "risk")

    # Glucose membership functions (mapped to min/max)
    glucose["low"] = fuzz.trapmf(glucose.universe, [glucose_min, glucose_min, 85, 95])
    glucose["normal"] = fuzz.trimf(glucose.universe, [85, 100, 125])
    glucose["high"] = fuzz.trimf(glucose.universe, [115, 140, 170])
    glucose["very_high"] = fuzz.trapmf(glucose.universe, [160, 190, glucose_max, glucose_max])

    # BMI membership functions (mapped to min/max)
    bmi["underweight"] = fuzz.trapmf(bmi.universe, [bmi_min, bmi_min, 17, 18.5])
    bmi["normal"] = fuzz.trimf(bmi.universe, [18, 22, 25])
    bmi["overweight"] = fuzz.trimf(bmi.universe, [24, 27.5, 30])
    bmi["obese"] = fuzz.trapmf(bmi.universe, [29, 32, bmi_max, bmi_max])

    # Risk membership functions
    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk["medium"] = fuzz.trimf(risk.universe, [30, 50, 70])
    risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(glucose["normal"] & bmi["normal"], risk["low"]),
        ctrl.Rule(glucose["low"], risk["low"]),
        ctrl.Rule(glucose["normal"] & bmi["overweight"], risk["medium"]),
        ctrl.Rule(glucose["normal"] & bmi["obese"], risk["medium"]),
        ctrl.Rule(glucose["high"] & bmi["normal"], risk["medium"]),
        ctrl.Rule(glucose["high"] & bmi["overweight"], risk["medium"]),
        ctrl.Rule(glucose["high"] & bmi["obese"], risk["high"]),
        ctrl.Rule(glucose["very_high"], risk["high"]),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system), glucose, bmi


def _label_from_membership(value: float, antecedent: ctrl.Antecedent) -> str:
    memberships = {}
    for label in antecedent.terms:
        memberships[label] = fuzz.interp_membership(
            antecedent.universe, antecedent[label].mf, value
        )
    return max(memberships, key=memberships.get)


def _risk_level_from_score(score: float) -> str:
    if score < 34:
        return "Low"
    if score < 67:
        return "Medium"
    return "High"


def _recommendation(level: str) -> str:
    if level == "Low":
        return "Maintain a balanced diet, regular exercise, and routine checkups."
    if level == "Medium":
        return "Improve lifestyle habits and consider consulting a clinician for guidance."
    return "Consult a healthcare professional promptly for evaluation."


def run_inference(user_data: Dict) -> Dict:
    """
    Process user_data through Fuzzy Logic and return standardized results.
    """
    inputs = get_inputs()
    glucose_min = inputs[0]["min"]
    glucose_max = inputs[0]["max"]
    bmi_min = inputs[1]["min"]
    bmi_max = inputs[1]["max"]

    glucose_val = float(user_data.get("glucose", glucose_min))
    bmi_val = float(user_data.get("bmi", bmi_min))

    sim, glucose, bmi = _build_fis(glucose_min, glucose_max, bmi_min, bmi_max)
    sim.input["glucose"] = np.clip(glucose_val, glucose_min, glucose_max)
    sim.input["bmi"] = np.clip(bmi_val, bmi_min, bmi_max)
    sim.compute()

    risk_score = float(sim.output["risk"])
    risk_level = _risk_level_from_score(risk_score)

    glucose_label = _label_from_membership(glucose_val, glucose)
    bmi_label = _label_from_membership(bmi_val, bmi)
    reasoning = (
        f"Fasting glucose is {glucose_label.replace('_', ' ')} and BMI is "
        f"{bmi_label.replace('_', ' ')}, leading to a {risk_level.lower()} risk."
    )

    result = {
        "risk_percentage": round(risk_score, 1),
        "risk_level": risk_level,
        "recommendation": _recommendation(risk_level),
        "reasoning": reasoning,
    }
    return result
