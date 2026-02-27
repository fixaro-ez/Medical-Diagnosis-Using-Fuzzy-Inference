"""
Diabetes module fuzzy logic engine.
Implements get_inputs() and run_inference(user_data) per .cursorrules contract.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_inputs() -> List[Dict]:
    stats = _dataset_stats()
    return [
        {
            "type": "slider",
            "name": "glucose",
            "label": "Fasting glucose",
            "unit": "mg/dL",
            "min": int(stats["ranges"]["Glucose"][0]),
            "max": int(stats["ranges"]["Glucose"][1]),
            "help": "Plasma glucose concentration.",
        },
        {
            "type": "slider",
            "name": "blood_pressure",
            "label": "Diastolic blood pressure",
            "unit": "mmHg",
            "min": int(stats["ranges"]["BloodPressure"][0]),
            "max": int(stats["ranges"]["BloodPressure"][1]),
            "help": "Diastolic blood pressure (mmHg).",
        },
        {
            "type": "slider",
            "name": "skin_thickness",
            "label": "Skin fold thickness",
            "unit": "mm",
            "min": int(stats["ranges"]["SkinThickness"][0]),
            "max": int(stats["ranges"]["SkinThickness"][1]),
            "help": "Triceps skin fold thickness (mm).",
        },
        {
            "type": "slider",
            "name": "bmi",
            "label": "Body mass index",
            "unit": "kg/m²",
            "min": round(stats["ranges"]["BMI"][0], 1),
            "max": round(stats["ranges"]["BMI"][1], 1),
            "help": "Body mass index.",
        },
        {
            "type": "selectbox",
            "name": "age",
            "label": "Age range",
            "unit": "years",
            "help": "Patient age range.",
            "options": ["1-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"],
        },
        {
            "type": "slider",
            "name": "insulin",
            "label": "Insulin",
            "unit": "mu U/ml",
            "min": int(stats["ranges"]["Insulin"][0]),
            "max": int(stats["ranges"]["Insulin"][1]),
            "help": "2-hour serum insulin level.",
        },
        {
            "type": "toggle",
            "name": "has_been_pregnant",
            "label": "Have you been pregnant?",
            "help": "Whether the patient has had any pregnancies.",
            "children": [
                {
                    "type": "slider",
                    "name": "pregnancies",
                    "label": "Number of pregnancies",
                    "unit": "",
                    "min": int(stats["ranges"]["Pregnancies"][0]),
                    "max": int(stats["ranges"]["Pregnancies"][1]),
                    "help": "Number of times pregnant.",
                },
            ],
        },
        {
            "type": "toggle",
            "name": "has_family_history",
            "label": "Family history of diabetes?",
            "help": "Whether any immediate family members have diabetes.",
            "children": [
                {
                    "type": "slider",
                    "name": "family_members_with_diabetes",
                    "label": "Family members with diabetes",
                    "unit": "people",
                    "min": 0,
                    "max": 10,
                    "help": "Number of immediate family members with diabetes.",
                },
            ],
        },
    ]


@lru_cache(maxsize=1)
def _dataset_stats() -> Dict:
    data_path = Path(__file__).resolve().parent / "data" / "diabetes.csv"
    df = pd.read_csv(data_path)

    for col in ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]:
        df.loc[df[col] == 0, col] = np.nan

    positive = df[df["Outcome"] == 1]
    negative = df[df["Outcome"] == 0]

    features = ["Glucose", "BMI", "Insulin", "Pregnancies", "Age", "BloodPressure", "SkinThickness"]

    quantiles = {}
    ranges = {}
    means = {}
    for col in features:
        valid = df[col].dropna()
        q1, q2, q3 = valid.quantile([0.25, 0.5, 0.75]).tolist()
        quantiles[col] = (float(q1), float(q2), float(q3))
        ranges[col] = (float(valid.min()), float(valid.max()))
        means[col] = {
            "positive": float(positive[col].dropna().mean()),
            "negative": float(negative[col].dropna().mean()),
        }

    return {"quantiles": quantiles, "ranges": ranges, "means": means}


def _add_memberships(ant: ctrl.Antecedent, q1: float, q2: float, q3: float, vmin: float, vmax: float):
    ant["low"] = fuzz.trapmf(ant.universe, [vmin, vmin, q1, q2])
    ant["medium"] = fuzz.trimf(ant.universe, [q1, q2, q3])
    ant["high"] = fuzz.trapmf(ant.universe, [q2, q3, vmax, vmax])


def _build_fis(stats: Dict):
    ranges = stats["ranges"]
    quantiles = stats["quantiles"]

    glucose = ctrl.Antecedent(np.linspace(ranges["Glucose"][0], ranges["Glucose"][1], 220), "glucose")
    blood_pressure = ctrl.Antecedent(np.linspace(ranges["BloodPressure"][0], ranges["BloodPressure"][1], 150), "blood_pressure")
    skin_thickness = ctrl.Antecedent(np.linspace(ranges["SkinThickness"][0], ranges["SkinThickness"][1], 100), "skin_thickness")
    bmi = ctrl.Antecedent(np.linspace(ranges["BMI"][0], ranges["BMI"][1], 200), "bmi")
    age = ctrl.Antecedent(np.linspace(ranges["Age"][0], ranges["Age"][1], 140), "age")
    insulin = ctrl.Antecedent(np.linspace(ranges["Insulin"][0], ranges["Insulin"][1], 240), "insulin")
    pregnancies = ctrl.Antecedent(np.linspace(ranges["Pregnancies"][0], ranges["Pregnancies"][1], 100), "pregnancies")
    family_members = ctrl.Antecedent(np.linspace(0, 10, 11), "family_members_with_diabetes")

    _add_memberships(glucose, *quantiles["Glucose"], *ranges["Glucose"])
    _add_memberships(blood_pressure, *quantiles["BloodPressure"], *ranges["BloodPressure"])
    _add_memberships(skin_thickness, *quantiles["SkinThickness"], *ranges["SkinThickness"])
    _add_memberships(bmi, *quantiles["BMI"], *ranges["BMI"])
    _add_memberships(age, *quantiles["Age"], *ranges["Age"])
    _add_memberships(insulin, *quantiles["Insulin"], *ranges["Insulin"])
    _add_memberships(pregnancies, *quantiles["Pregnancies"], *ranges["Pregnancies"])
    
    family_members["low"] = fuzz.trimf(family_members.universe, [0, 0, 2])
    family_members["medium"] = fuzz.trimf(family_members.universe, [1, 3, 5])
    family_members["high"] = fuzz.trimf(family_members.universe, [4, 10, 10])

    risk = ctrl.Consequent(np.linspace(0, 100, 101), "risk")
    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk["medium"] = fuzz.trimf(risk.universe, [30, 50, 70])
    risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(glucose["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & bmi["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & insulin["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & family_members["high"] & age["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & insulin["medium"] & bmi["high"], risk["high"]),
        ctrl.Rule(glucose["high"] & blood_pressure["high"], risk["high"]),
        ctrl.Rule(bmi["high"] & blood_pressure["high"] & age["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & bmi["medium"] & insulin["medium"], risk["medium"]),
        ctrl.Rule(bmi["high"] & family_members["high"], risk["medium"]),
        ctrl.Rule(insulin["high"] & bmi["high"], risk["medium"]),
        ctrl.Rule(age["high"] & family_members["medium"], risk["medium"]),
        ctrl.Rule(blood_pressure["high"] & bmi["medium"], risk["medium"]),
        ctrl.Rule(pregnancies["high"] & glucose["medium"], risk["medium"]),
        ctrl.Rule(skin_thickness["high"] & bmi["high"], risk["medium"]),
        ctrl.Rule(glucose["low"] & bmi["low"] & insulin["low"], risk["low"]),
        ctrl.Rule(glucose["low"] & family_members["low"], risk["low"]),
        ctrl.Rule(blood_pressure["low"] & glucose["low"], risk["low"]),
        ctrl.Rule(pregnancies["low"] & glucose["low"] & bmi["low"], risk["low"]),
    ]

    system = ctrl.ControlSystem(rules)
    return (
        ctrl.ControlSystemSimulation(system),
        {
            "glucose": glucose,
            "blood_pressure": blood_pressure,
            "skin_thickness": skin_thickness,
            "bmi": bmi,
            "age": age,
            "insulin": insulin,
            "pregnancies": pregnancies,
            "family_members_with_diabetes": family_members,
        },
    )


def _label_from_membership(value: float, antecedent: ctrl.Antecedent) -> str:
    memberships = {}
    for label in antecedent.terms:
        memberships[label] = fuzz.interp_membership(
            antecedent.universe, antecedent[label].mf, value
        )
    return max(memberships, key=memberships.get)


def _membership_strength(value: float, antecedent: ctrl.Antecedent, label: str) -> float:
    return float(fuzz.interp_membership(antecedent.universe, antecedent[label].mf, value))


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
    stats = _dataset_stats()
    ranges = stats["ranges"]

    values = {
        "glucose": float(user_data.get("glucose", ranges["Glucose"][0])),
        "blood_pressure": float(user_data.get("blood_pressure", ranges["BloodPressure"][0])),
        "skin_thickness": float(user_data.get("skin_thickness", ranges["SkinThickness"][0])),
        "bmi": float(user_data.get("bmi", ranges["BMI"][0])),
        "age": float(user_data.get("age", ranges["Age"][0])),
        "insulin": float(user_data.get("insulin", ranges["Insulin"][0])),
        "pregnancies": float(user_data.get("pregnancies", 0)),
        "family_members_with_diabetes": float(user_data.get("family_members_with_diabetes", 0)),
    }

    clip_map = {
        "glucose": ranges["Glucose"],
        "blood_pressure": ranges["BloodPressure"],
        "skin_thickness": ranges["SkinThickness"],
        "bmi": ranges["BMI"],
        "age": ranges["Age"],
        "insulin": ranges["Insulin"],
        "pregnancies": ranges["Pregnancies"],
        "family_members_with_diabetes": (0, 10),
    }
    for key, (low, high) in clip_map.items():
        values[key] = float(np.clip(values[key], low, high))

    sim, antecedents = _build_fis(stats)
    for key, value in values.items():
        sim.input[key] = value
    sim.compute()

    risk_score = float(sim.output["risk"])
    risk_level = _risk_level_from_score(risk_score)

    glucose_label = _label_from_membership(values["glucose"], antecedents["glucose"])
    bp_label = _label_from_membership(values["blood_pressure"], antecedents["blood_pressure"])
    bmi_label = _label_from_membership(values["bmi"], antecedents["bmi"])
    age_label = _label_from_membership(values["age"], antecedents["age"])
    insulin_label = _label_from_membership(values["insulin"], antecedents["insulin"])
    pregnancies_label = _label_from_membership(values["pregnancies"], antecedents["pregnancies"])
    family_members_label = _label_from_membership(
        values["family_members_with_diabetes"], antecedents["family_members_with_diabetes"]
    )
    reasoning = (
        f"Fasting glucose is {glucose_label.replace('_', ' ')} and BMI is "
        f"{bmi_label.replace('_', ' ')}; blood pressure is {bp_label}, insulin is {insulin_label}, age is {age_label}, "
        f"pregnancies are {pregnancies_label}, and family history risk is {family_members_label}. Combined fuzzy rules indicate a {risk_level.lower()} risk."
    )

    glucose_high = _membership_strength(values["glucose"], antecedents["glucose"], "high")
    glucose_medium = _membership_strength(values["glucose"], antecedents["glucose"], "medium")
    bmi_high = _membership_strength(values["bmi"], antecedents["bmi"], "high")
    insulin_high = _membership_strength(values["insulin"], antecedents["insulin"], "high")
    bp_high = _membership_strength(values["blood_pressure"], antecedents["blood_pressure"], "high")
    family_members_high = _membership_strength(values["family_members_with_diabetes"], antecedents["family_members_with_diabetes"], "high")
    age_high = _membership_strength(values["age"], antecedents["age"], "high")

    rule_trace = [
        {
            "rule": "IF glucose is high THEN diabetes risk is high",
            "strength": round(glucose_high, 2),
        },
        {
            "rule": "IF glucose is medium AND BMI is high THEN diabetes risk is high",
            "strength": round(float(min(glucose_medium, bmi_high)), 2),
        },
        {
            "rule": "IF glucose is medium AND insulin is high THEN diabetes risk is high",
            "strength": round(float(min(glucose_medium, insulin_high)), 2),
        },
        {
            "rule": "IF glucose is high AND blood pressure is high THEN risk is high",
            "strength": round(float(min(glucose_high, bp_high)), 2),
        },
        {
            "rule": "IF glucose is medium AND family history is high AND age is high THEN risk is high",
            "strength": round(float(min(glucose_medium, family_members_high, age_high)), 2),
        },
    ]
    rule_trace = [item for item in rule_trace if item["strength"] > 0]

    plain_summary = (
        f"Diabetes risk is {risk_level.lower()} at {round(risk_score, 1)}%. "
        f"Glucose and metabolic indicators are the primary contributors in this profile."
    )

    result = {
        "risk_percentage": round(risk_score, 1),
        "risk_level": risk_level,
        "recommendation": _recommendation(risk_level),
        "reasoning": reasoning,
        "rule_trace": rule_trace,
        "plain_summary": plain_summary,
    }
    return result
