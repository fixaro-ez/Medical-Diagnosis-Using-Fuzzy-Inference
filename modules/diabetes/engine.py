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
            "name": "bmi",
            "label": "Body mass index",
            "unit": "kg/m²",
            "min": round(stats["ranges"]["BMI"][0], 1),
            "max": round(stats["ranges"]["BMI"][1], 1),
            "help": "Body mass index.",
        },
        {
            "type": "slider",
            "name": "age",
            "label": "Age",
            "unit": "years",
            "min": int(stats["ranges"]["Age"][0]),
            "max": int(stats["ranges"]["Age"][1]),
            "help": "Patient age in years.",
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
            "type": "slider",
            "name": "diabetes_pedigree",
            "label": "Diabetes pedigree function",
            "unit": "score",
            "min": round(stats["ranges"]["DiabetesPedigreeFunction"][0], 3),
            "max": round(stats["ranges"]["DiabetesPedigreeFunction"][1], 3),
            "help": "Family-history related diabetes risk score.",
        },
    ]


@lru_cache(maxsize=1)
def _dataset_stats() -> Dict:
    data_path = Path(__file__).resolve().parent / "data" / "diabetes.csv"
    df = pd.read_csv(data_path)

    for col in ["Glucose", "BloodPressure", "BMI", "Insulin"]:
        df.loc[df[col] == 0, col] = np.nan

    positive = df[df["Outcome"] == 1]
    negative = df[df["Outcome"] == 0]

    features = ["Glucose", "BMI", "Insulin", "DiabetesPedigreeFunction", "Age"]

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
    bmi = ctrl.Antecedent(np.linspace(ranges["BMI"][0], ranges["BMI"][1], 200), "bmi")
    age = ctrl.Antecedent(np.linspace(ranges["Age"][0], ranges["Age"][1], 140), "age")
    insulin = ctrl.Antecedent(np.linspace(ranges["Insulin"][0], ranges["Insulin"][1], 240), "insulin")
    pedigree = ctrl.Antecedent(
        np.linspace(ranges["DiabetesPedigreeFunction"][0], ranges["DiabetesPedigreeFunction"][1], 220),
        "diabetes_pedigree",
    )

    _add_memberships(glucose, *quantiles["Glucose"], *ranges["Glucose"])
    _add_memberships(bmi, *quantiles["BMI"], *ranges["BMI"])
    _add_memberships(age, *quantiles["Age"], *ranges["Age"])
    _add_memberships(insulin, *quantiles["Insulin"], *ranges["Insulin"])
    _add_memberships(
        pedigree,
        *quantiles["DiabetesPedigreeFunction"],
        *ranges["DiabetesPedigreeFunction"],
    )

    risk = ctrl.Consequent(np.linspace(0, 100, 101), "risk")
    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 40])
    risk["medium"] = fuzz.trimf(risk.universe, [30, 50, 70])
    risk["high"] = fuzz.trimf(risk.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(glucose["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & bmi["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & insulin["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & pedigree["high"] & age["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & insulin["medium"] & bmi["high"], risk["high"]),
        ctrl.Rule(glucose["medium"] & bmi["medium"] & insulin["medium"], risk["medium"]),
        ctrl.Rule(bmi["high"] & pedigree["high"], risk["medium"]),
        ctrl.Rule(insulin["high"] & bmi["high"], risk["medium"]),
        ctrl.Rule(age["high"] & pedigree["medium"], risk["medium"]),
        ctrl.Rule(glucose["low"] & bmi["low"] & insulin["low"], risk["low"]),
        ctrl.Rule(glucose["low"] & pedigree["low"], risk["low"]),
    ]

    system = ctrl.ControlSystem(rules)
    return (
        ctrl.ControlSystemSimulation(system),
        {
            "glucose": glucose,
            "bmi": bmi,
            "age": age,
            "insulin": insulin,
            "diabetes_pedigree": pedigree,
        },
    )


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
    stats = _dataset_stats()
    ranges = stats["ranges"]

    values = {
        "glucose": float(user_data.get("glucose", ranges["Glucose"][0])),
        "bmi": float(user_data.get("bmi", ranges["BMI"][0])),
        "age": float(user_data.get("age", ranges["Age"][0])),
        "insulin": float(user_data.get("insulin", ranges["Insulin"][0])),
        "diabetes_pedigree": float(
            user_data.get("diabetes_pedigree", ranges["DiabetesPedigreeFunction"][0])
        ),
    }

    clip_map = {
        "glucose": ranges["Glucose"],
        "bmi": ranges["BMI"],
        "age": ranges["Age"],
        "insulin": ranges["Insulin"],
        "diabetes_pedigree": ranges["DiabetesPedigreeFunction"],
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
    bmi_label = _label_from_membership(values["bmi"], antecedents["bmi"])
    age_label = _label_from_membership(values["age"], antecedents["age"])
    insulin_label = _label_from_membership(values["insulin"], antecedents["insulin"])
    pedigree_label = _label_from_membership(
        values["diabetes_pedigree"], antecedents["diabetes_pedigree"]
    )
    reasoning = (
        f"Fasting glucose is {glucose_label.replace('_', ' ')} and BMI is "
        f"{bmi_label.replace('_', ' ')}; insulin is {insulin_label}, age is {age_label}, "
        f"and pedigree risk is {pedigree_label}. Combined fuzzy rules indicate a {risk_level.lower()} risk."
    )

    result = {
        "risk_percentage": round(risk_score, 1),
        "risk_level": risk_level,
        "recommendation": _recommendation(risk_level),
        "reasoning": reasoning,
    }
    return result
