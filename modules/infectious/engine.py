from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -------------------------
# Dropdown mapping
# -------------------------
LEVELS = ["None", "Mild", "Moderate", "Severe"]
LEVEL_MAP = {"None": 0.0, "Mild": 3.0, "Moderate": 6.0, "Severe": 9.0}


@lru_cache(maxsize=1)
def _dataset_calibration() -> dict:
    """Read infectious.csv and derive calibration thresholds for fuzzy sets/scores."""
    data_path = Path(__file__).resolve().parent / "data" / "infectious.csv"
    if not data_path.exists():
        return {
            "temp_q": (37.2, 38.2, 39.1),
            "temp_min": 36.0,
            "temp_max": 40.0,
            "age_q": (26.0, 51.0, 76.0),
            "score_center": 52.0,
            "score_contrast": 1.12,
        }

    df = pd.read_csv(data_path)
    temp = pd.to_numeric(df.get("Temperature"), errors="coerce").dropna()
    age = pd.to_numeric(df.get("Age"), errors="coerce").dropna()

    tq1, tq2, tq3 = temp.quantile([0.25, 0.50, 0.75]).tolist()
    aq1, aq2, aq3 = age.quantile([0.25, 0.50, 0.75]).tolist()

    severity_map = {"Normal": 20.0, "Mild Fever": 45.0, "High Fever": 78.0}
    sev = df.get("Fever_Severity", pd.Series(dtype="object")).map(severity_map).dropna()
    score_center = float(sev.mean()) if not sev.empty else 52.0

    # Keep moderate contrast so results are not over-concentrated at ~50,
    # while avoiding over-extreme 0/100 outputs.
    score_contrast = float(np.clip(1.03 + (float(temp.std()) / 18.0), 1.05, 1.2))

    return {
        "temp_q": (float(tq1), float(tq2), float(tq3)),
        "temp_min": float(temp.min()),
        "temp_max": float(temp.max()),
        "age_q": (float(aq1), float(aq2), float(aq3)),
        "score_center": score_center,
        "score_contrast": score_contrast,
    }


def get_inputs():
    """
    Used by main.py to auto-build UI later.
    """
    return [
        {
            "type": "slider",
            "name": "age",
            "label": "Age",
            "unit": "years",
            "min": 0,
            "max": 100,
            "default": 30,
            "help": "Patient age in years.",
        },
        {
            "type": "slider",
            "name": "temp_f",
            "label": "Temperature",
            "unit": "°F",
            "min": 95.0,
            "max": 105.8,
            "default": 100.4,
            "help": "Current body temperature (Fahrenheit).",
        },
        {
            "type": "slider",
            "name": "duration_days",
            "label": "Duration",
            "unit": "days",
            "min": 0,
            "max": 14,
            "default": 2,
            "help": "Symptom duration in days.",
        },
        {
            "type": "selectbox",
            "name": "cough",
            "label": "Cough severity",
            "unit": "",
            "options": LEVELS,
            "help": "Respiratory cough intensity.",
        },
        {
            "type": "selectbox",
            "name": "diarrhea",
            "label": "Diarrhea severity",
            "unit": "",
            "options": LEVELS,
            "help": "Gastrointestinal diarrhea intensity.",
        },
        {
            "type": "selectbox",
            "name": "diff_breath",
            "label": "Breathing difficulty",
            "unit": "",
            "options": LEVELS,
            "help": "Difficulty breathing severity.",
        },
    ]


def _symptom_sets(x):
    """Fuzzy sets for 0..9 symptom scale derived from dropdown."""
    x["none"] = fuzz.trapmf(x.universe, [0.0, 0.0, 1.0, 2.0])
    x["mild"] = fuzz.trimf(x.universe, [1.0, 3.0, 5.0])
    x["moderate"] = fuzz.trimf(x.universe, [4.0, 6.0, 8.0])
    x["severe"] = fuzz.trapmf(x.universe, [7.0, 8.0, 9.0, 9.0])


def _risk_sets(out):
    """Fuzzy sets for output risk 0..100 – wide overlapping trapezoids for smooth defuzzification."""
    out["low"] = fuzz.trapmf(out.universe, [0, 0, 20, 45])
    out["medium"] = fuzz.trapmf(out.universe, [20, 40, 60, 80])
    out["high"] = fuzz.trapmf(out.universe, [55, 80, 100, 100])


def _de_bias_score(score: float, center: float, contrast: float) -> float:
    """Light nudge away from center; keeps output inside 5-95 range."""
    shifted = center + ((score - center) * contrast)
    return float(np.clip(shifted, 5.0, 95.0))


def _fahrenheit_to_celsius(temp_f: float) -> float:
    return (temp_f - 32.0) * (5.0 / 9.0)


def _build_sim():
    calib = _dataset_calibration()

    # ---------- Inputs ----------
    age = ctrl.Antecedent(np.arange(0, 101, 1), "age")
    temp = ctrl.Antecedent(np.arange(35.0, 41.1, 0.1), "temp")
    duration = ctrl.Antecedent(np.arange(0.0, 14.1, 0.1), "duration")

    cough = ctrl.Antecedent(np.arange(0.0, 9.1, 0.1), "cough")
    diarrhea = ctrl.Antecedent(np.arange(0.0, 9.1, 0.1), "diarrhea")
    diff_breath = ctrl.Antecedent(np.arange(0.0, 9.1, 0.1), "diff_breath")

    # ---------- Outputs ----------
    viral = ctrl.Consequent(np.arange(0.0, 100.1, 1.0), "viral")
    flu = ctrl.Consequent(np.arange(0.0, 100.1, 1.0), "flu")
    gastro = ctrl.Consequent(np.arange(0.0, 100.1, 1.0), "gastro")
    highrisk = ctrl.Consequent(np.arange(0.0, 100.1, 1.0), "highrisk")

    # ---------- Age membership ----------
    age_q1, age_q2, age_q3 = calib["age_q"]
    age["child"] = fuzz.trapmf(age.universe, [0, 0, max(8, age_q1 * 0.45), max(14, age_q1 * 0.7)])
    age["adult"] = fuzz.trapmf(age.universe, [max(12, age_q1 * 0.6), max(18, age_q1), age_q3, min(90, age_q3 + 8)])
    age["elderly"] = fuzz.trapmf(age.universe, [max(55, age_q3 - 8), age_q3, 100, 100])

    # ---------- Temperature membership ----------
    t_q1, t_q2, t_q3 = calib["temp_q"]
    t_min = calib["temp_min"]
    t_max = calib["temp_max"]
    temp["normal"] = fuzz.trapmf(temp.universe, [t_min, t_min, t_q1 - 0.25, t_q1 + 0.05])
    temp["lowgrade"] = fuzz.trapmf(temp.universe, [t_q1 - 0.1, t_q1 + 0.05, t_q2 - 0.15, t_q2 + 0.1])
    temp["high"] = fuzz.trapmf(temp.universe, [t_q2 - 0.15, t_q2 + 0.15, t_q3 + 0.2, t_max])
    temp["veryhigh"] = fuzz.trapmf(temp.universe, [t_q3, t_q3 + 0.25, 41.0, 41.0])

    # ---------- Duration membership ----------
    duration["acute"] = fuzz.trapmf(duration.universe, [0.0, 0.0, 2.5, 3.5])
    duration["persistent"] = fuzz.trapmf(duration.universe, [2.0, 3.0, 6.5, 7.5])
    duration["prolonged"] = fuzz.trapmf(duration.universe, [6.0, 7.0, 14.0, 14.0])

    # ---------- Symptom membership (0..9) ----------
    for s in (cough, diarrhea, diff_breath):
        _symptom_sets(s)

    # ---------- Output membership ----------
    for out in (viral, flu, gastro, highrisk):
        _risk_sets(out)

    # ---------- Rules ----------
    # Graduated rules with clinically reasonable escalation.
    # Each condition has LOW / MEDIUM / HIGH rules covering the full input space.

    rules = []

    # ================================================================
    # VIRAL FEVER  –  typical: mild-to-moderate fever, short duration,
    # minimal respiratory / GI symptoms
    # ================================================================
    # HIGH: genuine high fever isolated, or persistent low-grade fever without other symptoms
    rules.append(ctrl.Rule(temp["high"] & cough["none"] & diarrhea["none"], viral["high"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & duration["persistent"] & cough["none"] & diarrhea["none"], viral["high"]))

    # Low-grade fever + short + no symptoms = medium only (not alarming)
    rules.append(ctrl.Rule(temp["lowgrade"] & duration["acute"] & cough["none"] & diarrhea["none"], viral["medium"]))

    # MEDIUM: low-grade fever with mild accompaniments, or borderline presentation
    rules.append(ctrl.Rule(temp["lowgrade"] & cough["mild"] & diarrhea["none"], viral["medium"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & cough["none"] & diarrhea["mild"], viral["medium"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & cough["mild"] & diarrhea["mild"], viral["medium"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & duration["prolonged"], viral["medium"]))
    rules.append(ctrl.Rule(temp["normal"] & cough["mild"] & diarrhea["none"] & diff_breath["none"], viral["medium"]))

    # LOW: no fever / strong other-condition signature
    rules.append(ctrl.Rule(temp["normal"] & cough["none"] & diarrhea["none"], viral["low"]))
    rules.append(ctrl.Rule(temp["normal"] & cough["none"] & diarrhea["mild"], viral["low"]))
    rules.append(ctrl.Rule(temp["high"] & cough["severe"], viral["low"]))
    rules.append(ctrl.Rule(temp["high"] & cough["moderate"], viral["low"]))
    rules.append(ctrl.Rule(diarrhea["severe"], viral["low"]))
    rules.append(ctrl.Rule(diarrhea["moderate"] & cough["none"], viral["low"]))

    # ================================================================
    # FLU-LIKE ILLNESS  –  high fever + significant cough / respiratory
    # ================================================================
    # HIGH: fever ≥ high + moderate-to-severe cough
    rules.append(ctrl.Rule(temp["high"] & cough["severe"], flu["high"]))
    rules.append(ctrl.Rule(temp["high"] & cough["moderate"], flu["high"]))
    rules.append(ctrl.Rule(temp["veryhigh"] & cough["moderate"], flu["high"]))
    rules.append(ctrl.Rule(temp["veryhigh"] & cough["severe"], flu["high"]))

    # MEDIUM: lower-fever with cough, or severe cough without fever
    rules.append(ctrl.Rule(temp["lowgrade"] & cough["moderate"], flu["medium"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & cough["severe"], flu["medium"]))
    rules.append(ctrl.Rule(temp["high"] & cough["mild"], flu["medium"]))
    rules.append(ctrl.Rule(temp["normal"] & cough["severe"], flu["medium"]))
    rules.append(ctrl.Rule(temp["normal"] & cough["moderate"], flu["medium"]))  # persistent cough even w/o fever
    rules.append(ctrl.Rule(temp["normal"] & cough["moderate"] & diff_breath["moderate"], flu["medium"]))

    # LOW: no/mild cough or gastro-dominant
    rules.append(ctrl.Rule(temp["normal"] & cough["none"], flu["low"]))
    rules.append(ctrl.Rule(temp["normal"] & cough["mild"] & diarrhea["none"], flu["low"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & cough["none"], flu["low"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & cough["mild"] & diarrhea["none"], flu["low"]))
    rules.append(ctrl.Rule(diarrhea["severe"] & cough["none"], flu["low"]))

    # ================================================================
    # GASTROENTERITIS  –  diarrhea dominates; fever optional
    # ================================================================
    # HIGH: severe diarrhea
    rules.append(ctrl.Rule(diarrhea["severe"] & cough["none"], gastro["high"]))
    rules.append(ctrl.Rule(diarrhea["severe"] & temp["normal"], gastro["high"]))
    rules.append(ctrl.Rule(diarrhea["severe"] & temp["lowgrade"], gastro["high"]))

    # MEDIUM: moderate diarrhea
    rules.append(ctrl.Rule(diarrhea["moderate"] & cough["none"], gastro["medium"]))
    rules.append(ctrl.Rule(diarrhea["moderate"] & cough["mild"], gastro["medium"]))
    rules.append(ctrl.Rule(diarrhea["moderate"] & temp["normal"], gastro["medium"]))
    rules.append(ctrl.Rule(diarrhea["moderate"] & temp["lowgrade"], gastro["medium"]))
    rules.append(ctrl.Rule(diarrhea["mild"] & cough["none"] & temp["lowgrade"], gastro["medium"]))
    rules.append(ctrl.Rule(diarrhea["severe"] & cough["moderate"], gastro["medium"]))

    # LOW: no/mild diarrhea or cough-dominant
    rules.append(ctrl.Rule(diarrhea["none"], gastro["low"]))
    rules.append(ctrl.Rule(diarrhea["mild"] & cough["moderate"], gastro["low"]))
    rules.append(ctrl.Rule(diarrhea["mild"] & cough["severe"], gastro["low"]))
    rules.append(ctrl.Rule(diarrhea["mild"] & diarrhea["mild"], gastro["low"]))

    # ================================================================
    # HIGH-RISK FEVER  –  red-flag signs: very high temp, breathing
    # difficulty, prolonged illness, vulnerable age groups
    # ================================================================
    # HIGH: clear danger signs
    rules.append(ctrl.Rule(temp["veryhigh"], highrisk["high"]))  # very high temp alone is always dangerous
    rules.append(ctrl.Rule(temp["veryhigh"] & diff_breath["severe"], highrisk["high"]))
    rules.append(ctrl.Rule(temp["veryhigh"] & diff_breath["moderate"], highrisk["high"]))
    rules.append(ctrl.Rule(temp["veryhigh"] & duration["prolonged"], highrisk["high"]))
    rules.append(ctrl.Rule(temp["high"] & diff_breath["severe"], highrisk["high"]))
    rules.append(ctrl.Rule(diff_breath["severe"] & duration["prolonged"], highrisk["high"]))
    rules.append(ctrl.Rule(age["child"] & temp["veryhigh"], highrisk["high"]))
    rules.append(ctrl.Rule(age["child"] & diff_breath["severe"], highrisk["high"]))
    rules.append(ctrl.Rule(age["elderly"] & temp["veryhigh"], highrisk["high"]))
    rules.append(ctrl.Rule(age["elderly"] & diff_breath["severe"], highrisk["high"]))
    rules.append(ctrl.Rule(age["elderly"] & temp["high"] & diff_breath["moderate"], highrisk["high"]))

    # MEDIUM: concerning but not critical
    rules.append(ctrl.Rule(temp["veryhigh"] & diff_breath["none"], highrisk["medium"]))
    rules.append(ctrl.Rule(temp["high"] & diff_breath["moderate"], highrisk["medium"]))
    rules.append(ctrl.Rule(temp["high"] & duration["prolonged"], highrisk["medium"]))
    rules.append(ctrl.Rule(diff_breath["moderate"] & temp["lowgrade"], highrisk["medium"]))
    rules.append(ctrl.Rule(diff_breath["moderate"] & duration["persistent"], highrisk["medium"]))
    rules.append(ctrl.Rule(diff_breath["severe"] & temp["normal"], highrisk["medium"]))
    rules.append(ctrl.Rule(age["child"] & temp["high"], highrisk["medium"]))
    rules.append(ctrl.Rule(age["elderly"] & temp["high"], highrisk["medium"]))
    rules.append(ctrl.Rule(age["child"] & diff_breath["moderate"], highrisk["medium"]))
    rules.append(ctrl.Rule(age["elderly"] & diff_breath["moderate"], highrisk["medium"]))
    rules.append(ctrl.Rule(duration["prolonged"] & temp["lowgrade"], highrisk["medium"]))

    # LOW: stable / benign presentation
    rules.append(ctrl.Rule(temp["normal"] & diff_breath["none"], highrisk["low"]))
    rules.append(ctrl.Rule(temp["normal"] & diff_breath["mild"], highrisk["low"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & diff_breath["none"], highrisk["low"]))
    rules.append(ctrl.Rule(temp["lowgrade"] & diff_breath["mild"], highrisk["low"]))
    rules.append(ctrl.Rule(age["adult"] & temp["lowgrade"] & diff_breath["none"], highrisk["low"]))
    rules.append(ctrl.Rule(temp["high"] & diff_breath["none"] & duration["acute"], highrisk["low"]))

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def run_inference(user_inputs: dict) -> dict:
    """
    Standard interface for your team's main.py.

    Returns:
      risk_level, risk_percentage, all_scores, red_flag, high_risk_score
    """
    sim = _build_sim()

    # Clamp numeric inputs
    age_years = min(max(float(user_inputs["age"]), 0.0), 100.0)
    if "temp_f" in user_inputs:
        temp_f = min(max(float(user_inputs["temp_f"]), 95.0), 105.8)
        temp_c = _fahrenheit_to_celsius(temp_f)
    else:
        temp_c = min(max(float(user_inputs["temp_c"]), 35.0), 41.0)
    duration_days = min(max(float(user_inputs["duration_days"]), 0.0), 14.0)

    def m(key: str) -> float:
        v = str(user_inputs[key])
        if v not in LEVEL_MAP:
            raise ValueError(f"{key} must be one of {LEVELS}, got {v!r}")
        return LEVEL_MAP[v]

    # Set inputs
    sim.input["age"] = age_years
    sim.input["temp"] = temp_c
    sim.input["duration"] = duration_days
    sim.input["cough"] = m("cough")
    sim.input["diarrhea"] = m("diarrhea")
    sim.input["diff_breath"] = m("diff_breath")

    sim.compute()

    scores = {
        "Viral Fever": float(sim.output.get("viral", 0.0)),
        "Flu-like Illness": float(sim.output.get("flu", 0.0)),
        "Gastroenteritis": float(sim.output.get("gastro", 0.0)),
        "High-Risk Fever": float(sim.output.get("highrisk", 0.0)),
    }

    calib = _dataset_calibration()
    adjusted_scores = {
        key: _de_bias_score(value, calib["score_center"], calib["score_contrast"])
        for key, value in scores.items()
    }

    # Safety-first decision
    if adjusted_scores["High-Risk Fever"] >= 55:
        top_name = "High-Risk Fever"
        top_score = adjusted_scores["High-Risk Fever"]
        level = "High"
    else:
        illness_only = {k: v for k, v in adjusted_scores.items() if k != "High-Risk Fever"}
        top_name = max(illness_only, key=illness_only.get)
        top_score = illness_only[top_name]

        if top_score < 30:
            level = "Low"
        elif top_score < 60:
            level = "Medium"
        else:
            level = "High"

    return {
        "risk_level": f"{top_name} ({level})",
        "risk_percentage": round(top_score, 1),
        "all_scores": {k: round(v, 1) for k, v in adjusted_scores.items()},
        "red_flag": adjusted_scores["High-Risk Fever"] >= 60,
        "high_risk_score": round(adjusted_scores["High-Risk Fever"], 1),
    }
