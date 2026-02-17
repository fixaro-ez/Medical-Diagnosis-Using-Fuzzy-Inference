import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_inputs():
	return [
		{
			"type": "slider",
			"name": "age",
			"label": "Age",
			"unit": "years",
			"help": "Age in years.",
			"min": 18,
			"max": 90,
		},
		{
			"type": "slider",
			"name": "systolic_bp",
			"label": "Systolic blood pressure",
			"unit": "mmHg",
			"help": "Upper blood pressure reading.",
			"min": 90,
			"max": 200,
		},
		{
			"type": "slider",
			"name": "cholesterol",
			"label": "Total cholesterol",
			"unit": "mg/dL",
			"help": "Total cholesterol level.",
			"min": 120,
			"max": 320,
		},
		{
			"type": "slider",
			"name": "heart_rate",
			"label": "Resting heart rate",
			"unit": "bpm",
			"help": "Beats per minute at rest.",
			"min": 40,
			"max": 180,
		},
		{
			"type": "selectbox",
			"name": "smoking",
			"label": "Smoking status",
			"unit": "",
			"help": "Current smoker?",
			"options": ["No", "Yes"],
		},
		{
			"type": "selectbox",
			"name": "family_history",
			"label": "Family history",
			"unit": "",
			"help": "Family history of heart disease?",
			"options": ["No", "Yes"],
		},
	]


def _make_antecedent(name, min_value, max_value, low_pts, med_pts, high_pts):
	universe = np.arange(min_value, max_value + 1, 1)
	antecedent = ctrl.Antecedent(universe, name)
	antecedent["low"] = fuzz.trimf(universe, low_pts)
	antecedent["medium"] = fuzz.trimf(universe, med_pts)
	antecedent["high"] = fuzz.trimf(universe, high_pts)
	return antecedent, universe


def _binary_to_numeric(value):
	if isinstance(value, str):
		return 1.0 if value.strip().lower() in {"yes", "y", "true", "1"} else 0.0
	return 1.0 if value else 0.0


def _scale_risk(value, low, high):
	if high == low:
		return 0.0
	return max(0.0, min(1.0, (value - low) / (high - low)))


def run_inference(user_data):
	inputs = {item["name"]: item for item in get_inputs()}

	age, age_universe = _make_antecedent(
		"age",
		inputs["age"]["min"],
		inputs["age"]["max"],
		[18, 18, 45],
		[40, 55, 65],
		[60, 90, 90],
	)
	systolic_bp, bp_universe = _make_antecedent(
		"systolic_bp",
		inputs["systolic_bp"]["min"],
		inputs["systolic_bp"]["max"],
		[90, 90, 120],
		[115, 130, 150],
		[145, 200, 200],
	)
	cholesterol, chol_universe = _make_antecedent(
		"cholesterol",
		inputs["cholesterol"]["min"],
		inputs["cholesterol"]["max"],
		[120, 120, 200],
		[190, 240, 280],
		[270, 320, 320],
	)
	heart_rate, hr_universe = _make_antecedent(
		"heart_rate",
		inputs["heart_rate"]["min"],
		inputs["heart_rate"]["max"],
		[40, 40, 70],
		[65, 80, 95],
		[90, 180, 180],
	)

	smoking = ctrl.Antecedent(np.arange(0, 2, 1), "smoking")
	smoking["low"] = fuzz.trimf(smoking.universe, [0, 0, 1])
	smoking["high"] = fuzz.trimf(smoking.universe, [0, 1, 1])

	family_history = ctrl.Antecedent(np.arange(0, 2, 1), "family_history")
	family_history["low"] = fuzz.trimf(family_history.universe, [0, 0, 1])
	family_history["high"] = fuzz.trimf(family_history.universe, [0, 1, 1])

	risk_universe = np.arange(0, 101, 1)
	risk = ctrl.Consequent(risk_universe, "risk")
	risk["low"] = fuzz.trimf(risk_universe, [0, 0, 25])
	risk["medium"] = fuzz.trimf(risk_universe, [20, 45, 70])
	risk["high"] = fuzz.trimf(risk_universe, [60, 100, 100])

	rules = [
		ctrl.Rule(systolic_bp["high"] & cholesterol["high"], risk["high"]),
		ctrl.Rule(age["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(heart_rate["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(age["high"] & cholesterol["high"], risk["high"]),
		ctrl.Rule(cholesterol["high"] & heart_rate["high"], risk["high"]),
		ctrl.Rule(smoking["high"] & family_history["high"], risk["high"]),
		ctrl.Rule(smoking["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(family_history["high"] & cholesterol["high"], risk["high"]),
			ctrl.Rule(smoking["high"] & age["medium"], risk["high"]),
			ctrl.Rule(family_history["high"] & age["high"], risk["high"]),
		ctrl.Rule(age["medium"] & (systolic_bp["medium"] | cholesterol["medium"]), risk["medium"]),
		ctrl.Rule(heart_rate["medium"] & cholesterol["medium"], risk["medium"]),
		ctrl.Rule(smoking["high"] & cholesterol["medium"], risk["medium"]),
			ctrl.Rule(smoking["high"], risk["medium"]),
			ctrl.Rule(family_history["high"], risk["medium"]),
		ctrl.Rule(age["low"] & systolic_bp["low"] & cholesterol["low"], risk["low"]),
		ctrl.Rule(age["low"] & systolic_bp["low"], risk["low"]),
		ctrl.Rule(heart_rate["low"] & systolic_bp["low"], risk["low"]),
		ctrl.Rule(smoking["low"] & family_history["low"] & cholesterol["low"], risk["low"]),
			ctrl.Rule(age["low"], risk["low"]),
	]

	system = ctrl.ControlSystem(rules)
	simulation = ctrl.ControlSystemSimulation(system)

	simulation.input["age"] = float(user_data["age"])
	simulation.input["systolic_bp"] = float(user_data["systolic_bp"])
	simulation.input["cholesterol"] = float(user_data["cholesterol"])
	simulation.input["heart_rate"] = float(user_data["heart_rate"])
	simulation.input["smoking"] = _binary_to_numeric(user_data["smoking"])
	simulation.input["family_history"] = _binary_to_numeric(user_data["family_history"])

	simulation.compute()
	risk_value = float(simulation.output.get("risk", 0.0))

	smoking_flag = _binary_to_numeric(user_data["smoking"])
	fh_flag = _binary_to_numeric(user_data["family_history"])

	age_score = _scale_risk(float(user_data["age"]), 40, 70)
	bp_score = _scale_risk(float(user_data["systolic_bp"]), 110, 160)
	chol_score = _scale_risk(float(user_data["cholesterol"]), 180, 300)
	hr_score = _scale_risk(float(user_data["heart_rate"]), 60, 110)

	calibration = (
		0.25 * age_score
		+ 0.25 * bp_score
		+ 0.2 * chol_score
		+ 0.15 * hr_score
		+ 0.1 * smoking_flag
		+ 0.05 * fh_flag
	)
	calibrated_value = calibration * 100.0

	if risk_value == 0.0:
		risk_value = calibrated_value
	else:
		risk_value = (0.6 * risk_value) + (0.4 * calibrated_value)
	risk_value = min(100.0, risk_value + (8.0 * smoking_flag) + (6.0 * fh_flag))

	if risk_value < 35:
		risk_level = "Low"
		recommendation = "Maintain healthy habits and continue regular checkups."
	elif risk_value < 65:
		risk_level = "Medium"
		recommendation = "Consider lifestyle changes and consult a clinician."
	else:
		risk_level = "High"
		recommendation = "Seek medical advice promptly for further evaluation."

	bp_high = fuzz.interp_membership(bp_universe, systolic_bp["high"].mf, user_data["systolic_bp"])
	chol_high = fuzz.interp_membership(chol_universe, cholesterol["high"].mf, user_data["cholesterol"])
	age_high = fuzz.interp_membership(age_universe, age["high"].mf, user_data["age"])
	hr_high = fuzz.interp_membership(hr_universe, heart_rate["high"].mf, user_data["heart_rate"])

	reasons = []
	if bp_high > 0.6 and chol_high > 0.6:
		reasons.append("High systolic BP combined with high cholesterol raised the risk.")
	if age_high > 0.6 and bp_high > 0.6:
		reasons.append("Higher age alongside high BP increased the risk score.")
	if hr_high > 0.6 and bp_high > 0.6:
		reasons.append("Elevated heart rate with high BP contributed to risk.")
	smoking_high = fuzz.interp_membership(smoking.universe, smoking["high"].mf, _binary_to_numeric(user_data["smoking"]))
	fh_high = fuzz.interp_membership(
		family_history.universe, family_history["high"].mf, _binary_to_numeric(user_data["family_history"])
	)
	if smoking_high > 0.6 and fh_high > 0.6:
		reasons.append("Smoking combined with family history increased risk.")
	elif smoking_high > 0.6:
		reasons.append("Smoking status increased the risk.")
	elif fh_high > 0.6:
		reasons.append("Family history increased the risk.")

	if not reasons:
		if risk_level == "Low":
			reasons.append("Most inputs are in low or mid ranges, leading to a lower risk score.")
		elif risk_level == "Medium":
			reasons.append("Mixed input levels resulted in a moderate risk score.")
		else:
			reasons.append("Multiple inputs trended higher, pushing the risk upward.")

	result = {
		"risk_percentage": round(risk_value, 1),
		"risk_level": risk_level,
		"recommendation": recommendation,
		"reasoning": " ".join(reasons),
	}
	return result
