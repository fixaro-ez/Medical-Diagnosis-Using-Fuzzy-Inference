import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_inputs():
	return [
		{
			"type": "selectbox",
			"name": "age",
			"label": "Age range",
			"unit": "years",
			"help": "Patient age range.",
			"options": ["1-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76-90"],
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
			"name": "diastolic_bp",
			"label": "Diastolic blood pressure",
			"unit": "mmHg",
			"help": "Lower blood pressure reading.",
			"min": 50,
			"max": 130,
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
			"name": "fasting_blood_sugar",
			"label": "Fasting blood sugar",
			"unit": "mg/dL",
			"help": "Fasting blood sugar level.",
			"min": 50,
			"max": 200,
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
			"type": "toggle",
			"name": "is_smoker",
			"label": "Do you smoke?",
			"help": "Whether the patient currently smokes or has recently smoked.",
			"children": [
				{
					"type": "slider",
					"name": "cigarettes_per_day",
					"label": "Cigarettes per day",
					"unit": "cigarettes",
					"help": "Average number of cigarettes smoked per day.",
					"min": 0,
					"max": 60,
				},
				{
					"type": "slider",
					"name": "smoking_days_per_month",
					"label": "Smoking days per month",
					"unit": "days",
					"help": "How many days per month you smoke.",
					"min": 0,
					"max": 30,
				},
				{
					"type": "slider",
					"name": "days_since_last_cigarette",
					"label": "Days since last cigarette",
					"unit": "days",
					"help": "How many days since your last cigarette.",
					"min": 0,
					"max": 365,
				},
			],
		},
		{
			"type": "toggle",
			"name": "exercises_regularly",
			"label": "Do you exercise regularly?",
			"help": "Whether the patient exercises on a regular basis.",
			"children": [
				{
					"type": "slider",
					"name": "exercise_hours_per_week",
					"label": "Exercise hours per week",
					"unit": "hours",
					"help": "Average number of hours of exercise per week.",
					"min": 0.0,
					"max": 30.0,
				},
			],
		},
		{
			"type": "toggle",
			"name": "has_family_history",
			"label": "Family history of heart disease?",
			"help": "Whether any immediate family members have heart disease.",
			"children": [
				{
					"type": "slider",
					"name": "affected_family_members",
					"label": "Affected family members",
					"unit": "people",
					"help": "Number of immediate family members with heart disease.",
					"min": 0,
					"max": 10,
				},
			],
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
	inputs = {}
	for item in get_inputs():
		inputs[item["name"]] = item
		for child in item.get("children", []):
			inputs[child["name"]] = child

	age, age_universe = _make_antecedent(
		"age",
		1,
		90,
		[1, 1, 30],
		[25, 45, 60],
		[55, 90, 90],
	)
	systolic_bp, bp_universe = _make_antecedent(
		"systolic_bp",
		inputs["systolic_bp"]["min"],
		inputs["systolic_bp"]["max"],
		[90, 90, 120],
		[115, 130, 150],
		[145, 200, 200],
	)
	diastolic_bp, dbp_universe = _make_antecedent(
		"diastolic_bp",
		inputs["diastolic_bp"]["min"],
		inputs["diastolic_bp"]["max"],
		[50, 50, 70],
		[65, 80, 90],
		[85, 100, 130],
	)
	cholesterol, chol_universe = _make_antecedent(
		"cholesterol",
		inputs["cholesterol"]["min"],
		inputs["cholesterol"]["max"],
		[120, 120, 200],
		[190, 240, 280],
		[270, 320, 320],
	)
	fasting_blood_sugar, fbs_universe = _make_antecedent(
		"fasting_blood_sugar",
		inputs["fasting_blood_sugar"]["min"],
		inputs["fasting_blood_sugar"]["max"],
		[50, 50, 100],
		[90, 110, 130],
		[120, 150, 200],
	)
	heart_rate, hr_universe = _make_antecedent(
		"heart_rate",
		inputs["heart_rate"]["min"],
		inputs["heart_rate"]["max"],
		[40, 40, 70],
		[65, 80, 95],
		[90, 180, 180],
	)
	exercise_hours, ex_universe = _make_antecedent(
		"exercise_hours_per_week",
		0,
		30,
		[0, 0, 2],
		[1, 5, 10],
		[7, 15, 30],
	)

	cigarettes_per_day, cpd_universe = _make_antecedent(
		"cigarettes_per_day",
		inputs["cigarettes_per_day"]["min"],
		inputs["cigarettes_per_day"]["max"],
		[0, 0, 5],
		[0, 10, 20],
		[15, 30, 60],
	)

	affected_family_members, afm_universe = _make_antecedent(
		"affected_family_members",
		inputs["affected_family_members"]["min"],
		inputs["affected_family_members"]["max"],
		[0, 0, 1],
		[0, 2, 3],
		[2, 4, 10],
	)

	risk_universe = np.arange(0, 101, 1)
	risk = ctrl.Consequent(risk_universe, "risk")
	risk["low"] = fuzz.trimf(risk_universe, [0, 0, 25])
	risk["medium"] = fuzz.trimf(risk_universe, [20, 45, 70])
	risk["high"] = fuzz.trimf(risk_universe, [60, 100, 100])

	rules = [
		# --- HIGH risk rules ---
		ctrl.Rule(systolic_bp["high"] & cholesterol["high"], risk["high"]),
		ctrl.Rule(age["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(heart_rate["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(age["high"] & cholesterol["high"], risk["high"]),
		ctrl.Rule(cholesterol["high"] & heart_rate["high"], risk["high"]),
		ctrl.Rule(cigarettes_per_day["high"] & affected_family_members["high"], risk["high"]),
		ctrl.Rule(cigarettes_per_day["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(affected_family_members["high"] & cholesterol["high"], risk["high"]),
		ctrl.Rule(cigarettes_per_day["high"] & age["medium"], risk["high"]),
		ctrl.Rule(affected_family_members["high"] & age["high"], risk["high"]),
		ctrl.Rule(diastolic_bp["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(fasting_blood_sugar["high"] & cholesterol["high"], risk["high"]),
		ctrl.Rule(fasting_blood_sugar["high"] & systolic_bp["high"], risk["high"]),
		ctrl.Rule(exercise_hours["low"] & cigarettes_per_day["high"], risk["high"]),
		ctrl.Rule(exercise_hours["low"] & cholesterol["high"] & age["high"], risk["high"]),
		# --- MEDIUM risk rules ---
		ctrl.Rule(age["medium"] & (systolic_bp["medium"] | cholesterol["medium"]), risk["medium"]),
		ctrl.Rule(heart_rate["medium"] & cholesterol["medium"], risk["medium"]),
		ctrl.Rule(cigarettes_per_day["high"] & cholesterol["medium"], risk["medium"]),
		ctrl.Rule(cigarettes_per_day["high"], risk["medium"]),
		ctrl.Rule(affected_family_members["high"], risk["medium"]),
		ctrl.Rule(diastolic_bp["high"] & cholesterol["medium"], risk["medium"]),
		ctrl.Rule(fasting_blood_sugar["medium"] & cholesterol["medium"], risk["medium"]),
		ctrl.Rule(exercise_hours["low"] & age["medium"], risk["medium"]),
		ctrl.Rule(exercise_hours["medium"] & cigarettes_per_day["medium"], risk["medium"]),
		# --- LOW risk rules ---
		ctrl.Rule(age["low"] & systolic_bp["low"] & cholesterol["low"], risk["low"]),
		ctrl.Rule(age["low"] & systolic_bp["low"], risk["low"]),
		ctrl.Rule(heart_rate["low"] & systolic_bp["low"], risk["low"]),
		ctrl.Rule(cigarettes_per_day["low"] & affected_family_members["low"] & cholesterol["low"], risk["low"]),
		ctrl.Rule(age["low"], risk["low"]),
		ctrl.Rule(exercise_hours["high"] & cigarettes_per_day["low"], risk["low"]),
		ctrl.Rule(exercise_hours["high"] & cholesterol["low"], risk["low"]),
		ctrl.Rule(fasting_blood_sugar["low"] & cholesterol["low"], risk["low"]),
	]

	system = ctrl.ControlSystem(rules)
	simulation = ctrl.ControlSystemSimulation(system)

	simulation.input["age"] = float(user_data["age"])
	simulation.input["systolic_bp"] = float(user_data["systolic_bp"])
	simulation.input["diastolic_bp"] = float(user_data["diastolic_bp"])
	simulation.input["cholesterol"] = float(user_data["cholesterol"])
	simulation.input["fasting_blood_sugar"] = float(user_data["fasting_blood_sugar"])
	simulation.input["heart_rate"] = float(user_data["heart_rate"])
	simulation.input["exercise_hours_per_week"] = float(user_data.get("exercise_hours_per_week", 0))
	simulation.input["cigarettes_per_day"] = float(user_data.get("cigarettes_per_day", 0))
	simulation.input["affected_family_members"] = float(user_data.get("affected_family_members", 0))

	simulation.compute()
	risk_value = float(simulation.output.get("risk", 0.0))

	smoking_flag = _scale_risk(float(user_data.get("cigarettes_per_day", 0)), 0, 20)
	# Adjust smoking impact by frequency and recency
	smoking_days = float(user_data.get("smoking_days_per_month", 30))
	days_since = float(user_data.get("days_since_last_cigarette", 0))
	frequency_factor = smoking_days / 30.0  # 1.0 = daily, 0.0 = never
	recency_factor = max(0.0, 1.0 - (days_since / 365.0))  # fades over a year
	smoking_flag = smoking_flag * frequency_factor * recency_factor

	fh_flag = _scale_risk(float(user_data.get("affected_family_members", 0)), 0, 3)

	age_score = _scale_risk(float(user_data["age"]), 40, 70)
	bp_score = _scale_risk(float(user_data["systolic_bp"]), 110, 160)
	dbp_score = _scale_risk(float(user_data["diastolic_bp"]), 70, 100)
	chol_score = _scale_risk(float(user_data["cholesterol"]), 180, 300)
	fbs_score = _scale_risk(float(user_data["fasting_blood_sugar"]), 100, 150)
	hr_score = _scale_risk(float(user_data["heart_rate"]), 60, 110)
	exercise_score = 1.0 - _scale_risk(float(user_data.get("exercise_hours_per_week", 0)), 0, 10)

	calibration = (
		0.15 * age_score
		+ 0.15 * bp_score
		+ 0.08 * dbp_score
		+ 0.15 * chol_score
		+ 0.10 * fbs_score
		+ 0.10 * hr_score
		+ 0.10 * exercise_score
		+ 0.10 * smoking_flag
		+ 0.07 * fh_flag
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
	dbp_high_m = fuzz.interp_membership(dbp_universe, diastolic_bp["high"].mf, user_data["diastolic_bp"])
	chol_high = fuzz.interp_membership(chol_universe, cholesterol["high"].mf, user_data["cholesterol"])
	fbs_high_m = fuzz.interp_membership(fbs_universe, fasting_blood_sugar["high"].mf, user_data["fasting_blood_sugar"])
	age_high = fuzz.interp_membership(age_universe, age["high"].mf, user_data["age"])
	hr_high = fuzz.interp_membership(hr_universe, heart_rate["high"].mf, user_data["heart_rate"])
	ex_low_m = fuzz.interp_membership(ex_universe, exercise_hours["low"].mf, user_data.get("exercise_hours_per_week", 0))

	reasons = []
	if bp_high > 0.6 and chol_high > 0.6:
		reasons.append("High systolic BP combined with high cholesterol raised the risk.")
	if dbp_high_m > 0.6 and bp_high > 0.6:
		reasons.append("Both systolic and diastolic blood pressure are elevated.")
	if age_high > 0.6 and bp_high > 0.6:
		reasons.append("Higher age alongside high BP increased the risk score.")
	if hr_high > 0.6 and bp_high > 0.6:
		reasons.append("Elevated heart rate with high BP contributed to risk.")
	if fbs_high_m > 0.6:
		reasons.append("Elevated fasting blood sugar increases cardiovascular risk.")
	if ex_low_m > 0.6:
		reasons.append("Low exercise level contributes to increased risk.")
	smoking_high = fuzz.interp_membership(cpd_universe, cigarettes_per_day["high"].mf, user_data.get("cigarettes_per_day", 0))
	fh_high = fuzz.interp_membership(
		afm_universe, affected_family_members["high"].mf, user_data.get("affected_family_members", 0)
	)
	if smoking_high > 0.6 and fh_high > 0.6:
		reasons.append("High cigarette consumption combined with family history increased risk.")
	elif smoking_high > 0.6:
		reasons.append("High cigarette consumption increased the risk.")
	elif fh_high > 0.6:
		reasons.append("Family history increased the risk.")

	if not reasons:
		if risk_level == "Low":
			reasons.append("Most inputs are in low or mid ranges, leading to a lower risk score.")
		elif risk_level == "Medium":
			reasons.append("Mixed input levels resulted in a moderate risk score.")
		else:
			reasons.append("Multiple inputs trended higher, pushing the risk upward.")

	rule_trace = [
		{
			"rule": "IF systolic BP is high AND cholesterol is high THEN cardiovascular risk is high",
			"strength": round(float(min(bp_high, chol_high)), 2),
		},
		{
			"rule": "IF age is high AND systolic BP is high THEN cardiovascular risk is high",
			"strength": round(float(min(age_high, bp_high)), 2),
		},
		{
			"rule": "IF cigarette consumption is high AND family history is high THEN cardiovascular risk is high",
			"strength": round(float(min(smoking_high, fh_high)), 2),
		},
	]
	rule_trace = [item for item in rule_trace if item["strength"] > 0]

	plain_summary = (
		f"Heart risk is {risk_level.lower()} at {round(risk_value, 1)}%. "
		f"Main drivers include blood pressure, cholesterol, and lifestyle/family history patterns."
	)

	result = {
		"risk_percentage": round(risk_value, 1),
		"risk_level": risk_level,
		"recommendation": recommendation,
		"reasoning": " ".join(reasons),
		"rule_trace": rule_trace,
		"plain_summary": plain_summary,
	}
	return result
