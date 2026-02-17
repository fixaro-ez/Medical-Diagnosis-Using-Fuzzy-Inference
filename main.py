import altair as alt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st

from modules.diabetes import engine as diabetes
from modules.heart import engine as heart
from modules.infectious import engine as infectious
from modules.respiratory import engine as respiratory


def _label_with_unit(label, unit):
    return f"{label} ({unit})" if unit else label


def _coerce_slider_value(min_value, max_value):
    if isinstance(min_value, float) or isinstance(max_value, float):
        mid = (float(min_value) + float(max_value)) / 2.0
        return round(mid, 1)
    return int((min_value + max_value) / 2)


def _render_input(item):
    field_type = item.get("type")
    label = _label_with_unit(item.get("label", ""), item.get("unit", ""))
    help_text = item.get("help", "")

    if field_type == "slider":
        min_value = item.get("min", 0)
        max_value = item.get("max", 100)
        value = item.get("default", _coerce_slider_value(min_value, max_value))
        step = 0.1 if isinstance(value, float) else 1
        return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, help=help_text)

    if field_type == "selectbox":
        options = item.get("options", [])
        return st.selectbox(label, options, help=help_text)

    if field_type == "number":
        min_value = item.get("min", 0)
        max_value = item.get("max", 100)
        value = item.get("default", _coerce_slider_value(min_value, max_value))
        step = item.get("step", 0.1 if isinstance(value, float) else 1)
        return st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            help=help_text,
        )

    if field_type == "select":
        options = item.get("options", [])
        return st.selectbox(label, options, help=help_text)

    st.warning(f"Unsupported input type: {field_type}")
    return None


def _normalize_inputs(inputs):
    normalized = []
    for item in inputs:
        field_type = item.get("type", "slider")
        if field_type == "select":
            field_type = "selectbox"

        normalized_item = {
            "type": field_type,
            "name": item.get("name"),
            "label": item.get("label", item.get("name", "")),
            "unit": item.get("unit", ""),
            "help": item.get("help", ""),
        }

        if field_type in {"slider", "number"}:
            normalized_item["min"] = item.get("min", 0)
            normalized_item["max"] = item.get("max", 100)
            if "default" in item:
                normalized_item["default"] = item.get("default")
            if "step" in item:
                normalized_item["step"] = item.get("step")

        if field_type == "selectbox":
            normalized_item["options"] = item.get("options", [])

        normalized.append(normalized_item)
    return normalized


def _validate_inputs(inputs):
    required = {"type", "name", "label", "unit", "help"}
    for item in inputs:
        missing = required - set(item.keys())
        if missing:
            st.warning(f"Input '{item.get('name', 'unknown')}' missing keys: {', '.join(sorted(missing))}")
        if item.get("type") in {"slider", "number"}:
            if "min" not in item or "max" not in item:
                st.warning(f"Numeric input '{item.get('name', 'unknown')}' needs 'min' and 'max'.")
        if item.get("type") in {"selectbox", "select"} and "options" not in item:
            st.warning(f"Selectbox '{item.get('name', 'unknown')}' needs 'options'.")


def _membership_curves(min_value, max_value):
    mid = (min_value + max_value) / 2.0
    universe = np.linspace(min_value, max_value, 200)
    low = fuzz.trimf(universe, [min_value, min_value, mid])
    medium = fuzz.trimf(universe, [min_value, mid, max_value])
    high = fuzz.trimf(universe, [mid, max_value, max_value])
    return universe, low, medium, high


def _membership_dataframe(min_value, max_value):
    universe, low, medium, high = _membership_curves(min_value, max_value)
    return pd.DataFrame(
        {
            "x": universe,
            "Low": low,
            "Medium": medium,
            "High": high,
        }
    )


def _membership_long_dataframe(min_value, max_value):
    df = _membership_dataframe(min_value, max_value)
    return df.melt("x", var_name="level", value_name="degree")


def _membership_chart(title, min_value, max_value, value=None):
    df_long = _membership_long_dataframe(min_value, max_value)
    base = (
        alt.Chart(df_long)
        .mark_line()
        .encode(
            x=alt.X("x:Q", title=title),
            y=alt.Y("degree:Q", title="Membership"),
            color=alt.Color("level:N", title="Level"),
        )
    )

    if value is None:
        return base

    marker_df = pd.DataFrame({"x": [value], "label": [f"{value:.1f}"]})
    marker = (
        alt.Chart(marker_df)
        .mark_rule(color="#e63946", strokeDash=[4, 4], strokeWidth=2)
        .encode(x="x:Q")
    )
    label = (
        alt.Chart(marker_df)
        .mark_text(align="left", dx=6, dy=-6, color="#e63946")
        .encode(x="x:Q", y=alt.value(0), text="label:N")
    )
    return base + marker + label


def _membership_degrees(min_value, max_value, value):
    universe, low, medium, high = _membership_curves(min_value, max_value)
    low_d = fuzz.interp_membership(universe, low, value)
    medium_d = fuzz.interp_membership(universe, medium, value)
    high_d = fuzz.interp_membership(universe, high, value)
    return {"low": low_d, "medium": medium_d, "high": high_d}


def _input_contributions(user_inputs, input_schema):
    contributions = []
    for item in input_schema:
        name = item.get("name")
        label = _label_with_unit(item.get("label", ""), item.get("unit", ""))
        value = user_inputs.get(name)
        if item.get("type") in {"slider", "number"}:
            degrees = _membership_degrees(item.get("min", 0), item.get("max", 100), value)
            score = (degrees["medium"] * 0.5) + degrees["high"]
            detail = f"Low {degrees['low']:.2f}, Medium {degrees['medium']:.2f}, High {degrees['high']:.2f}"
        else:
            score = 1.0 if str(value).strip().lower() in {"yes", "y", "true", "1"} else 0.0
            detail = "Yes" if score == 1.0 else "No"

        contributions.append(
            {
                "Input": label,
                "Value": value,
                "Contribution": round(score, 2),
                "Detail": detail,
            }
        )
    return pd.DataFrame(contributions)


st.set_page_config(page_title="Medical Diagnosis", layout="centered")
st.title("Multi-Disease Fuzzy Expert System")

modules = {
    "Heart": heart,
    "Diabetes": diabetes,
    "Respiratory": respiratory,
    "Infectious": infectious,
}

selected = st.selectbox("Select Disease", list(modules.keys()))
module = modules[selected]
input_schema = _normalize_inputs(module.get_inputs())
_validate_inputs(input_schema)

diagnosis_tab, explanation_tab, impact_tab = st.tabs([
    "Diagnosis",
    "Explanation",
    "Input Impact",
])

with diagnosis_tab:
    st.subheader("Patient Inputs")
    user_inputs = {}
    for item in input_schema:
        user_inputs[item["name"]] = _render_input(item)

    if user_inputs:
        summary_rows = []
        for item in input_schema:
            summary_rows.append(
                {
                    "Input": _label_with_unit(item.get("label", ""), item.get("unit", "")),
                    "Value": user_inputs.get(item.get("name")),
                }
            )
        with st.sidebar:
            st.markdown("#### Input Summary")
            st.table(pd.DataFrame(summary_rows))

    if st.button("Diagnose"):
        results = module.run_inference(user_inputs)
        st.success(f"Risk: {results['risk_level']} ({results['risk_percentage']}%)")
        st.write(results.get("recommendation", ""))
        st.caption(results.get("reasoning", ""))
        if "all_scores" in results:
            st.markdown("#### Condition Scores")
            score_df = pd.DataFrame(
                [{"Condition": key, "Score": value} for key, value in results["all_scores"].items()]
            ).sort_values("Score", ascending=False)
            st.dataframe(score_df, use_container_width=True)
        if results.get("red_flag"):
            st.error("Red flag detected: high-risk infectious pattern. Please seek urgent medical attention.")
        st.session_state["last_risk"] = results.get("risk_percentage")

with explanation_tab:
    st.subheader("How the Fuzzy System Works")
    st.markdown(
        "The system follows four main steps: fuzzification, membership functions, inference rules, "
        "and defuzzification. Each step keeps the logic interpretable for reviewers."
    )
    st.info(
        "This system supports doctors by describing how likely a disease may occur using fuzzy rules, "
        "rather than a fixed yes or no answer."
    )


    st.markdown("#### 1) Fuzzification")
    st.write(
        "Raw inputs are mapped to membership degrees using low/medium/high curves. "
        "This converts a single value into multiple fuzzy truths."
    )

    st.markdown("#### 2) Membership Functions")
    for item in input_schema:
        if item.get("type") != "slider":
            continue
        label = _label_with_unit(item.get("label", ""), item.get("unit", ""))
        value = user_inputs.get(item.get("name"))
        min_value = item.get("min", 0)
        max_value = item.get("max", 100)
        st.markdown(f"**{label}**")
        chart = _membership_chart(label, min_value, max_value, value)
        st.altair_chart(chart, use_container_width=True)
        if value is None:
            continue
        degrees = _membership_degrees(min_value, max_value, value)
        st.caption(
            f"Fuzzified degrees -> Low: {degrees['low']:.2f}, Medium: {degrees['medium']:.2f}, High: {degrees['high']:.2f}"
        )

    st.markdown("#### 3) Inference")
    inference_examples = {
        "Heart": "Example rule: If systolic BP is high and cholesterol is high, then risk is high.",
        "Diabetes": "Example rule: If glucose is high, risk is high; if glucose is medium with high BMI or insulin and high pedigree/age, risk increases.",
        "Respiratory": "Example rule: If coughing and shortness of breath are high, Asthma probability increases; if fever and chills are high, Pneumonia probability increases.",
        "Infectious": "Example rule: If temperature is very high or breathing difficulty is severe, high-risk fever rises; flu and gastro scores are inferred from symptom clusters.",
    }
    st.write(
        f"{inference_examples.get(selected, 'The engine combines fuzzy rules to estimate risk.')} "
        "The engine combines multiple rules and aggregates their outputs."
    )

    st.markdown("#### 4) Defuzzification")
    st.write(
        "The final step converts aggregated fuzzy outputs into a single risk percentage using a centroid method."
    )

    st.markdown("#### Output Membership")
    output_value = st.session_state.get("last_risk")
    output_chart = _membership_chart("Risk Percentage", 0, 100, output_value)
    st.altair_chart(output_chart, use_container_width=True)
    st.caption("Low: 0-35, Medium: 35-65, High: 65-100 (approximate ranges).")

with impact_tab:
    st.subheader("How Each Input Contributes")
    st.write(
        "These contribution scores are a heuristic based on fuzzy membership degrees. "
        "They indicate which inputs are pushing the risk higher, but they are not exact rule weights."
    )
    if user_inputs:
        contribution_df = _input_contributions(user_inputs, input_schema)
        chart = (
            alt.Chart(contribution_df)
            .mark_bar()
            .encode(
                x=alt.X("Contribution:Q", scale=alt.Scale(domain=[0, 1.5])),
                y=alt.Y("Input:N", sort="-x"),
                tooltip=["Input", "Value", "Contribution", "Detail"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(contribution_df, use_container_width=True)
    else:
        st.info("Enter inputs in the Diagnosis tab to see contribution details here.")


if __name__ == "__main__":
    print("Welcome to the Medical Diagnosis System!")
