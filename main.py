from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st

from modules.app_services import (
    load_cases,
    module_numeric_averages,
    parse_json_column,
    require_authentication,
    save_case,
    update_case_notes,
    visible_cases,
)
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


def _to_widget_key(name: str) -> str:
    return f"input_{name}"


def _render_input(item):
    field_type = item.get("type")
    label = _label_with_unit(item.get("label", ""), item.get("unit", ""))
    help_text = item.get("help", "")
    widget_key = _to_widget_key(item.get("name", "field"))

    if field_type == "slider":
        min_value = item.get("min", 0)
        max_value = item.get("max", 100)
        value = item.get("default", _coerce_slider_value(min_value, max_value))
        if widget_key not in st.session_state:
            st.session_state[widget_key] = value
        step = item.get("step", 0.1 if isinstance(st.session_state[widget_key], float) else 1)
        return st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=widget_key,
            help=help_text,
        )

    if field_type in {"selectbox", "select"}:
        options = item.get("options", [])
        default = options[0] if options else None
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default
        return st.selectbox(label, options, key=widget_key, help=help_text)

    if field_type == "number":
        min_value = item.get("min", 0)
        max_value = item.get("max", 100)
        value = item.get("default", _coerce_slider_value(min_value, max_value))
        if widget_key not in st.session_state:
            st.session_state[widget_key] = value
        step = item.get("step", 0.1 if isinstance(st.session_state[widget_key], float) else 1)
        return st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=widget_key,
            help=help_text,
        )

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
            score = 1.0 if str(value).strip().lower() in {"yes", "y", "true", "1", "mild", "moderate", "severe"} else 0.0
            detail = str(value)

        contributions.append(
            {
                "Input": label,
                "Value": value,
                "Contribution": round(score, 2),
                "Detail": detail,
            }
        )
    return pd.DataFrame(contributions).sort_values("Contribution", ascending=False)


def _default_rule_trace(user_inputs, input_schema, risk_level):
    contribution_df = _input_contributions(user_inputs, input_schema)
    top = contribution_df.head(3)
    traces = []
    for _, row in top.iterrows():
        traces.append(
            {
                "rule": f"IF {row['Input']} is elevated THEN risk tends {risk_level}",
                "strength": float(min(1.0, row["Contribution"])),
            }
        )
    return traces


def _default_plain_summary(selected, result, contributions):
    top_inputs = contributions.head(2)["Input"].tolist()
    drivers = ", ".join(top_inputs) if top_inputs else "overall symptom pattern"
    return (
        f"{selected} assessment indicates {result.get('risk_level', 'an unclear')} risk "
        f"with score {result.get('risk_percentage', 0)}%. Main contributing factors: {drivers}."
    )


def _format_timestamp(value):
    timestamp_text = str(value).strip()
    if not timestamp_text:
        return ""

    parsed = pd.to_datetime(timestamp_text, errors="coerce")
    if pd.isna(parsed):
        return timestamp_text
    return parsed.strftime("%d %b %Y, %I:%M %p")


def _build_case_selector_text(row):
    patient = row.get("patient_name") or row.get("patient_id") or "Unknown"
    formatted_time = _format_timestamp(row.get("timestamp", ""))
    return f"{formatted_time} | {patient} | {row.get('disease_module', '')} | {row.get('risk_level', '')}"


def _render_result_block(selected, result, input_schema, user_inputs):
    st.success(f"Risk: {result['risk_level']} ({result['risk_percentage']}%)")
    st.write(result.get("recommendation", ""))
    st.caption(result.get("reasoning", ""))

    contribution_df = _input_contributions(user_inputs, input_schema)
    if not result.get("plain_summary"):
        result["plain_summary"] = _default_plain_summary(selected, result, contribution_df)

    if not result.get("rule_trace"):
        result["rule_trace"] = _default_rule_trace(user_inputs, input_schema, result.get("risk_level", "medium"))

    st.markdown("#### Plain-Language Summary")
    st.info(result.get("plain_summary", "No summary generated."))

    trace_df = pd.DataFrame(result.get("rule_trace", []))
    if not trace_df.empty:
        if "strength" in trace_df.columns:
            trace_df["strength"] = trace_df["strength"].astype(float).round(2)
        st.markdown("#### Rule Trace (Most Activated)")
        st.dataframe(trace_df, use_container_width=True)

    if "all_scores" in result:
        st.markdown("#### Condition Distribution")
        raw_scores = result["all_scores"]
        total = sum(raw_scores.values())
        if total > 0:
            dist = {k: round(v / total * 100, 1) for k, v in raw_scores.items()}
        else:
            dist = {k: 0.0 for k in raw_scores}
        score_df = pd.DataFrame(
            [{"Condition": key, "Probability (%)": value} for key, value in dist.items()]
        ).sort_values("Probability (%)", ascending=False)
        dist_chart = (
            alt.Chart(score_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Probability (%):Q"),
                color=alt.Color("Condition:N", legend=alt.Legend(title="Condition")),
                tooltip=["Condition", "Probability (%)"],
            )
        )
        st.altair_chart(dist_chart, use_container_width=True)
        st.dataframe(score_df, use_container_width=True)

    if result.get("red_flag"):
        st.error("Red flag detected: high-risk infectious pattern. Please seek urgent medical attention.")


def _render_comparison(selected, user_inputs, all_cases):
    st.subheader("Comparative Analysis (Anonymized)")
    if all_cases.empty:
        st.info("No historical cases yet. Save more patient sessions to unlock comparison analytics.")
        return

    module_cases = all_cases[all_cases["disease_module"] == selected]
    if module_cases.empty:
        st.info("No historical cases found for the selected disease module.")
        return

    cohort_avg_risk = float(module_cases["risk_percentage"].astype(float).mean())
    current_risk = float(st.session_state.get("last_result", {}).get("risk_percentage", 0.0))

    col1, col2 = st.columns(2)
    col1.metric("Current Case Risk", f"{current_risk:.1f}%")
    col2.metric("Historical Average", f"{cohort_avg_risk:.1f}%")

    averages = module_numeric_averages(all_cases, selected)
    numeric_current = {k: float(v) for k, v in user_inputs.items() if isinstance(v, (int, float))}

    compare_rows = []
    for key, cur_value in numeric_current.items():
        if key in averages:
            compare_rows.append({"Input": key, "Current": cur_value, "Historical Avg": averages[key]})

    if compare_rows:
        compare_df = pd.DataFrame(compare_rows)
        compare_long = compare_df.melt("Input", var_name="Type", value_name="Value")
        chart = (
            alt.Chart(compare_long)
            .mark_bar()
            .encode(
                x=alt.X("Value:Q"),
                y=alt.Y("Input:N", sort="-x"),
                color=alt.Color("Type:N"),
                tooltip=["Input", "Type", "Value"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(compare_df, use_container_width=True)
    else:
        st.caption("No comparable numeric inputs available for this module yet.")


def _render_history_panel(selected, input_schema, username):
    st.subheader("Patient Case History")
    cases = visible_cases(load_cases(), username)
    if cases.empty:
        st.info("No saved cases yet.")
        return

    filtered = cases[cases["disease_module"] == selected]
    if filtered.empty:
        st.info("No saved cases for this disease module.")
        return

    filtered = filtered.sort_values("timestamp", ascending=False)
    options = { _build_case_selector_text(row): row for _, row in filtered.iterrows() }
    chosen = st.selectbox("Select a saved case", list(options.keys()))
    selected_row = options[chosen]

    if st.button("Load selected case into form", key="load_case_btn"):
        payload = parse_json_column(selected_row.get("inputs_json", "{}"), {})
        for item in input_schema:
            key = item.get("name")
            if key in payload:
                st.session_state[_to_widget_key(key)] = payload[key]
        st.success("Case loaded into the current form.")
        st.rerun()

    st.markdown("#### Saved Case Snapshot")
    st.write(f"Patient: {selected_row.get('patient_name', '')} ({selected_row.get('patient_id', '')})")
    st.write(f"Result: {selected_row.get('risk_level', '')} ({selected_row.get('risk_percentage', 0)}%)")
    st.caption(selected_row.get("plain_summary", ""))


def _render_patient_profile(username):
    st.header("Patient Profile View")
    cases = visible_cases(load_cases(), username)
    if cases.empty:
        st.info("No patient cases to display.")
        return

    cases = cases.sort_values("timestamp", ascending=False)
    unique_patients = (
        cases[["patient_id", "patient_name"]]
        .fillna("")
        .drop_duplicates()
    )
    patient_options = {}
    for _, patient in unique_patients.iterrows():
        patient_id = str(patient.get("patient_id", "")).strip()
        patient_name = str(patient.get("patient_name", "")).strip()
        if not patient_id and not patient_name:
            continue
        label = f"{patient_id or 'No ID'} | {patient_name or 'No Name'}"
        patient_options[label] = (patient_id, patient_name)

    if not patient_options:
        st.info("Saved cases are missing patient ID and patient name. Save a case with patient details first.")
        return

    search_text = st.text_input("Search patient", placeholder="Type patient ID or name")
    filtered_labels = list(patient_options.keys())
    if search_text.strip():
        query = search_text.strip().lower()
        filtered_labels = [label for label in filtered_labels if query in label.lower()]

    if not filtered_labels:
        st.info("No patients match your search.")
        return

    selected_patient_label = st.selectbox("Select patient", filtered_labels)
    selected_patient_id, selected_patient_name = patient_options[selected_patient_label]

    patient_id_series = cases["patient_id"].fillna("").astype(str).str.strip()
    patient_name_series = cases["patient_name"].fillna("").astype(str).str.strip()

    patient_cases = cases[
        ((patient_id_series == selected_patient_id) if selected_patient_id else False)
        | ((patient_name_series == selected_patient_name) if selected_patient_name else False)
    ]

    patient_name = patient_cases["patient_name"].dropna().astype(str).head(1)
    display_name = patient_name.iloc[0] if not patient_name.empty else "Unknown"
    st.write(f"Patient Name: {display_name}")
    st.write(f"Patient ID: {selected_patient_id or 'Not provided'}")

    timeline = patient_cases[["timestamp", "disease_module", "risk_level", "risk_percentage", "doctor_name"]].copy()
    timeline["timestamp"] = timeline["timestamp"].map(_format_timestamp)
    st.markdown("#### Diagnosis History")
    st.dataframe(timeline, use_container_width=True)

    options = {_build_case_selector_text(row): row for _, row in patient_cases.iterrows()}
    chosen = st.selectbox("Open a diagnosis record", list(options.keys()), key="patient_profile_case")
    row = options[chosen]

    st.markdown("#### Case Details")
    inputs = parse_json_column(row.get("inputs_json", "{}"), {})
    st.dataframe(pd.DataFrame([inputs]), use_container_width=True)
    st.write(f"Recommendation: {row.get('recommendation', '')}")
    st.caption(row.get("plain_summary", ""))

    note_key = f"profile_note_{row.get('case_id', '')}"
    raw_note = row.get("notes", "")
    default_notes = "" if pd.isna(raw_note) else str(raw_note)
    if default_notes.strip().lower() == "nan":
        default_notes = ""
    notes = st.text_area("Doctor notes", value=default_notes, key=note_key)
    if st.button("Update notes", key=f"update_note_btn_{row.get('case_id', '')}"):
        if update_case_notes(str(row.get("case_id", "")), notes):
            st.success("Notes updated.")
            st.rerun()
        else:
            st.error("Unable to update notes for this case.")


st.set_page_config(page_title="Medical Diagnosis", layout="wide")
st.markdown("<h1 style='text-align: center;'>Multi-Disease Fuzzy Expert System</h1>", unsafe_allow_html=True)

user = require_authentication()

modules = {
    "Heart": heart,
    "Diabetes": diabetes,
    "Respiratory": respiratory,
    "Infectious": infectious,
}

with st.sidebar:
    page = st.radio("Navigation", ["Diagnosis", "Patient Profile"])

if page == "Patient Profile":
    _render_patient_profile(user["username"])
else:
    selected = st.selectbox("Select Disease", list(modules.keys()), key="selected_module")
    module = modules[selected]
    input_schema = _normalize_inputs(module.get_inputs())
    _validate_inputs(input_schema)

    diagnosis_tab, explanation_tab, impact_tab, comparison_tab, history_tab = st.tabs([
        "Diagnosis",
        "Explanation",
        "Input Impact",
        "Comparison",
        "Case History",
    ])

    with diagnosis_tab:
        st.subheader("Patient Inputs")

        my_cases = visible_cases(load_cases(), user["username"])
        patient_choices = {"New patient": ("", "")}
        if not my_cases.empty:
            unique_patients = (
                my_cases[["patient_id", "patient_name"]]
                .fillna("")
                .drop_duplicates()
            )
            for _, row in unique_patients.iterrows():
                pid = str(row.get("patient_id", "")).strip()
                pname = str(row.get("patient_name", "")).strip()
                if not pid and not pname:
                    continue
                label = f"{pid or 'No ID'} | {pname or 'No Name'}"
                patient_choices[label] = (pid, pname)

        patient_search = st.text_input(
            "Search existing patient",
            placeholder="Type patient ID or name",
            key="diagnosis_patient_search",
        )
        filtered_patient_labels = list(patient_choices.keys())
        if patient_search.strip():
            query = patient_search.strip().lower()
            filtered_patient_labels = [label for label in filtered_patient_labels if query in label.lower()]
        if not filtered_patient_labels:
            filtered_patient_labels = ["New patient"]

        selected_patient = st.selectbox("Existing patient (optional)", filtered_patient_labels)
        if selected_patient != "New patient":
            selected_id, selected_name = patient_choices[selected_patient]
            st.session_state["patient_id"] = selected_id
            st.session_state["patient_name"] = selected_name

        profile_col1, profile_col2 = st.columns(2)
        patient_id = profile_col1.text_input("Patient ID", key="patient_id")
        patient_name = profile_col2.text_input("Patient Name", key="patient_name")

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

        if st.button("Diagnose", key="diagnose_btn"):
            result = module.run_inference(user_inputs)
            st.session_state["last_result"] = result
            st.session_state["last_user_inputs"] = user_inputs
            st.session_state["last_schema"] = input_schema
            st.session_state["last_selected"] = selected

        if "last_result" in st.session_state and st.session_state.get("last_selected") == selected:
            _render_result_block(
                selected,
                st.session_state["last_result"],
                st.session_state["last_schema"],
                st.session_state["last_user_inputs"],
            )

            st.markdown("#### Save Patient Session")
            notes = st.text_area("Doctor Notes", key="save_case_notes")
            if st.button("Save Case", key="save_case_btn"):
                if not patient_id.strip() and not patient_name.strip():
                    st.error("Enter at least Patient ID or Patient Name before saving.")
                else:
                    case_id = save_case(
                        doctor_username=user["username"],
                        doctor_name=user["name"],
                        patient_id=patient_id,
                        patient_name=patient_name,
                        disease_module=selected,
                        user_inputs=st.session_state["last_user_inputs"],
                        result=st.session_state["last_result"],
                        notes=notes,
                    )
                    st.success(f"Case saved successfully (ID: {case_id[:8]}...).")

    with explanation_tab:
        st.subheader("How the Fuzzy System Works")
        st.markdown(
            "The system follows four main steps: fuzzification, membership functions, inference rules, and defuzzification."
        )

        st.markdown("#### 1) Fuzzification")
        st.write(
            "Raw inputs are mapped to membership degrees using low/medium/high curves. "
            "This converts a single value into multiple fuzzy truths."
        )

        st.markdown("#### 2) Membership Functions")
        current_inputs = st.session_state.get("last_user_inputs", user_inputs)
        for item in input_schema:
            if item.get("type") != "slider":
                continue
            label = _label_with_unit(item.get("label", ""), item.get("unit", ""))
            value = current_inputs.get(item.get("name"))
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

        st.markdown("#### 3) Rule Tracing")
        last_result = st.session_state.get("last_result", {})
        trace_df = pd.DataFrame(last_result.get("rule_trace", []))
        if trace_df.empty and current_inputs:
            trace_df = pd.DataFrame(_default_rule_trace(current_inputs, input_schema, "medium"))
        if trace_df.empty:
            st.info("Run a diagnosis to see activated rules.")
        else:
            st.dataframe(trace_df, use_container_width=True)

        st.markdown("#### 4) Defuzzification")
        st.write("The final step converts aggregated fuzzy outputs into a single risk percentage using centroid logic.")
        output_value = st.session_state.get("last_result", {}).get("risk_percentage")
        output_chart = _membership_chart("Risk Percentage", 0, 100, output_value)
        st.altair_chart(output_chart, use_container_width=True)

    with impact_tab:
        st.subheader("How Each Input Contributes")
        st.write(
            "These contribution scores are heuristic values derived from fuzzy membership degrees."
        )
        current_inputs = st.session_state.get("last_user_inputs", user_inputs)
        if current_inputs:
            contribution_df = _input_contributions(current_inputs, input_schema)
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

    with comparison_tab:
        _render_comparison(selected, st.session_state.get("last_user_inputs", user_inputs), load_cases())

    with history_tab:
        _render_history_panel(selected, input_schema, user["username"])


if __name__ == "__main__":
    print("Welcome to the Medical Diagnosis System!")
