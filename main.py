from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st

from modules.app_services import (
    load_cases,
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
from modules.recommender.engine import SymptomRecommender
from modules.reporting import generate_pdf_report


def _label_with_unit(label, unit):
    return f"{label} ({unit})" if unit else label


def _parse_age_range(value):
    """Convert an age range string like '26-35' to its midpoint float."""
    if isinstance(value, str):
        value = value.strip()
        if "+" in value:
            # Treat "76+" effectively as the 76-90 group (midpoint ~83)
            try:
                base = float(value.replace("+", "").strip())
                return base + 7.0 
            except ValueError:
                return 83.0

        if "-" in value:
            parts = value.split("-")
            try:
                low, high = float(parts[0]), float(parts[1])
                return (low + high) / 2.0
            except (ValueError, IndexError):
                return 30.0
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return 30.0


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

        if field_type == "toggle":
            normalized_item["options"] = ["No", "Yes"]
            children = item.get("children", [])
            normalized_item["children"] = _normalize_inputs(children)

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
        if item.get("type") == "toggle":
            for child in item.get("children", []):
                _validate_inputs([child])


def _flatten_schema(input_schema):
    """Flatten schema including children from toggle items, for downstream functions."""
    flat = []
    for item in input_schema:
        if item.get("type") == "toggle":
            flat.append(item)
            for child in item.get("children", []):
                flat.append(child)
        else:
            flat.append(item)
    return flat


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
        if item.get("type") == "toggle":
            continue
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
    st.markdown("### 📊 Diagnostic Results")
    
    risk_pct = result.get('risk_percentage', 0)
    risk_level = result.get('risk_level', 'Unknown')
    
    # Visual Metrics using Columns
    m1, m2, m3 = st.columns(3)
    with m1:
         st.metric(label="Risk Probability", value=f"{risk_pct}%", delta=f"{risk_level} Risk")
    with m2:
         st.metric(label="Assessed Condition", value=selected)
    with m3:
         confidence = "High" if risk_pct > 70 or risk_pct < 30 else "Moderate"
         st.metric(label="Model Confidence", value=confidence)

    # Detailed Recommendation
    with st.expander("📝 View Clinical Recommendation", expanded=True):
        st.info(f"**Recommendation:** {result.get('recommendation', 'No specific recommendation.')}")
        st.caption(f"**Reasoning:** {result.get('reasoning', '')}")

    contribution_df = _input_contributions(user_inputs, input_schema)

    if not result.get("rule_trace"):
        result["rule_trace"] = _default_rule_trace(user_inputs, input_schema, result.get("risk_level", "medium"))

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
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Probability (%):Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Condition:N", sort="-x", title="Condition"),
                color=alt.Color("Condition:N", legend=None),
                tooltip=[
                    alt.Tooltip("Condition:N", title="Condition"),
                    alt.Tooltip("Probability (%):Q", title="Probability", format=".1f"),
                ],
            )
        )
        # Add text labels on bars
        dist_text = (
            alt.Chart(score_df)
            .mark_text(align="left", dx=4, fontSize=12)
            .encode(
                x=alt.X("Probability (%):Q"),
                y=alt.Y("Condition:N", sort="-x"),
                text=alt.Text("Probability (%):Q", format=".1f"),
            )
        )
        st.altair_chart(dist_chart + dist_text, use_container_width=True)
        st.dataframe(score_df, use_container_width=True)

    if result.get("red_flag"):
        st.error("Red flag detected: high-risk infectious pattern. Please seek urgent medical attention.")


def _render_history_panel(selected, input_schema, username):
    st.subheader("Patient Case History")
    if st.button("Refresh history", key="refresh_history_btn"):
        st.rerun()

    cases = visible_cases(load_cases(), username)
    if cases.empty:
        st.info("No saved cases yet.")
        return

    known_modules = {"Heart", "Diabetes", "Respiratory", "Infectious"}
    data_modules = set(cases["disease_module"].dropna().astype(str).str.strip().tolist())
    module_options = ["All modules"] + sorted(known_modules | data_modules)
    default_module = selected if selected in module_options else "All modules"
    selected_module = st.selectbox(
        "Module filter",
        module_options,
        index=module_options.index(default_module),
        key="history_module_filter",
    )

    if selected_module == "All modules":
        filtered = cases.copy()
    else:
        filtered = cases[cases["disease_module"] == selected_module].copy()

    if filtered.empty:
        st.info("No saved cases found for the selected filter.")
        return

    filtered = filtered.copy()
    filtered["_ts_sort"] = pd.to_datetime(filtered["timestamp"], format="mixed", errors="coerce")
    filtered = filtered.sort_values(["_ts_sort", "timestamp"], ascending=False)
    st.caption(f"Total cases shown: {len(filtered)}")

    case_options = {}
    label_counts = {}
    for _, row in filtered.iterrows():
        case_id = str(row.get("case_id", ""))
        base_label = _build_case_selector_text(row)
        seen = label_counts.get(base_label, 0)
        label_counts[base_label] = seen + 1
        label = base_label if seen == 0 else f"{base_label} ({seen + 1})"
        case_options[label] = case_id

    chosen = st.selectbox("Select a saved case", list(case_options.keys()), key="history_case_selector")
    selected_case_id = case_options[chosen]
    selected_row = filtered[filtered["case_id"].astype(str) == str(selected_case_id)].iloc[0]

    if st.button("Load selected case into form", key="load_case_btn"):
        payload = parse_json_column(selected_row.get("inputs_json", "{}"), {})
        st.session_state["pending_case_load"] = {
            "inputs": payload,
            "patient_id": str(selected_row.get("patient_id", "")),
            "patient_name": str(selected_row.get("patient_name", "")),
            "disease_module": str(selected_row.get("disease_module", "")),
        }
        st.success("Case selected. Loading into form...")
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

    cases = cases.copy()
    cases["_ts_sort"] = pd.to_datetime(cases["timestamp"], format="mixed", errors="coerce")
    cases = cases.sort_values(["_ts_sort", "timestamp"], ascending=False)
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

    # ── Chronological Patient Timeline ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### Chronological Patient Timeline")

    # Build timeline data with parsed timestamps
    tl = patient_cases.copy()
    tl["_dt"] = pd.to_datetime(tl["timestamp"], format="mixed", errors="coerce")
    tl = tl.dropna(subset=["_dt"]).sort_values("_dt")

    if tl.empty:
        st.info("No timestamped records to display on the timeline.")
    else:
        # ── Disease Module Filter ───────────────────────────────────────
        modules_in_data = sorted(tl["disease_module"].dropna().astype(str).unique().tolist())
        module_filter_options = ["All Modules"] + modules_in_data
        module_filter = st.selectbox(
            "Filter timeline by disease",
            module_filter_options,
            key="timeline_module_filter",
        )
        tl_filtered = tl if module_filter == "All Modules" else tl[tl["disease_module"] == module_filter]

        # ── 1) Risk Trend Chart ─────────────────────────────────────────
        st.markdown("##### Risk Score Trend")
        risk_tl = tl_filtered[["_dt", "disease_module", "risk_percentage"]].copy()
        risk_tl["risk_percentage"] = pd.to_numeric(risk_tl["risk_percentage"], errors="coerce")
        risk_tl = risk_tl.dropna(subset=["risk_percentage"])

        if not risk_tl.empty:
            risk_tl = risk_tl.reset_index(drop=True)
            risk_tl = risk_tl.rename(columns={
                "_dt": "Date",
                "disease_module": "Module",
                "risk_percentage": "Risk",
            })
            # Format date as readable label for the x-axis
            risk_tl["Visit"] = risk_tl["Date"].dt.strftime("%d %b %Y, %I:%M %p")
            # Preserve chronological order
            visit_order = risk_tl["Visit"].tolist()

            # Add threshold rows into the same dataframe so no cross-dataset layering
            risk_tl["Low_Med"] = 35
            risk_tl["Med_High"] = 65

            bars = (
                alt.Chart(risk_tl)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Visit:N", title="Date", sort=visit_order,
                            axis=alt.Axis(labelAngle=-35)),
                    y=alt.Y("Risk:Q", scale=alt.Scale(domain=[0, 100]), title="Risk %"),
                    color=alt.Color("Module:N", title="Disease Module"),
                    tooltip=[
                        alt.Tooltip("Visit:N", title="Date"),
                        alt.Tooltip("Module:N", title="Module"),
                        alt.Tooltip("Risk:Q", title="Risk %", format=".1f"),
                    ],
                )
            )
            # Value labels on top of each bar
            bar_text = (
                alt.Chart(risk_tl)
                .mark_text(dy=-8, fontSize=12, fontWeight="bold")
                .encode(
                    x=alt.X("Visit:N", sort=visit_order),
                    y=alt.Y("Risk:Q"),
                    text=alt.Text("Risk:Q", format=".1f"),
                )
            )
            # Threshold lines from the same dataset (avoids cross-data color conflicts)
            rule_low = (
                alt.Chart(risk_tl)
                .mark_rule(color="#f39c12", strokeDash=[5, 5], opacity=0.6, strokeWidth=1.5)
                .encode(y="Low_Med:Q")
            )
            rule_high = (
                alt.Chart(risk_tl)
                .mark_rule(color="#e74c3c", strokeDash=[5, 5], opacity=0.6, strokeWidth=1.5)
                .encode(y="Med_High:Q")
            )
            st.altair_chart(bars + bar_text + rule_low + rule_high, use_container_width=True)
        else:
            st.caption("No risk scores available for charting.")

        # ── 2) Key Metric Trends ────────────────────────────────────────
        # Extract numeric inputs from each case and plot trends
        metric_rows = []
        for _, case_row in tl_filtered.iterrows():
            dt = case_row["_dt"]
            module = str(case_row.get("disease_module", ""))
            payload = parse_json_column(case_row.get("inputs_json", "{}"), {})
            for k, v in payload.items():
                try:
                    numeric_v = float(v)
                except (ValueError, TypeError):
                    continue
                # Skip toggle flags and age ranges
                if k.startswith("is_") or k.startswith("has_") or k.startswith("exercises_") or k.startswith("had_"):
                    continue
                if k == "age":
                    continue
                metric_rows.append({
                    "Date": dt,
                    "Module": module,
                    "Metric": k.replace("_", " ").title(),
                    "Value": numeric_v,
                })

        if metric_rows:
            metric_df = pd.DataFrame(metric_rows)
            available_metrics = sorted(metric_df["Metric"].unique().tolist())
            st.markdown("##### Input Metric Trends")
            st.caption("Select metrics to see how individual measurements changed over time.")
            selected_metrics = st.multiselect(
                "Metrics to plot",
                available_metrics,
                default=available_metrics[:3],
                key="timeline_metric_select",
            )
            if selected_metrics:
                filtered_metrics = metric_df[metric_df["Metric"].isin(selected_metrics)].copy()
                # Format date as readable label for x-axis
                filtered_metrics["Visit"] = filtered_metrics["Date"].dt.strftime("%d %b %Y, %I:%M %p")
                metric_visit_order = (
                    filtered_metrics[["Date", "Visit"]]
                    .drop_duplicates()
                    .sort_values("Date")["Visit"]
                    .tolist()
                )
                metric_chart = (
                    alt.Chart(filtered_metrics)
                    .mark_line(point=alt.OverlayMarkDef(size=50, filled=True))
                    .encode(
                        x=alt.X("Visit:N", title="Date", sort=metric_visit_order,
                                axis=alt.Axis(labelAngle=-35)),
                        y=alt.Y("Value:Q", title="Value"),
                        color=alt.Color("Metric:N", title="Metric"),
                        strokeDash=alt.StrokeDash("Module:N", title="Module"),
                        tooltip=[
                            alt.Tooltip("Visit:N", title="Date"),
                            alt.Tooltip("Module:N", title="Module"),
                            alt.Tooltip("Metric:N", title="Metric"),
                            alt.Tooltip("Value:Q", title="Value", format=".1f"),
                        ],
                    )
                )
                st.altair_chart(metric_chart, use_container_width=True)

        # ── 3) Scrollable Visit Timeline ────────────────────────────────
        st.markdown("##### Visit History")
        risk_colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
        for _, visit in tl_filtered.iterrows():
            dt_str = visit["_dt"].strftime("%d %b %Y, %I:%M %p")
            module = str(visit.get("disease_module", ""))
            risk_level = str(visit.get("risk_level", ""))
            risk_pct = visit.get("risk_percentage", "")
            doctor = str(visit.get("doctor_name", ""))
            # Determine color from risk level text
            color = "#999"
            for level_key, level_color in risk_colors.items():
                if level_key.lower() in risk_level.lower():
                    color = level_color
                    break
            # Extract key inputs for this visit
            visit_inputs = parse_json_column(visit.get("inputs_json", "{}"), {})
            input_pills = ""
            for ik, iv in visit_inputs.items():
                if ik.startswith("is_") or ik.startswith("has_") or ik.startswith("exercises_") or ik.startswith("had_"):
                    continue
                if ik == "age":
                    continue
                try:
                    float(iv)
                except (ValueError, TypeError):
                    continue
                label = ik.replace("_", " ").title()
                input_pills += (
                    f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
                    f'padding:2px 8px;border-radius:12px;margin-right:5px;margin-top:3px;'
                    f'font-size:0.8em;display:inline-block;">{label}: {iv}</span>'
                )
            st.markdown(
                f"""<div style="border-left: 4px solid {color}; padding: 10px 14px; margin-bottom: 10px;
                background: rgba(0,0,0,0.02); border-radius: 6px;">
                <div><strong>{dt_str}</strong> &nbsp;|&nbsp; <strong>{module}</strong> &nbsp;|&nbsp;
                <span style="color:{color}; font-weight:bold;">{risk_level} ({risk_pct}%)</span>
                &nbsp;|&nbsp; Dr. {doctor}</div>
                <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px;">{input_pills}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Individual Case Details ─────────────────────────────────────────
    options = {_build_case_selector_text(row): row for _, row in patient_cases.iterrows()}
    chosen = st.selectbox("Open a diagnosis record", list(options.keys()), key="patient_profile_case")
    row = options[chosen]

    st.markdown("#### Case Details")
    inputs = parse_json_column(row.get("inputs_json", "{}"), {})
    st.dataframe(pd.DataFrame([inputs]), use_container_width=True)
    st.write(f"Recommendation: {row.get('recommendation', '')}")

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

st.markdown("<h1 style='text-align: center;'>Multi-Disease Fuzzy Decision Support System</h1>", unsafe_allow_html=True)

user = require_authentication()

modules = {
    "Heart": heart,
    "Diabetes": diabetes,
    "Respiratory": respiratory,
    "Infectious": infectious,
}


# --- CUSTOM CSS TO IMPROVE INPUT VISIBILITY ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&display=swap');

    /* Apply serif font to headings to match the image */
    h1, h2, h3, .css-10trblm {
        font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
        color: #1f3b4d !important; /* Dark teal/slate color from image */
    }

    /* Make input fields more visible */
    input[type="text"], input[type="number"], .stNumberInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #004085 !important;
        border-radius: 5px !important;
        color: #002752 !important;
    }
    /* Enhance dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #004085 !important;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.title("Med-Fuzzy Logic")
    page = st.radio("Main Menu", ["Diagnosis", "Patient Profile"])
    
    st.info(f"**Dr. {user['name']}**")
    st.divider()
    st.caption("v1.2.0 • System Online")

# --- INNOVATIVE FEATURE: AI SYMPTOM CHECKER ---
if page == "Diagnosis":
    with st.expander("🤖 AI Symptom Checker (Describe patient symptoms)", expanded=True):
        col_ai_1, col_ai_2 = st.columns([3, 1])
        with col_ai_1:
            symptom_text = st.text_input("Enter symptoms (e.g., 'Patient feels very thirsty and tired'):", key="symptom_input")
        
        if symptom_text:
            recommender = SymptomRecommender()
            # predict() now returns a list of tuples: [('heart', 0.8), ('diabetes', 0.1)...]
            results = recommender.predict(symptom_text)
            
            if results:
                # 1. Get Top Prediction
                top_category, top_confidence = results[0]
                
                # Map model output to module names (Capitalized)
                module_map = {
                    "diabetes": "Diabetes",
                    "heart": "Heart",
                    "infectious": "Infectious",
                    "respiratory": "Respiratory"
                }
                mapped_top_category = module_map.get(str(top_category).lower(), str(top_category).capitalize())
                
                with col_ai_2:
                    st.markdown(f"<br>", unsafe_allow_html=True) # Spacer
                    if st.button(f"Go to {mapped_top_category} Module", type="primary"):
                        st.session_state["selected_module"] = mapped_top_category
                        st.rerun()
                
                st.info(f"**Primary Diagnosis:** detected signs of **{mapped_top_category}** (Confidence: {float(top_confidence)*100:.1f}%)")
                
                # 2. Show Differential Diagnosis
                if len(results) > 1:
                    with st.expander("View Differential Diagnosis (Other Possibilities)"):
                        st.markdown("**Alternative possibilities based on symptoms:**")
                        for cat, conf in results[1:]:
                            mapped_alt = module_map.get(str(cat).lower(), str(cat).capitalize())
                            st.write(f"- **{mapped_alt}**: {float(conf)*100:.1f}% probability")
            else:
                 st.warning("Could not predict a category. Try describing more symptoms.")

pending_case_load = st.session_state.pop("pending_case_load", None)
if isinstance(pending_case_load, dict):
    pending_module = str(pending_case_load.get("disease_module", "")).strip()
    if pending_module in modules:
        st.session_state["selected_module"] = pending_module

    pending_patient_id = str(pending_case_load.get("patient_id", "")).strip()
    pending_patient_name = str(pending_case_load.get("patient_name", "")).strip()
    st.session_state["patient_id"] = pending_patient_id
    st.session_state["patient_name"] = pending_patient_name

    pending_inputs = pending_case_load.get("inputs", {})
    if isinstance(pending_inputs, dict):
        for key, value in pending_inputs.items():
            st.session_state[_to_widget_key(str(key))] = value

if page == "Patient Profile":
    _render_patient_profile(user["username"])
else:
    selected = st.selectbox("Select Disease", list(modules.keys()), key="selected_module")
    module = modules[selected]
    input_schema = _normalize_inputs(module.get_inputs())
    _validate_inputs(input_schema)

    diagnosis_tab, explanation_tab, impact_tab, history_tab = st.tabs([
        "Diagnosis",
        "Explanation",
        "Input Impact",
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

        st.markdown("### Clinical Indicators")
        # Grid Layout for Inputs
        col1, col2 = st.columns(2)
        
        user_inputs = {}
        for idx, item in enumerate(input_schema):
            target_col = col1 if idx % 2 == 0 else col2
            with target_col:
                if item.get("type") == "toggle":
                    toggle_name = item["name"]
                    toggle_label = item.get("label", toggle_name)
                    help_text = item.get("help", "")
                    widget_key = _to_widget_key(toggle_name)
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = "No"
                    toggle_val = st.selectbox(
                        toggle_label,
                        ["No", "Yes"],
                        key=widget_key,
                        help=help_text,
                    )
                    user_inputs[toggle_name] = toggle_val
                    children = item.get("children", [])
                    if toggle_val == "Yes":
                        for child in children:
                            user_inputs[child["name"]] = _render_input(child)
                    else:
                        for child in children:
                            default_val = child.get("min", 0)
                            if child.get("type") == "slider" or child.get("type") == "number":
                                default_val = child.get("min", 0)
                            user_inputs[child["name"]] = default_val
                else:
                    user_inputs[item["name"]] = _render_input(item)


        flat_schema = _flatten_schema(input_schema)

        st.markdown("---")
        if st.button("🚀 Run Diagnosis", key="diagnose_btn", type="primary", use_container_width=True):
            inference_inputs = dict(user_inputs)
            if "age" in inference_inputs:
                inference_inputs["age"] = _parse_age_range(inference_inputs["age"])
            result = module.run_inference(inference_inputs)
            st.session_state["last_result"] = result
            st.session_state["last_user_inputs"] = user_inputs
            st.session_state["last_schema"] = flat_schema
            st.session_state["last_selected"] = selected

        if "last_result" in st.session_state and st.session_state.get("last_selected") == selected:
            _render_result_block(
                selected,
                st.session_state["last_result"],
                st.session_state["last_schema"],
                st.session_state["last_user_inputs"],
            )

            st.markdown("#### Save & Report")
            notes = st.text_area("Doctor Notes", key="save_case_notes")
            
            col_save, col_report = st.columns([1, 1])
            
            with col_save:
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

            with col_report:
                if (patient_name or patient_id) and st.session_state.get("last_result"):
                    try:
                        pdf_bytes = generate_pdf_report(
                            patient_name=patient_name,
                            patient_id=patient_id,
                            disease_module=selected,
                            inputs=st.session_state["last_user_inputs"],
                            result=st.session_state["last_result"],
                            doctor_name=user["name"],
                            notes=notes
                        )
                        st.download_button(
                            label="📄 Download Medical Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"Report_{patient_name}_{selected}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.warning(f"Report generation unavailable: {e}")



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
        for item in flat_schema:
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
            trace_df = pd.DataFrame(_default_rule_trace(current_inputs, flat_schema, "medium"))
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
            contribution_df = _input_contributions(current_inputs, flat_schema)
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

    with history_tab:
        _render_history_panel(selected, flat_schema, user["username"])


if __name__ == "__main__":
    print("Welcome to the Medical Diagnosis System!")
