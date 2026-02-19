from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


BASE_DIR = Path(__file__).resolve().parent.parent
APP_DATA_DIR = BASE_DIR / "app_data"
AUTH_CONFIG_PATH = APP_DATA_DIR / "auth_config.yaml"
CASES_PATH = APP_DATA_DIR / "cases.csv"


CASE_COLUMNS = [
    "case_id",
    "timestamp",
    "doctor_username",
    "doctor_name",
    "patient_id",
    "patient_name",
    "disease_module",
    "risk_level",
    "risk_percentage",
    "recommendation",
    "reasoning",
    "plain_summary",
    "notes",
    "inputs_json",
    "all_scores_json",
    "rule_trace_json",
]


def _default_auth_config() -> Dict:
    return {
        "credentials": {
            "usernames": {
                "doctor": {
                    "email": "doctor@hospital.local",
                    "name": "Doctor User",
                    "password": "doctor123",
                    "role": "doctor",
                },
            }
        },
        "cookie": {
            "expiry_days": 1,
            "key": "medical_diagnosis_cookie_key",
            "name": "medical_diagnosis_auth",
        },
        "preauthorized": {"emails": []},
    }


def _ensure_data_files() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not AUTH_CONFIG_PATH.exists():
        with AUTH_CONFIG_PATH.open("w", encoding="utf-8") as file_obj:
            yaml.safe_dump(_default_auth_config(), file_obj, sort_keys=False)
    if not CASES_PATH.exists():
        pd.DataFrame(columns=CASE_COLUMNS).to_csv(CASES_PATH, index=False)


def _ensure_hashed_passwords(config: Dict) -> Dict:
    usernames = config.get("credentials", {}).get("usernames", {})
    changed = False
    for _, user_data in usernames.items():
        password = user_data.get("password", "")
        if not isinstance(password, str) or not password.startswith("$2"):
            user_data["password"] = stauth.Hasher.hash(str(password))
            changed = True
    if changed:
        with AUTH_CONFIG_PATH.open("w", encoding="utf-8") as file_obj:
            yaml.safe_dump(config, file_obj, sort_keys=False)
    return config


def _normalize_auth_config(config: Dict) -> Dict:
    defaults = _default_auth_config()
    changed = False

    if not isinstance(config, dict):
        config = {}
        changed = True

    if "credentials" not in config or not isinstance(config.get("credentials"), dict):
        config["credentials"] = {}
        changed = True
    if "usernames" not in config["credentials"] or not isinstance(config["credentials"].get("usernames"), dict):
        config["credentials"]["usernames"] = {}
        changed = True

    usernames = config["credentials"]["usernames"]
    default_usernames = defaults["credentials"]["usernames"]

    for username, user_data in list(usernames.items()):
        if not isinstance(user_data, dict):
            user_data = {"password": str(user_data)}
            usernames[username] = user_data
            changed = True

        default_profile = default_usernames.get(
            username,
            {
                "email": f"{username}@hospital.local",
                "name": username.replace("_", " ").title(),
                "password": "change_me",
                "role": "doctor",
            },
        )

        for field in ["email", "name", "role"]:
            if field not in user_data or user_data[field] in {None, ""}:
                user_data[field] = default_profile[field]
                changed = True

        if str(user_data.get("role", "doctor")).strip().lower() == "admin":
            user_data["role"] = "doctor"
            changed = True

        if "password" not in user_data or user_data["password"] in {None, ""}:
            user_data["password"] = default_profile["password"]
            changed = True

    if "cookie" not in config or not isinstance(config.get("cookie"), dict):
        config["cookie"] = {}
        changed = True
    for key, value in defaults["cookie"].items():
        if key not in config["cookie"] or config["cookie"][key] in {None, ""}:
            config["cookie"][key] = value
            changed = True

    if "preauthorized" not in config or not isinstance(config.get("preauthorized"), dict):
        config["preauthorized"] = dict(defaults["preauthorized"])
        changed = True
    elif "emails" not in config["preauthorized"]:
        config["preauthorized"]["emails"] = []
        changed = True

    if changed:
        with AUTH_CONFIG_PATH.open("w", encoding="utf-8") as file_obj:
            yaml.safe_dump(config, file_obj, sort_keys=False)

    return config


def get_authenticator() -> Tuple[stauth.Authenticate, Dict]:
    _ensure_data_files()
    with AUTH_CONFIG_PATH.open("r", encoding="utf-8") as file_obj:
        config = yaml.load(file_obj, Loader=SafeLoader) or {}
    config = _normalize_auth_config(config)
    config = _ensure_hashed_passwords(config)
    try:
        authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config.get("preauthorized", {}),
        )
    except Exception as exc:
        if "pre_authorized" not in str(exc) and "preauthorized" not in str(exc):
            raise
        authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
        )
    return authenticator, config


def load_cases() -> pd.DataFrame:
    _ensure_data_files()
    df = pd.read_csv(CASES_PATH)
    if df.empty:
        return pd.DataFrame(columns=CASE_COLUMNS)
    for col in CASE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[CASE_COLUMNS]


def save_case(
    doctor_username: str,
    doctor_name: str,
    patient_id: str,
    patient_name: str,
    disease_module: str,
    user_inputs: Dict,
    result: Dict,
    notes: str,
) -> str:
    df = load_cases()
    case_id = str(uuid.uuid4())
    row = {
        "case_id": case_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "doctor_username": doctor_username,
        "doctor_name": doctor_name,
        "patient_id": patient_id.strip(),
        "patient_name": patient_name.strip(),
        "disease_module": disease_module,
        "risk_level": str(result.get("risk_level", "")),
        "risk_percentage": float(result.get("risk_percentage", 0.0)),
        "recommendation": str(result.get("recommendation", "")),
        "reasoning": str(result.get("reasoning", "")),
        "plain_summary": str(result.get("plain_summary", "")),
        "notes": notes.strip(),
        "inputs_json": json.dumps(user_inputs),
        "all_scores_json": json.dumps(result.get("all_scores", {})),
        "rule_trace_json": json.dumps(result.get("rule_trace", [])),
    }
    updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(CASES_PATH, index=False)
    return case_id


def update_case_notes(case_id: str, notes: str) -> bool:
    df = load_cases()
    if df.empty or case_id not in set(df["case_id"]):
        return False
    df.loc[df["case_id"] == case_id, "notes"] = notes.strip()
    df.to_csv(CASES_PATH, index=False)
    return True


def parse_json_column(value, fallback):
    if pd.isna(value) or value == "":
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def visible_cases(df: pd.DataFrame, username: str) -> pd.DataFrame:
    return df[df["doctor_username"] == username].copy()


def module_numeric_averages(cases_df: pd.DataFrame, module_name: str) -> Dict[str, float]:
    module_df = cases_df[cases_df["disease_module"] == module_name]
    if module_df.empty:
        return {}
    records: List[Dict] = [
        parse_json_column(item, {})
        for item in module_df["inputs_json"].tolist()
    ]
    numeric_rows = []
    for record in records:
        numeric_only = {k: float(v) for k, v in record.items() if isinstance(v, (int, float))}
        if numeric_only:
            numeric_rows.append(numeric_only)
    if not numeric_rows:
        return {}
    stats_df = pd.DataFrame(numeric_rows)
    return {k: float(v) for k, v in stats_df.mean(numeric_only=True).dropna().to_dict().items()}


def common_positive_inputs(cases_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    for payload in cases_df.get("inputs_json", pd.Series(dtype="object")).tolist():
        record = parse_json_column(payload, {})
        for key, value in record.items():
            is_positive = False
            if isinstance(value, (int, float)):
                is_positive = float(value) > 0
            elif isinstance(value, str):
                normalized = value.strip().lower()
                is_positive = normalized in {"yes", "mild", "moderate", "severe", "high", "true", "1"}
            if is_positive:
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["Input", "Count"])
    summary = pd.DataFrame(
        [{"Input": key, "Count": value} for key, value in counts.items()]
    ).sort_values("Count", ascending=False)
    return summary.head(top_n)


def require_authentication() -> Dict:
    authenticator, config = get_authenticator()

    left_col, center_col, right_col = st.columns([1, 1.2, 1])
    with center_col:
        authenticator.login(location="main")

    authentication_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")
    name = st.session_state.get("name")

    if authentication_status is False:
        st.error("Invalid username or password.")
        st.stop()
    if authentication_status is None:
        st.info("Please login to access patient cases and diagnosis dashboard.")
        st.stop()

    user_data = config.get("credentials", {}).get("usernames", {}).get(username, {})
    role = "doctor"
    with st.sidebar:
        st.success(f"Signed in as {name}")
        authenticator.logout("Logout", location="sidebar")

    return {"username": username, "name": name, "role": role}