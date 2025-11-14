"""Session state helpers shared across Streamlit pages."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from datetime import date
from typing import Any, Dict, Iterable, List

import streamlit as st

from models.entities import Transaction
from utils.i18n import I18n
from utils.storage import save_to_storage

logger = logging.getLogger(__name__)


DEFAULT_STATE: Dict[str, Any] = {
    "transactions": [],
    "ocr_raw_text": "",
    "ocr_results": [],
    "analysis_summary": [],
    "chat_history": [],
    "user_profile": None,
    "product_recommendations": [],
    "anomaly_flags": [],
    "anomaly_history": [],
    "trusted_merchants": [],
    "anomaly_message": "",
    "locale": "zh_CN",
    "chat_cache": {},
    "chat_cache_version": 0,
    "monthly_budget": 5000.0,
    "data_restored": False,
    "selected_page": "home",  # 确保默认从首页开始
}


def init_session_state() -> None:
    """Ensure every expected key exists in `st.session_state`."""
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            if isinstance(default_value, list):
                st.session_state[key] = list(default_value)
            elif isinstance(default_value, dict):
                st.session_state[key] = dict(default_value)
            else:
                st.session_state[key] = default_value

    if "i18n" not in st.session_state:
        st.session_state["i18n"] = I18n(st.session_state.get("locale", "zh_CN"))


def reset_session_state(keys: List[str] | None = None) -> None:
    """Clear selected session keys, or all known keys when omitted."""
    target_keys = keys if keys is not None else list(DEFAULT_STATE.keys())
    for key in target_keys:
        if key in st.session_state:
            del st.session_state[key]


def _normalize_transaction(entry: Transaction | dict) -> Transaction:
    if isinstance(entry, Transaction):
        return entry
    return Transaction(**entry)


def _serialize_transaction_entry(entry: Transaction | dict) -> Dict[str, Any]:
    """Convert transaction input into a JSON-serialisable dict."""
    if isinstance(entry, Transaction):
        data = entry.model_dump(mode="json")
    else:
        data = dict(entry)
        if isinstance(data.get("date"), date):
            data["date"] = data["date"].isoformat()
    return data


def _persist_state(key: str, value: Any) -> None:
    """Persist a session value to storage while swallowing I/O errors."""
    try:
        save_to_storage(key, value)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to persist %s: %s", key, exc)


def _invalidate_chat_cache() -> None:
    """重置聊天缓存并增加版本号，确保依赖数据变化后不会读旧值。"""

    cache = st.session_state.get("chat_cache")
    if isinstance(cache, dict):
        cache.clear()
    else:
        st.session_state["chat_cache"] = {}
    st.session_state["chat_cache_version"] = (
        st.session_state.get("chat_cache_version", 0) + 1
    )


def get_transactions() -> List[Transaction]:
    """Return transactions stored in session as `Transaction` models."""
    return [
        _normalize_transaction(entry)
        for entry in st.session_state.get("transactions", [])
    ]


def set_transactions(transactions: Iterable[Transaction | dict]) -> None:
    """Persist a new transaction list into session state."""
    serialized = [_serialize_transaction_entry(txn) for txn in transactions]
    st.session_state["transactions"] = serialized
    _persist_state("transactions", serialized)
    _invalidate_chat_cache()


def get_trusted_merchants() -> List[str]:
    """Return the merchant whitelist."""
    merchants = st.session_state.get("trusted_merchants", [])
    if isinstance(merchants, list):
        return list(dict.fromkeys(m.strip() for m in merchants if m))
    return []


def add_trusted_merchant(name: str) -> None:
    """Add a merchant to the whitelist if not already present."""
    name = name.strip()
    if not name:
        return
    merchants = get_trusted_merchants()
    if name not in merchants:
        merchants.append(name)
        st.session_state["trusted_merchants"] = merchants


def remove_trusted_merchant(name: str) -> None:
    """Remove a merchant from the whitelist."""
    merchants = [m for m in get_trusted_merchants() if m != name]
    st.session_state["trusted_merchants"] = merchants


def _serialize_anomaly(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure anomaly data is JSON-serialisable before storing."""
    result = deepcopy(record)
    if isinstance(result.get("date"), date):
        result["date"] = result["date"].isoformat()
    return result


def get_active_anomalies() -> List[Dict[str, Any]]:
    """Return current active anomalies."""
    return [dict(item) for item in st.session_state.get("anomaly_flags", [])]


def get_anomaly_history() -> List[Dict[str, Any]]:
    """Return resolved anomalies with user feedback."""
    return [dict(item) for item in st.session_state.get("anomaly_history", [])]


def update_anomaly_state(
    *,
    active: Iterable[Dict[str, Any]] | None = None,
    history: Iterable[Dict[str, Any]] | None = None,
    message: str | None = None,
) -> None:
    """Persist anomaly state back into session."""
    if active is not None:
        st.session_state["anomaly_flags"] = [
            _serialize_anomaly(item) for item in active
        ]
    if history is not None:
        st.session_state["anomaly_history"] = [
            _serialize_anomaly(item) for item in history
        ]
    if message is not None:
        st.session_state["anomaly_message"] = message


def record_anomaly_feedback(anomaly: Dict[str, Any], status: str) -> None:
    """Store user feedback for a specific anomaly."""
    anomaly_id = anomaly.get("transaction_id")
    record = dict(anomaly)
    record["status"] = status
    existing = [
        item
        for item in get_anomaly_history()
        if item.get("transaction_id") != anomaly_id
    ]
    existing.append(record)
    update_anomaly_state(history=existing)


def sync_anomaly_state(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Merge freshly detected anomalies with existing user feedback.

    Returns the active anomalies list for immediate use.
    """
    history_map = {item.get("transaction_id"): item for item in get_anomaly_history()}
    active: List[Dict[str, Any]] = []
    for anomaly in report.get("items", []):
        anomaly = dict(anomaly)
        txn_id = anomaly.get("transaction_id")
        history = history_map.get(txn_id)
        if history and history.get("status") in {"confirmed", "fraud"}:
            continue
        if history:
            anomaly["status"] = history.get("status", "review")
        active.append(anomaly)
    update_anomaly_state(active=active, message=report.get("message"))
    return active


def get_i18n() -> I18n:
    """Return the current I18n instance from session."""
    init_session_state()
    i18n = st.session_state.get("i18n")
    if isinstance(i18n, I18n):
        return i18n
    i18n = I18n(st.session_state.get("locale", "zh_CN"))
    st.session_state["i18n"] = i18n
    return i18n


def switch_locale(locale: str) -> None:
    """Switch application locale and refresh I18n instance."""
    st.session_state["locale"] = locale
    i18n = get_i18n()
    i18n.switch_locale(locale)


def get_monthly_budget() -> float:
    """Return the globally configured monthly budget."""
    return float(st.session_state.get("monthly_budget", 5000.0))


def set_monthly_budget(amount: float) -> None:
    """Persist the monthly budget."""
    normalized = max(0.0, float(amount))
    st.session_state["monthly_budget"] = normalized
    _persist_state("monthly_budget", normalized)
    _invalidate_chat_cache()


def get_chat_history() -> List[Dict[str, Any]]:
    """Return a copy of the conversational history."""
    history = st.session_state.get("chat_history", [])
    return [dict(message) for message in history]


def set_chat_history(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Persist chat history updates."""
    serialized = [dict(message) for message in messages]
    st.session_state["chat_history"] = serialized
    _persist_state("chat_history", serialized)
    return serialized


def set_analysis_summary(items: Iterable[Dict[str, Any]]) -> None:
    """Persist the analysis summary cards."""
    summary = [dict(item) for item in items]
    st.session_state["analysis_summary"] = summary
    _persist_state("analysis_summary", summary)


def set_product_recommendations(items: Iterable[Dict[str, Any]]) -> None:
    """Persist generated investment recommendations."""
    recommendations = [dict(item) for item in items]
    st.session_state["product_recommendations"] = recommendations
    _persist_state("product_recommendations", recommendations)


def build_chat_cache_key(
    prompt: str,
    transactions: Iterable[Transaction | dict],
    budget: float,
    locale: str,
) -> str:
    """基于提示词+账本+预算生成缓存键，避免跨数据复用。"""

    entries = [_serialize_transaction_entry(txn) for txn in transactions]
    entries.sort(key=lambda item: item.get("id", ""))
    payload = {
        "prompt": prompt.strip(),
        "budget": round(float(budget), 2),
        "locale": locale,
        "transactions": entries,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
