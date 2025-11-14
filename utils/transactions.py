"""Transaction helper utilities for consistent identifiers."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any


def _normalize_date(value: Any) -> str:
    """把多种日期类型统一成字符串，保证哈希稳定。"""

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def generate_transaction_id(
    *,
    merchant: str,
    date_value: Any,
    amount: float,
    currency: str = "CNY",
    source_hash: str | None = None,
    sequence: int | None = None,
) -> str:
    """生成稳定的交易ID，避免重复上传造成ID漂移。"""

    merchant_key = (merchant or "").strip().lower()
    normalized_date = _normalize_date(date_value)
    currency_key = (currency or "CNY").strip().upper()
    parts = [merchant_key, normalized_date, f"{float(amount):.2f}", currency_key]
    if source_hash:
        parts.append(source_hash)
    if sequence is not None:
        parts.append(str(sequence))
    payload = "|".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


__all__ = ["generate_transaction_id"]
