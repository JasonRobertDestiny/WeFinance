"""Data models shared across modules."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """Single line item in a receipt (商品明细)."""

    description: str = Field(..., description="商品描述，如 'RF MODELLING CLAY'")
    quantity: float = Field(default=1.0, description="数量")
    unit_price: float = Field(..., description="单价")
    amount: float = Field(..., description="小计金额")
    discount: Optional[float] = Field(default=None, description="单项折扣金额")


class Transaction(BaseModel):
    """Normalized transaction record parsed from OCR or CSV."""

    id: str = Field(..., description="Deterministic identifier (uuid4 or hash).")
    date: dt.date = Field(..., description="Transaction posting date.")
    merchant: str = Field(..., description="Merchant or counterparty name.")
    category: str = Field(..., description="Spending category label.")
    amount: float = Field(..., description="Positive numbers represent expenses.")
    currency: str = Field(
        default="CNY", description="Currency code in ISO-4217 format."
    )
    payment_method: Optional[str] = Field(
        default=None, description="Payment instrument such as 微信支付."
    )
    raw_text: Optional[str] = Field(
        default=None, description="Original OCR text snippet for explainability."
    )

    # === 扩展字段：支持真实收据的商品明细 ===
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="商品明细列表。简单账单为空，真实收据包含详细商品。",
    )
    subtotal: Optional[float] = Field(
        default=None, description="小计金额（折扣前）"
    )
    total_discount: Optional[float] = Field(
        default=None, description="总折扣金额"
    )
    tax: Optional[float] = Field(default=None, description="税费")
    receipt_number: Optional[str] = Field(
        default=None, description="收据编号，如 'TD01167104'"
    )
    receipt_time: Optional[dt.datetime] = Field(
        default=None, description="交易精确时间（包含时分秒）"
    )


class SpendingInsight(BaseModel):
    """High-level summary of spending behaviour for the dashboard."""

    title: str
    detail: str
    actions: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations with quantified benefits."
    )
    delta: Optional[float] = Field(
        default=None, description="Month-over-month delta expressed as a percentage."
    )


class Recommendation(BaseModel):
    """Financial product or budgeting recommendation."""

    title: str
    summary: str
    rationale_steps: List[str] = Field(
        default_factory=list, description="Explainable reasoning chain."
    )
    risk_level: Optional[str] = Field(
        default=None, description="Optional qualitative risk label."
    )


class OCRParseResult(BaseModel):
    """Full OCR pipeline outcome for a single uploaded file."""

    filename: str
    text: str
    transactions: List[Transaction] = Field(default_factory=list)
