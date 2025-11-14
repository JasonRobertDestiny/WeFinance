"""Streamlit page for spending analytics and visualisations."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from models.entities import SpendingInsight, Transaction
from modules.analysis import (
    calculate_category_totals,
    calculate_spending_trend,
    compute_anomaly_report,
    generate_insights,
)
from utils import session as session_utils


@st.cache_data(show_spinner=False)
def _prepare_dashboard_data(
    transactions_dump: Tuple[Tuple[Tuple[str, object], ...], ...],
    whitelist: Tuple[str, ...],
    base_threshold: float,
) -> dict:
    """Pre-compute analytics outputs for the dashboard."""
    transactions = [Transaction(**dict(entry)) for entry in transactions_dump]

    category_totals = calculate_category_totals(transactions)
    trend_daily = calculate_spending_trend(transactions, frequency="D")
    trend_monthly = calculate_spending_trend(transactions, frequency="M")
    anomaly_report = compute_anomaly_report(
        transactions,
        base_threshold=base_threshold,
        whitelist_merchants=whitelist,
    )
    insights = generate_insights(transactions)

    return {
        "category_totals": category_totals,
        "trend_daily": trend_daily,
        "trend_monthly": trend_monthly,
        "anomaly_report": anomaly_report,
        "insights": [ins.model_dump() for ins in insights],
    }


def _render_active_anomalies(
    anomalies: List[dict], threshold_used: float, i18n
) -> None:
    """Display active anomalies with action buttons."""
    if anomalies:
        st.subheader(f"⚠️ {i18n.t('spending.anomaly_pending')}")
        st.caption(
            i18n.t("spending.anomaly_threshold", threshold=f"{threshold_used:.1f}")
        )
    for anomaly in anomalies:
        date_str = anomaly.get("date") or "-"
        merchant = anomaly.get("merchant", "未知商户")
        amount = anomaly.get("amount", 0.0)
        reason = anomaly.get("reason", "")
        status = anomaly.get("status", "new")

        box = st.warning if status == "new" else st.info
        with box(
            f"{i18n.t('app.anomaly_info', date=date_str, merchant=merchant, amount=float(amount))}"
        ):
            if reason:
                st.caption(reason)
            cols = st.columns(2)
            confirm_key = f"confirm_{anomaly['transaction_id']}"
            fraud_key = f"fraud_{anomaly['transaction_id']}"

            if cols[0].button(i18n.t("common.btn_confirm"), key=confirm_key):
                session_utils.record_anomaly_feedback(anomaly, "confirmed")
                remaining = [
                    item
                    for item in session_utils.get_active_anomalies()
                    if item.get("transaction_id") != anomaly["transaction_id"]
                ]
                session_utils.update_anomaly_state(active=remaining)
                st.toast(i18n.t("common.toast_confirmed"))
                st.rerun()

            if cols[1].button(i18n.t("common.btn_mark_fraud"), key=fraud_key):
                session_utils.record_anomaly_feedback(anomaly, "fraud")
                remaining = [
                    item
                    for item in session_utils.get_active_anomalies()
                    if item.get("transaction_id") != anomaly["transaction_id"]
                ]
                session_utils.update_anomaly_state(active=remaining)
                st.toast(i18n.t("common.toast_fraud"))
                st.rerun()


def _render_sidebar_controls(trusted_merchants: List[str], i18n) -> None:
    """Render merchant whitelist management and anomaly history in sidebar."""
    with st.sidebar.expander(i18n.t("spending.trusted_manager"), expanded=False):
        with st.form("trusted_merchants_form"):
            new_merchant = st.text_input(i18n.t("spending.trusted_add_placeholder"))
            added = st.form_submit_button(i18n.t("spending.trusted_add"))
            if added:
                session_utils.add_trusted_merchant(new_merchant)
                st.toast(i18n.t("common.toast_added", name=new_merchant))
                st.rerun()

        if trusted_merchants:
            st.caption(i18n.t("spending.trusted_list_title"))
            for idx, merchant in enumerate(trusted_merchants, start=1):
                cols = st.columns([0.8, 0.2])
                cols[0].write(f"{idx}. {merchant}")
                if cols[1].button("✖", key=f"remove_whitelist_{idx}"):
                    session_utils.remove_trusted_merchant(merchant)
                    st.toast(i18n.t("common.toast_removed", name=merchant))
                    st.rerun()
        else:
            st.info(i18n.t("spending.trusted_empty"))

    history = session_utils.get_anomaly_history()
    with st.sidebar.expander(i18n.t("spending.history_title"), expanded=False):
        if not history:
            st.write(i18n.t("spending.history_empty"))
        else:
            for record in history:
                merchant = record.get("merchant", "未知商户")
                amount = record.get("amount", 0.0)
                status = record.get("status", "confirmed")
                date_str = record.get("date", "-")
                label = (
                    "✅ " + i18n.t("common.btn_confirm")
                    if status == "confirmed"
                    else "🚨 " + i18n.t("common.btn_mark_fraud")
                )
                st.write(f"{date_str} | {merchant} | ¥{amount:.2f} | {label}")


def render() -> None:
    """Render enhanced analytics dashboard with Plotly visualisations."""
    i18n = session_utils.get_i18n()
    st.title(i18n.t("spending.title"))
    st.write(i18n.t("spending.description"))

    transactions = session_utils.get_transactions()
    if not transactions:
        st.warning(i18n.t("spending.require_upload"))
        return

    trusted_merchants = session_utils.get_trusted_merchants()
    _render_sidebar_controls(trusted_merchants, i18n)

    serialized = tuple(
        tuple(sorted(tx.model_dump().items(), key=lambda item: item[0]))
        for tx in transactions
    )
    whitelist_tuple = tuple(sorted(trusted_merchants))
    results = _prepare_dashboard_data(serialized, whitelist_tuple, base_threshold=2.5)

    totals = results["category_totals"]
    trend_daily: pd.DataFrame = results["trend_daily"]
    trend_monthly: pd.DataFrame = results["trend_monthly"]
    anomaly_report = results["anomaly_report"]
    insights_payload = results["insights"]
    insights = [SpendingInsight(**ins) for ins in insights_payload]

    active_anomalies = session_utils.sync_anomaly_state(anomaly_report)

    anomaly_message = anomaly_report.get("message")
    if anomaly_message:
        st.info(i18n.t(anomaly_message))

    _render_active_anomalies(
        active_anomalies, anomaly_report.get("threshold_used", 2.5), i18n
    )

    if totals:
        with st.expander(i18n.t("spending.category_title"), expanded=False):
            pie_df = pd.DataFrame(
                [{"category": cat, "amount": amt} for cat, amt in totals.items()]
            )
            fig_pie = px.pie(
                pie_df,
                names="category",
                values="amount",
                title="分类占比",
                hole=0.4,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

            bar_df = pie_df.sort_values("amount", ascending=False)
            fig_bar = px.bar(
                bar_df,
                x="category",
                y="amount",
                text="amount",
                title=i18n.t("spending.category_title"),
                labels={
                    "category": i18n.t("spending.label_category"),
                    "amount": i18n.t("spending.label_amount"),
                },
            )
            fig_bar.update_traces(texttemplate="¥%{text:.2f}", textposition="outside")
            fig_bar.update_layout(yaxis_title=i18n.t("spending.label_amount"))
            st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander(i18n.t("spending.trend_title"), expanded=False):
        if not trend_daily.empty:
            fig_line = px.line(
                trend_daily,
                x="period",
                y="amount",
                markers=True,
                title=i18n.t("spending.trend_title"),
                labels={
                    "period": i18n.t("spending.label_date"),
                    "amount": i18n.t("spending.label_amount"),
                },
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info(i18n.t("spending.trend_daily_empty"))

        if not trend_monthly.empty and len(trend_monthly) > 1:
            fig_month = px.line(
                trend_monthly,
                x="period",
                y="amount",
                markers=True,
                title=i18n.t("spending.trend_title"),
                labels={
                    "period": i18n.t("spending.label_month"),
                    "amount": i18n.t("spending.label_amount"),
                },
            )
            st.plotly_chart(fig_month, use_container_width=True)

    with st.expander(i18n.t("spending.insight_title"), expanded=True):
        if insights:
            for insight in insights:
                st.markdown(f"### {insight.title}")
                st.write(insight.detail)
                if insight.actions:
                    st.markdown("**💡 行动建议：**")
                    for idx, action in enumerate(insight.actions, 1):
                        st.markdown(f"{idx}. {action}")
                st.markdown("---")
        else:
            st.info(i18n.t("spending.insight_none"))


if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    render()
