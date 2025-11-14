"""Streamlit entry point for the WeFinance Copilot prototype."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable

import streamlit as st

from models.entities import Transaction
from pages import advisor_chat, bill_upload, investment_recs, spending_insights
from modules.analysis import compute_anomaly_report
from utils import session as session_utils
from utils.session import (
    get_i18n,
    init_session_state,
    reset_session_state,
    switch_locale,
)
from utils.storage import clear_all_storage, load_from_storage

logger = logging.getLogger(__name__)


def restore_data_from_storage() -> None:
    """Hydrate Streamlit session state from persisted storage."""
    if st.session_state.get("data_restored", False):
        return

    try:
        transactions_payload = load_from_storage("transactions", []) or []
        if transactions_payload:
            transactions: list[Transaction] = []
            for entry in transactions_payload:
                if isinstance(entry, Transaction):
                    transactions.append(entry)
                elif isinstance(entry, dict):
                    transactions.append(Transaction(**entry))
            st.session_state["transactions"] = [
                txn.model_dump(mode="json") for txn in transactions
            ]
            logger.info("Restored %d transactions from storage", len(transactions))

        budget = load_from_storage("monthly_budget", 5000.0)
        if budget is not None:
            st.session_state["monthly_budget"] = float(budget)

        chat_history = load_from_storage("chat_history", []) or []
        st.session_state["chat_history"] = list(chat_history)

        analysis_summary = load_from_storage("analysis_summary", None)
        if analysis_summary:
            st.session_state["analysis_summary"] = analysis_summary

        product_recommendations = load_from_storage("product_recommendations", None)
        if product_recommendations:
            st.session_state["product_recommendations"] = product_recommendations

        st.session_state["data_restored"] = True
        logger.info("Data restoration completed")
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to restore data from storage: %s", exc)
        st.session_state["data_restored"] = True


restore_data_from_storage()

st.set_page_config(
    page_title="WeFinance Copilot",
    page_icon="💰",
    layout="wide",
    menu_items={
        "Get help": "mailto:team@wefinance.ai",
        "Report a bug": "https://github.com/wefinance/issues",
        "About": "WeFinance Copilot — 让AI成为你的私人CFO。",
    },
)


def _render_home() -> None:
    """Render the landing page with value proposition and card-based navigation."""
    i18n = get_i18n()

    # Purple gradient value proposition banner
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        ">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">
                {i18n.t("app.value_banner_title")}
            </h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                {i18n.t("app.value_banner_subtitle")}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Anomaly warnings (prioritized above metrics)
    active_anomalies = session_utils.get_active_anomalies()
    anomaly_message = st.session_state.get("anomaly_message", "")

    if active_anomalies:
        st.error(i18n.t("app.anomaly_warning"))
        for anomaly in active_anomalies[:3]:
            date_str = anomaly.get("date", "-")
            merchant = anomaly.get(
                "merchant",
                i18n.t("common.unknown_merchant"),
            )
            amount = anomaly.get("amount", 0.0)
            reason = anomaly.get("reason", "")
            with st.container():
                st.markdown(
                    f"**{i18n.t('app.anomaly_info', date=date_str, merchant=merchant, amount=float(amount))}**"
                )
                if reason:
                    st.caption(reason)
                cols = st.columns(2)
                confirm_key = f"home_confirm_{anomaly['transaction_id']}"
                fraud_key = f"home_fraud_{anomaly['transaction_id']}"

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
        st.divider()
    elif anomaly_message:
        st.info(anomaly_message)

    # Calculate metrics
    transactions = session_utils.get_transactions()
    monthly_budget = session_utils.get_monthly_budget()
    chat_history = st.session_state.get("chat_history", [])

    total_spent = sum(txn.amount for txn in transactions)
    budget_remaining = monthly_budget - total_spent

    # 3-column metrics cards with direct navigation buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=i18n.t("app.metric_transactions"),
            value=i18n.t("app.metric_transactions_unit", count=len(transactions))
        )
        if st.button(
            i18n.t("app.btn_upload_bills"),
            key="home_upload_btn",
            type="primary",
            use_container_width=True
        ):
            st.session_state["selected_page"] = "bill_upload"
            st.rerun()

    with col2:
        st.metric(
            label=i18n.t("app.metric_budget_remaining"),
            value=f"¥{budget_remaining:,.0f}",
            delta=i18n.t("app.metric_budget_spent", spent=total_spent)
        )
        if st.button(
            i18n.t("app.btn_view_analysis"),
            key="home_analysis_btn",
            use_container_width=True
        ):
            st.session_state["selected_page"] = "spending_insights"
            st.rerun()

    with col3:
        st.metric(
            label=i18n.t("app.metric_chat_history"),
            value=i18n.t("app.metric_chat_unit", count=len(chat_history))
        )
        if st.button(
            i18n.t("app.btn_start_chat"),
            key="home_chat_btn",
            use_container_width=True
        ):
            st.session_state["selected_page"] = "advisor_chat"
            st.rerun()

    # Comparison table
    st.markdown("---")
    st.subheader(i18n.t("app.comparison_title"))

    import pandas as pd
    comparison_data = {
        i18n.t("app.comparison_feature"): [
            i18n.t("app.comparison_ocr_rate"),
            i18n.t("app.comparison_processing"),
            i18n.t("app.comparison_multilingual"),
            i18n.t("app.comparison_ai_advisor"),
        ],
        i18n.t("app.comparison_wefinance"): [
            i18n.t("app.comparison_ocr_wefinance"),
            i18n.t("app.comparison_processing_wefinance"),
            i18n.t("app.comparison_multilingual_wefinance"),
            i18n.t("app.comparison_ai_wefinance"),
        ],
        i18n.t("app.comparison_traditional"): [
            i18n.t("app.comparison_ocr_traditional"),
            i18n.t("app.comparison_processing_traditional"),
            i18n.t("app.comparison_multilingual_traditional"),
            i18n.t("app.comparison_ai_traditional"),
        ],
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # Get recommendations button
    st.markdown("---")
    col_invest = st.columns([1, 2, 1])[1]
    with col_invest:
        if st.button(
            i18n.t("app.btn_get_recommendations"),
            key="home_invest_btn",
            type="secondary",
            use_container_width=True
        ):
            st.session_state["selected_page"] = "investment_recs"
            st.rerun()

    # Project info footer
    st.markdown("---")
    st.info(i18n.t("app.info_note"))


PAGES: dict[str, Callable[[], None]] = {
    "home": _render_home,
    "bill_upload": bill_upload.render,
    "spending_insights": spending_insights.render,
    "advisor_chat": advisor_chat.render,
    "investment_recs": investment_recs.render,
}


def _refresh_anomaly_state() -> None:
    """Recompute anomaly detection results to keep homepage alerts up-to-date."""
    i18n = get_i18n()
    transactions = session_utils.get_transactions()
    if not transactions:
        session_utils.update_anomaly_state(
            active=[],
            message=i18n.t("app.no_transaction_data"),
        )
        return

    report = compute_anomaly_report(
        transactions,
        whitelist_merchants=session_utils.get_trusted_merchants(),
    )
    session_utils.sync_anomaly_state(report)


def main() -> None:
    """Application bootstrap."""
    init_session_state()
    _refresh_anomaly_state()
    i18n = get_i18n()

    with st.sidebar:
        st.header(i18n.t("app.sidebar_header"))

        locale_labels = {
            "zh_CN": i18n.t("app.locale_zh"),
            "en_US": i18n.t("app.locale_en"),
        }
        current_locale = st.session_state.get("locale", "zh_CN")
        locale_display = locale_labels.get(current_locale, "中文")
        selected_display = st.selectbox(
            i18n.t("app.language_label"),
            options=list(locale_labels.values()),
            index=list(locale_labels.values()).index(locale_display),
        )
        selected_locale = next(
            key for key, value in locale_labels.items() if value == selected_display
        )
        if selected_locale != current_locale:
            switch_locale(selected_locale)
            st.rerun()

        nav_labels = {
            "home": i18n.t("navigation.home"),
            "bill_upload": i18n.t("navigation.bill_upload"),
            "spending_insights": i18n.t("navigation.spending_insights"),
            "advisor_chat": i18n.t("navigation.advisor_chat"),
            "investment_recs": i18n.t("navigation.investment_recs"),
        }
        radio_options = list(nav_labels.values())
        selected_page = st.session_state.get("selected_page", "home")
        if selected_page not in nav_labels:
            selected_page = "home"
        default_index = radio_options.index(nav_labels[selected_page])
        selection_label = st.radio(
            i18n.t("app.navigation_label"),
            radio_options,
            index=default_index,
        )
        selection = next(
            key for key, value in nav_labels.items() if value == selection_label
        )
        st.session_state["selected_page"] = selection

        # ============ 全局Budget设置 (Global Budget Setting) ============
        st.markdown("---")
        st.markdown(f"**{i18n.t('app.global_settings_title')}**")

        # 获取当前budget (Get current budget)
        current_budget = session_utils.get_monthly_budget()

        # Budget输入框 (Budget input)
        new_budget = st.number_input(
            i18n.t("app.monthly_budget_label"),
            min_value=0.0,
            max_value=1000000.0,
            value=float(current_budget),
            step=500.0,
            format="%.0f",
            help=i18n.t("app.monthly_budget_help"),
            key="global_budget_sidebar"
        )

        # 更新session state (Update session state)
        if new_budget != current_budget:
            session_utils.set_monthly_budget(new_budget)
            st.toast(i18n.t("app.budget_updated"))

        # 显示当前预算（带格式化）(Display current budget with formatting)
        latest_budget = session_utils.get_monthly_budget()
        st.caption(
            f"💰 {i18n.t('app.current_budget_display', budget=f'¥{latest_budget:,.0f}')}"
        )

        st.markdown("---")
        st.markdown(f"**{i18n.t('app.data_management_title')}**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                i18n.t("app.export_data"),
                help=i18n.t("app.export_data_help"),
                key="export_data_btn",
                use_container_width=True,
            ):
                export_data = {
                    "transactions": load_from_storage("transactions", []),
                    "monthly_budget": load_from_storage("monthly_budget", 5000.0),
                    "chat_history": load_from_storage("chat_history", []),
                    "analysis_summary": load_from_storage("analysis_summary", []),
                    "product_recommendations": load_from_storage(
                        "product_recommendations", []
                    ),
                    "export_time": datetime.now().isoformat(),
                }
                json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label=i18n.t("app.download_json"),
                    data=json_data,
                    file_name=f"wefinance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json_btn",
                    use_container_width=True,
                )

        with col2:
            if st.button(
                i18n.t("app.clear_data"),
                help=i18n.t("app.clear_data_help"),
                key="clear_data_btn",
                type="secondary",
                use_container_width=True,
            ):
                if st.session_state.get("confirm_clear", False):
                    clear_all_storage()
                    protected_keys = {"selected_page", "locale", "data_restored"}
                    for state_key in list(st.session_state.keys()):
                        if state_key not in protected_keys:
                            del st.session_state[state_key]
                    st.toast(i18n.t("app.data_cleared"))
                    st.session_state["confirm_clear"] = False
                    st.session_state["data_restored"] = False
                    st.rerun()
                else:
                    st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.warning(i18n.t("app.confirm_clear_warning"))

        st.markdown("---")

        if st.button(i18n.t("app.reset")):
            reset_session_state()
            st.success(i18n.t("app.reset_success"))
            st.rerun()

    render = PAGES.get(selection)
    if render is None:
        st.error(i18n.t("errors.page_missing"))
        return

    try:
        render()
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("页面渲染失败：%s", exc)
        st.error(f"😥 {i18n.t('errors.render_failed')}")
        st.stop()


if __name__ == "__main__":
    main()
