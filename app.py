"""Streamlit entry point for the WeFinance Copilot prototype."""

from __future__ import annotations

import logging
from typing import Callable

import streamlit as st

from pages import advisor_chat, bill_upload, investment_recs, spending_insights
from modules.analysis import compute_anomaly_report
from utils import session as session_utils
from utils.session import (
    get_i18n,
    init_session_state,
    reset_session_state,
    switch_locale,
)

logger = logging.getLogger(__name__)

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
    """Render the landing page with quick project hints."""
    i18n = get_i18n()
    st.title(i18n.t("app.title"))
    st.subheader(i18n.t("app.subtitle"))

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

    # 定义4步理财流程 (4-step financial planning workflow)
    steps = [
        {
            "id": "upload",
            "name": i18n.t("app.step_upload_bills"),
            "page_key": "bill_upload",
            "hint": i18n.t("app.hint_upload_bills"),
            "done": bool(session_utils.get_transactions()),
            "icon": "📸"
        },
        {
            "id": "insights",
            "name": i18n.t("app.step_view_insights"),
            "page_key": "spending_insights",
            "hint": i18n.t("app.hint_view_insights"),
            "done": bool(st.session_state.get("analysis_summary")),
            "icon": "📊"
        },
        {
            "id": "chat",
            "name": i18n.t("app.step_chat_advisor"),
            "page_key": "advisor_chat",
            "hint": i18n.t("app.hint_chat_advisor"),
            "done": len(st.session_state.get("chat_history", [])) > 0,
            "icon": "💬"
        },
        {
            "id": "invest",
            "name": i18n.t("app.step_get_recommendations"),
            "page_key": "investment_recs",
            "hint": i18n.t("app.hint_get_recommendations"),
            "done": bool(st.session_state.get("product_recommendations")),
            "icon": "💰"
        },
    ]

    # 渲染进度引导卡片 (Render progress guide cards)
    st.markdown("---")
    st.subheader(i18n.t("app.progress_guide_title"))
    st.caption(i18n.t("app.progress_guide_subtitle"))

    for step in steps:
        col1, col2, col3 = st.columns([0.08, 0.8, 0.12])

        with col1:
            # 完成状态图标 (Completion status icon)
            if step["done"]:
                st.markdown("✅")
            else:
                st.markdown("⭕")

        with col2:
            # 步骤名称和提示 (Step name and hint)
            st.markdown(f"{step['icon']} **{step['name']}**")

            if not step["done"]:
                st.caption(step["hint"])

        with col3:
            # 只为第一个未完成步骤显示按钮 (Only show button for first incomplete step)
            if not step["done"]:
                button_key = f"goto_{step['id']}"
                if st.button(i18n.t("app.btn_go"), key=button_key, type="primary"):
                    # 设置选中页面（通过侧边栏导航机制）
                    st.session_state["selected_page"] = step["page_key"]
                    st.rerun()

                # 只显示第一个未完成步骤的按钮 (Show only first incomplete button)
                break

    # 如果全部完成，显示鼓励信息 (Show celebration when all complete)
    if all(step["done"] for step in steps):
        st.success(i18n.t("app.all_steps_completed"))
        st.balloons()  # 庆祝动画 (Celebration animation)

    st.markdown(
        "\n".join(
            [
                i18n.t("app.welcome"),
                "",
                i18n.t("app.feature_bill"),
                i18n.t("app.feature_analysis"),
                i18n.t("app.feature_chat"),
                i18n.t("app.feature_recs"),
            ]
        )
    )
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
        current_budget = st.session_state.get("monthly_budget", 5000.0)

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
            st.session_state["monthly_budget"] = new_budget
            st.toast(i18n.t("app.budget_updated"))

        # 显示当前预算（带格式化）(Display current budget with formatting)
        st.caption(f"💰 {i18n.t('app.current_budget_display', budget=f'¥{new_budget:,.0f}')}")

        st.markdown("---")

        if st.button(i18n.t("app.reset")):
            reset_session_state()
            st.success(i18n.t("app.reset_success"))
            st.rerun()

        st.divider()
        st.caption(
            f"{i18n.t('app.sidebar_version', version='0.1.0')}\n"
            f"{i18n.t('app.sidebar_competition')}\n"
            f"{i18n.t('app.sidebar_goal')}"
        )

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
