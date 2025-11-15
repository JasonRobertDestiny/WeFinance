"""Conversational financial advisor interface."""

from __future__ import annotations

from typing import List

import streamlit as st

from models.entities import Transaction
from modules.chat_manager import ChatManager
from utils.session import (
    build_chat_cache_key,
    get_chat_history,
    get_i18n,
    get_monthly_budget,
    get_transactions,
    set_chat_history,
)
from utils.ui_components import render_financial_health_card


def _init_session_defaults() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("chat_cache", {})


def render() -> None:
    """Render chat UI backed by ChatManager and GPT-4o."""
    i18n = get_i18n()
    st.title(i18n.t("chat.title"))
    st.write(i18n.t("chat.subtitle"))

    _init_session_defaults()
    history: List[dict] = get_chat_history()
    transactions_list = get_transactions()
    transactions = st.session_state.get("transactions", [])

    current_budget = get_monthly_budget()

    # 显示财务健康卡片（整合预算与支出）
    if transactions_list:
        render_financial_health_card(transactions_list)

    # 智能示例问题（根据数据动态生成，可点击）
    if transactions_list:
        # 分析消费数据生成个性化示例问题
        categories = {}
        for txn in transactions_list:
            cat = txn.category if hasattr(txn, "category") else "其他"
            amount = txn.amount if hasattr(txn, "amount") else 0
            categories[cat] = categories.get(cat, 0) + amount

        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "餐饮"
        total_spent = sum(txn.amount if hasattr(txn, "amount") else 0 for txn in transactions_list)
        remaining = current_budget - total_spent

        # 根据实际数据生成4个示例问题
        sample_questions = [
            f"我这个月{top_category}花了多少？",
            f"还剩{remaining:.0f}元预算，怎么合理安排？",
            f"我的{top_category}消费算高吗？如何优化？",
            "给我一个适合我的理财建议",
        ]
    else:
        # 无数据时显示通用问题
        sample_questions = [
            i18n.t("chat.sample_q1"),
            i18n.t("chat.sample_q2"),
            i18n.t("chat.sample_q3"),
            i18n.t("chat.sample_q4"),
        ]

    # 显示为可点击的按钮组
    st.markdown(f"**{i18n.t('chat.sample_title')}**")
    cols = st.columns(2)
    for idx, question in enumerate(sample_questions):
        with cols[idx % 2]:
            if st.button(question, key=f"sample_q_{idx}", use_container_width=True):
                # 点击后自动发送该问题
                st.session_state["auto_query"] = question
                st.rerun()


    locale = st.session_state.get("locale", "zh_CN")
    chat_manager = ChatManager(
        history=history,
        transactions=transactions,
        monthly_budget=current_budget,
        locale=locale,
    )
    chat_manager.update_transactions(transactions)
    chat_manager.set_monthly_budget(current_budget)

    if history:
        for message in history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        st.info(i18n.t("chat.empty_history"))

    user_prompt = st.chat_input(i18n.t("chat.input_placeholder"))

    # 处理自动查询（来自示例问题点击）
    auto_query = st.session_state.pop("auto_query", None)
    if auto_query:
        user_prompt = auto_query

    if not user_prompt:
        return

    history.append({"role": "user", "content": user_prompt})
    history = set_chat_history(history)
    with st.chat_message("user"):
        st.write(user_prompt)

    cache: dict = st.session_state["chat_cache"]
    cache_key = build_chat_cache_key(
        user_prompt,
        transactions,
        current_budget,
        locale,
    )
    if cache_key in cache:
        cached_reply = cache[cache_key]
        history.append({"role": "assistant", "content": cached_reply})
        history = set_chat_history(history)
        with st.chat_message("assistant"):
            st.write(cached_reply)
            st.caption(i18n.t("common.cache_hit"))
        return

    with st.chat_message("assistant"):
        with st.spinner(i18n.t("common.loading_thinking")):
            response = chat_manager.generate_response(user_prompt)
            cache[cache_key] = response
            if len(cache) > 20:
                first_key = next(iter(cache))
                cache.pop(first_key, None)
            st.write(response)
    history.append({"role": "assistant", "content": response})
    set_chat_history(history)


if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    render()
