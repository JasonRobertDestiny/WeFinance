"""Conversational financial advisor interface."""

from __future__ import annotations

from typing import List

import streamlit as st

from modules.chat_manager import ChatManager
from utils.session import get_i18n, get_monthly_budget


def _init_session_defaults() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("chat_cache", {})


def render() -> None:
    """Render chat UI backed by ChatManager and GPT-4o."""
    i18n = get_i18n()
    st.title(i18n.t("chat.title"))
    st.write(i18n.t("chat.subtitle"))

    _init_session_defaults()
    history: List[dict] = st.session_state["chat_history"]
    transactions = st.session_state.get("transactions", [])

    current_budget = get_monthly_budget()

    # 显示当前使用的budget（提示可在侧边栏修改）
    # Display current budget (hint: can be changed in sidebar)
    st.info(f"💰 {i18n.t('chat.current_budget_info', budget=f'¥{current_budget:,.0f}')}")

    # 示例问题（现在占全宽）(Sample questions - now full width)
    st.markdown(
        "\n".join(
            [
                f"**{i18n.t('chat.sample_title')}**",
                f"- {i18n.t('chat.sample_q1')}",
                f"- {i18n.t('chat.sample_q2')}",
                f"- {i18n.t('chat.sample_q3')}",
                f"- {i18n.t('chat.sample_q4')}",
            ]
        )
    )

    chat_manager = ChatManager(
        history=history,
        transactions=transactions,
        monthly_budget=current_budget,
        locale=st.session_state.get("locale", "zh_CN"),
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
    if not user_prompt:
        return

    history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    cache: dict = st.session_state["chat_cache"]
    if user_prompt in cache:
        cached_reply = cache[user_prompt]
        history.append({"role": "assistant", "content": cached_reply})
        with st.chat_message("assistant"):
            st.write(cached_reply)
            st.caption(i18n.t("common.cache_hit"))
        return

    with st.chat_message("assistant"):
        with st.spinner(i18n.t("common.loading_thinking")):
            response = chat_manager.generate_response(user_prompt)
            cache[user_prompt] = response
            if len(cache) > 20:
                first_key = next(iter(cache))
                cache.pop(first_key, None)
            st.write(response)
