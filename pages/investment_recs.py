"""Investment recommendation and explainability view."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from models.entities import Recommendation, Transaction
from services.recommendation_service import RecommendationService
from utils.session import get_i18n

RISK_QUESTIONS: List[Dict[str, object]] = [
    {
        "id": "q1",
        "prompt": "您能接受的最大亏损是多少？",
        "options": [
            ("5%以内，几乎不能亏损", 1),
            ("10%左右，可接受一定波动", 2),
            ("20%以上，只要长期有收益", 3),
        ],
    },
    {
        "id": "q2",
        "prompt": "您的投资期限是多久？",
        "options": [
            ("1年以内，需要资金的流动性", 1),
            ("1-3年，可以阶段性锁定资金", 2),
            ("3年以上，长期增值为主", 3),
        ],
    },
    {
        "id": "q3",
        "prompt": "您对投资波动的态度如何？",
        "options": [
            ("波动让我焦虑，尽量避免", 1),
            ("适度波动可以接受", 2),
            ("波动越大越有机会", 3),
        ],
    },
]


def _coerce_transactions(transactions_raw: Iterable[object]) -> List[Transaction]:
    normalized: List[Transaction] = []
    for entry in transactions_raw:
        if isinstance(entry, Transaction):
            normalized.append(entry)
        elif isinstance(entry, dict):
            normalized.append(Transaction(**entry))
    return normalized


@st.cache_data(show_spinner=False)
def _generate_cached_recommendation(
    transactions_dump: Tuple[Tuple[Tuple[str, object], ...], ...],
    responses_tuple: Tuple[Tuple[str, int], ...],
    goal: str,
    locale: str,
) -> Dict[str, object]:
    """Cacheable wrapper producing recommendation payload."""
    service = RecommendationService()
    transactions = [Transaction(**dict(entry)) for entry in transactions_dump]
    responses = dict(responses_tuple)
    return service.generate(
        transactions=transactions,
        responses=responses,
        investment_goal=goal,
        locale=locale,
    )


def _collect_risk_answers() -> Tuple[Dict[str, int], str]:
    i18n = get_i18n()
    st.subheader(i18n.t("recommendation.step1"))
    answers: Dict[str, int] = {}
    for question in RISK_QUESTIONS:
        key = f"risk_{question['id']}"
        label = i18n.t(
            {
                "q1": "recommendation.question_loss",
                "q2": "recommendation.question_term",
                "q3": "recommendation.question_risk",
            }[question["id"]]
        )
        options: List[Tuple[str, int]] = question["options"]  # type: ignore[assignment]
        option_labels = [
            i18n.t(
                {
                    ("q1", 1): "recommendation.option_loss_low",
                    ("q1", 2): "recommendation.option_loss_mid",
                    ("q1", 3): "recommendation.option_loss_high",
                    ("q2", 1): "recommendation.option_term_short",
                    ("q2", 2): "recommendation.option_term_mid",
                    ("q2", 3): "recommendation.option_term_long",
                    ("q3", 1): "recommendation.option_risk_low",
                    ("q3", 2): "recommendation.option_risk_mid",
                    ("q3", 3): "recommendation.option_risk_high",
                }[(question["id"], score)]
            )
            for opt_label, score in options
        ]
        default_index = 0
        selected = st.radio(
            label, options=option_labels, index=default_index, key=key, horizontal=False
        )
        for (opt_label, score), localised in zip(options, option_labels):
            if localised == selected:
                answers[question["id"]] = score
                break

    st.subheader(i18n.t("recommendation.step2"))
    goal = st.text_input(
        i18n.t("recommendation.prompt_goal"),
        placeholder=i18n.t("recommendation.goal_placeholder"),
        key="investment_goal",
    )
    return answers, goal


def _render_allocation_chart(allocation: Dict[str, float]) -> None:
    i18n = get_i18n()
    asset_col = i18n.t("recommendation.label_asset")
    ratio_col = i18n.t("recommendation.label_ratio")
    allocation_df = pd.DataFrame(
        {
            asset_col: list(allocation.keys()),
            ratio_col: [value * 100 for value in allocation.values()],
        }
    )
    fig = px.pie(
        allocation_df,
        names=asset_col,
        values=ratio_col,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def _render_results(results: Dict[str, object]) -> None:
    i18n = get_i18n()
    recommendation: Recommendation = results["recommendation"]  # type: ignore[assignment]
    explanation: str = results["explanation"]  # type: ignore[assignment]
    metrics: Dict[str, float] = results["metrics"]  # type: ignore[assignment]
    allocation: Dict[str, float] = results["allocation"]  # type: ignore[assignment]
    risk_level: str = results["risk_level"]  # type: ignore[assignment]

    st.success(i18n.t("recommendation.risk_result", risk=risk_level))
    st.markdown(
        i18n.t("recommendation.core_suggestion", summary=recommendation.summary)
    )

    st.subheader(i18n.t("recommendation.allocation_title"))
    _render_allocation_chart(allocation)

    st.subheader(i18n.t("recommendation.metrics_title"))
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            i18n.t("recommendation.metric_return"), f"{metrics['expected_return']:.1f}%"
        )
    with col2:
        st.metric(
            i18n.t("recommendation.metric_drawdown"), f"{metrics['max_drawdown']:.1f}%"
        )

    st.subheader(i18n.t("recommendation.steps_title"))
    for idx, step in enumerate(recommendation.rationale_steps, start=1):
        st.write(f"{idx}. {step}")

    with st.expander(i18n.t("recommendation.xai_title"), expanded=False):
        st.markdown(explanation.replace("\n", "  \n"))


def render() -> None:
    """Render investment recommendation workflow with XAI explanation."""
    i18n = get_i18n()
    st.title(i18n.t("recommendation.title"))
    st.write(i18n.t("recommendation.subtitle"))

    transactions_raw = st.session_state.get("transactions", [])
    transactions = _coerce_transactions(transactions_raw)

    answers, goal = _collect_risk_answers()
    responses_tuple = tuple(sorted(answers.items()))
    transactions_dump = tuple(
        tuple(sorted(tx.model_dump().items(), key=lambda item: item[0]))
        for tx in transactions
    )

    st.subheader(i18n.t("recommendation.step3"))
    if st.button(i18n.t("recommendation.button_generate"), type="primary"):
        try:
            with st.spinner(i18n.t("common.loading_recommendation")):
                results = _generate_cached_recommendation(
                    transactions_dump,
                    responses_tuple,
                    goal,
                    st.session_state.get("locale", "zh_CN"),
                )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"{i18n.t('errors.structuring_fail')} ({exc})")
            return

        _render_results(results)
        # Persist to session for downstream usage or export.
        st.session_state["product_recommendations"] = [
            results["recommendation"].model_dump()  # type: ignore[attr-defined]
        ]
        st.session_state["recommendation_explanation"] = results
    else:
        cached = st.session_state.get("recommendation_explanation")
        if cached:
            st.info(i18n.t("recommendation.history_loaded"))
            _render_results(cached)
        else:
            st.info(i18n.t("recommendation.info_wait"))


if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    render()
