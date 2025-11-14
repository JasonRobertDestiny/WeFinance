"""Investment recommendation and explainability view."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from models.entities import Recommendation, Transaction
from services.recommendation_service import RecommendationService
from utils import session as session_utils
from utils.session import get_i18n, set_product_recommendations

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
    result = service.generate(
        transactions=transactions,
        responses=responses,
        investment_goal=goal,
        locale=locale,
    )
    recs = result.get("recommendations", [])
    serialized = []
    for rec in recs:
        if isinstance(rec, Recommendation):
            serialized.append(rec.model_dump())
        elif isinstance(rec, dict):
            serialized.append(rec)
    result["recommendations"] = serialized
    return result


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


def _render_results(results: Dict[str, object]) -> None:
    i18n = get_i18n()
    recommendations_raw = results.get("recommendations", [])
    recommendations = [
        rec
        if isinstance(rec, Recommendation)
        else Recommendation(**rec)
        for rec in recommendations_raw
    ]
    profile: Dict[str, object] = results.get("financial_profile", {})  # type: ignore[assignment]
    risk_level: str = results.get("risk_level", "")  # type: ignore[assignment]

    st.success(i18n.t("recommendation.risk_result", risk=risk_level))

    st.subheader(i18n.t("recommendation.financial_profile_title"))
    col1, col2, col3 = st.columns(3)
    monthly_avg = float(profile.get("monthly_average", 0.0) or 0.0)
    volatility = float(profile.get("spending_volatility", 0.0) or 0.0)
    investable = float(profile.get("investable_amount", 0.0) or 0.0)
    with col1:
        st.metric(
            i18n.t("recommendation.metric_monthly_avg"), f"¥{monthly_avg:,.0f}"
        )
    with col2:
        st.metric(
            i18n.t("recommendation.metric_investable"), f"¥{investable:,.0f}"
        )
    with col3:
        st.metric(
            i18n.t("recommendation.metric_volatility"), f"{volatility*100:.1f}%"
        )

    breakdown: Dict[str, float] = profile.get("category_breakdown", {})  # type: ignore[assignment]
    if breakdown:
        st.subheader(i18n.t("recommendation.category_breakdown_title"))
        df = pd.DataFrame(
            [{"category": cat, "share": share * 100} for cat, share in breakdown.items()]
        )
        fig = px.bar(
            df,
            x="category",
            y="share",
            text="share",
            labels={
                "category": i18n.t("spending.label_category"),
                "share": i18n.t("recommendation.label_ratio"),
            },
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_title=i18n.t("recommendation.label_ratio"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(i18n.t("recommendation.recommendation_list_title"))
    for rec in recommendations:
        st.markdown(f"### {rec.title}")
        st.write(rec.summary)
        for idx, step in enumerate(rec.rationale_steps, start=1):
            st.write(f"{idx}. {step}")


def render() -> None:
    """Render investment recommendation workflow with XAI explanation."""
    i18n = get_i18n()
    st.title(i18n.t("recommendation.title"))
    st.write(i18n.t("recommendation.subtitle"))

    transactions = session_utils.get_transactions()
    if not transactions:
        st.warning(i18n.t("recommendation.require_upload"))
        return

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
        recommendation_payload = [dict(item) for item in results["recommendations"]]
        set_product_recommendations(recommendation_payload)
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
