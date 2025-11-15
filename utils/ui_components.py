"""可复用的UI组件，用于在各个页面之间保持一致的视觉体验"""

from __future__ import annotations

from typing import List

import streamlit as st

from models.entities import Transaction
from utils.session import get_i18n, get_monthly_budget


def render_financial_health_card(transactions: List[Transaction]) -> None:
    """
    渲染财务健康卡片（顶部状态栏）

    显示：
    - 月度预算
    - 本月支出
    - 剩余预算
    - 预算使用率（带颜色编码）

    Args:
        transactions: 交易记录列表
    """
    i18n = get_i18n()
    budget = get_monthly_budget()

    # 计算支出总额
    total_spent = sum(tx.amount for tx in transactions)
    remaining = budget - total_spent
    usage_rate = (total_spent / budget * 100) if budget > 0 else 0

    # 健康状态判断
    if usage_rate < 60:
        status = "健康" if i18n.locale == "zh_CN" else "Healthy"
        status_color = "🟢"
    elif usage_rate < 85:
        status = "良好" if i18n.locale == "zh_CN" else "Good"
        status_color = "🟡"
    elif usage_rate < 100:
        status = "警告" if i18n.locale == "zh_CN" else "Warning"
        status_color = "🟠"
    else:
        status = "超支" if i18n.locale == "zh_CN" else "Overspent"
        status_color = "🔴"

    # 渲染卡片
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1;">
                    <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.3rem;">
                        {"财务健康状况" if i18n.locale == "zh_CN" else "Financial Health"}
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700;">
                        {status_color} {status}
                    </div>
                </div>
                <div style="flex: 2; display: flex; gap: 2rem; justify-content: flex-end;">
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; opacity: 0.85;">
                            {"月度预算" if i18n.locale == "zh_CN" else "Monthly Budget"}
                        </div>
                        <div style="font-size: 1.3rem; font-weight: 600;">
                            ¥{budget:,.0f}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; opacity: 0.85;">
                            {"已支出" if i18n.locale == "zh_CN" else "Spent"}
                        </div>
                        <div style="font-size: 1.3rem; font-weight: 600;">
                            ¥{total_spent:,.0f}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; opacity: 0.85;">
                            {"剩余" if i18n.locale == "zh_CN" else "Remaining"}
                        </div>
                        <div style="font-size: 1.3rem; font-weight: 600;">
                            ¥{remaining:,.0f}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; opacity: 0.85;">
                            {"使用率" if i18n.locale == "zh_CN" else "Usage"}
                        </div>
                        <div style="font-size: 1.3rem; font-weight: 600;">
                            {usage_rate:.1f}%
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
