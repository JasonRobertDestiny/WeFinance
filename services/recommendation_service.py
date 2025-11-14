"""Service layer for generating explainable financial recommendations."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from models.entities import Recommendation, Transaction
from utils.error_handling import safe_call
from utils.i18n import I18n

load_dotenv()
logger = logging.getLogger(__name__)


class RecommendationService:
    """Generate risk-aware allocation plans with explainable rationale."""

    # 保留作为fallback规则（LLM失败时使用）
    ALLOCATION_RULES: Dict[str, Dict[str, float]] = {
        "conservative": {"债券基金": 0.7, "混合理财": 0.3},
        "balanced": {"债券基金": 0.5, "股票基金": 0.3, "货币基金": 0.2},
        "aggressive": {"股票基金": 0.6, "成长基金": 0.3, "货币基金": 0.1},
    }

    EXPECTED_RETURN = {"conservative": 4.5, "balanced": 6.8, "aggressive": 9.5}
    MAX_DRAWDOWN = {"conservative": 5.0, "balanced": 12.0, "aggressive": 20.0}

    def __init__(self):
        """Initialize with OpenAI client for LLM-powered recommendations."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("OPENAI_API_KEY not configured")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @safe_call(timeout=30, fallback=None, error_message="LLM风险评估失败")
    def _conduct_risk_assessment_llm(
        self,
        responses: Dict[str, int],
        user_profile: Dict[str, float],
        locale: str = "zh_CN",
    ) -> Tuple[str, Dict[str, float], List[str]] | None:
        """
        使用LLM进行综合风险评估和资产配置

        Args:
            responses: 风险问卷回答
            user_profile: 用户财务画像（monthly_avg, volatility, investable等）
            locale: 语言区域

        Returns:
            (risk_profile, allocation, reasoning) 或 None（触发fallback）
        """
        try:
            client = self._ensure_client()
        except RuntimeError as e:
            logger.warning(f"LLM client初始化失败: {e}")
            return None

        score = sum(responses.values())
        monthly_avg = float(user_profile.get("monthly_avg", 0))
        volatility = float(user_profile.get("volatility", 0))
        investable = float(user_profile.get("investable", 0))

        if locale == "en_US":
            prompt = f"""You are a professional financial advisor. Assess user's true risk tolerance comprehensively.

Risk Questionnaire:
{json.dumps(responses, ensure_ascii=False, indent=2)}
Total Score: {score}

User Financial Profile:
- Monthly spending: ¥{monthly_avg:.2f}
- Spending volatility: {volatility:.2%} (higher = less stable)
- Investable amount: ¥{investable:.2f}/month

Comprehensive Assessment Rules:
1. Don't rely solely on questionnaire score - real data is more important
2. If volatility > 30%, actual risk tolerance is lower than questionnaire suggests (income unstable)
3. If investable < 500元/month, not suitable for aggressive strategy
4. If monthly spending < 3000元, might be student/low-income, be cautious

Analyze and return JSON:
{{
  "risk_profile": "conservative/balanced/aggressive",
  "allocation": {{
    "Bond Funds": 0.5,
    "Stock Funds": 0.3,
    "Money Market": 0.2
  }},
  "reasoning": [
    "Step 1: Based on XX analysis...",
    "Step 2: Considering XX factors...",
    "Step 3: Overall recommendation..."
  ]
}}

Requirements:
- allocation must sum to 1.0
- each asset ratio between 0-1
- reasoning must show causality
"""
        else:
            prompt = f"""你是专业的财务顾问，需要综合评估用户的风险承受能力。

风险测评问卷回答:
{json.dumps(responses, ensure_ascii=False, indent=2)}
问卷总分: {score}

用户真实财务画像:
- 月均消费: ¥{monthly_avg:.2f}
- 消费波动率: {volatility:.2%}（越高说明收入/支出越不稳定）
- 可投资金额: ¥{investable:.2f}/月

综合评估规则:
1. 不要只看问卷分数，消费数据更重要
2. 如果消费波动率>30%，实际风险承受能力低于问卷显示（收入不稳定）
3. 如果可投资金额<500元/月，不适合激进策略
4. 如果月均消费<3000元，可能是学生/低收入群体，需谨慎

请分析并返回JSON:
{{
  "risk_profile": "conservative/balanced/aggressive",
  "allocation": {{
    "债券基金": 0.5,
    "股票基金": 0.3,
    "货币基金": 0.2
  }},
  "reasoning": [
    "分析步骤1: 基于XX判断...",
    "分析步骤2: 考虑到XX因素...",
    "分析步骤3: 综合建议..."
  ]
}}

要求:
- allocation总和必须等于1.0
- 每个资产类别占比0-1之间
- reasoning必须体现因果关系
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.0,  # 风险评估需要稳定输出
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional risk assessment advisor with comprehensive financial analysis."
                            if locale == "en_US"
                            else "你是专业的风险评估顾问，综合分析用户财务状况。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            content = response.choices[0].message.content
            if not content:
                return None

            # 清理markdown代码块
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            risk_profile = data.get("risk_profile", "")
            allocation = data.get("allocation", {})
            reasoning = data.get("reasoning", [])

            # 验证
            if risk_profile not in ["conservative", "balanced", "aggressive"]:
                logger.warning(f"无效的risk_profile: {risk_profile}")
                return None

            if not allocation:
                logger.warning("allocation为空")
                return None

            total = sum(allocation.values())
            if not (0.99 <= total <= 1.01):  # 允许浮点误差
                logger.warning(f"allocation总和={total}, 不等于1.0")
                # 归一化
                allocation = {k: v / total for k, v in allocation.items()}

            logger.info(f"LLM风险评估成功: {risk_profile}, 配置: {allocation}")
            return risk_profile, allocation, reasoning

        except json.JSONDecodeError as e:
            logger.warning(f"LLM响应JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLM风险评估异常: {e}")
            return None

    def conduct_risk_assessment(
        self, responses: Dict[str, int], user_profile: Dict[str, float] | None = None
    ) -> str:
        """Map questionnaire responses to a risk profile key (增强版：优先LLM，fallback到规则)."""

        # 尝试LLM评估
        if user_profile:
            llm_result = self._conduct_risk_assessment_llm(responses, user_profile)
            if llm_result:
                risk_profile, allocation, reasoning = llm_result
                # 存储LLM推荐的资产配置和推理过程
                self._llm_allocation = allocation
                self._llm_reasoning = reasoning
                logger.info(f"使用LLM风险评估结果: {risk_profile}")
                return risk_profile

        # Fallback到硬编码规则
        logger.info("LLM风险评估失败，使用fallback规则")
        score = sum(responses.values())
        if score <= 4:
            return "conservative"
        if score <= 7:
            return "balanced"
        return "aggressive"

    def generate_allocation(self, risk_profile: str) -> Dict[str, float]:
        """Return asset allocation percentages based on risk appetite (增强版：优先LLM推荐，fallback到固定规则)."""

        # 如果有LLM推荐的配置，使用它
        if hasattr(self, "_llm_allocation") and self._llm_allocation:
            logger.info("使用LLM推荐的资产配置")
            return self._llm_allocation

        # Fallback到固定规则
        logger.info("使用fallback资产配置规则")
        return self.ALLOCATION_RULES.get(
            risk_profile, self.ALLOCATION_RULES["balanced"]
        )

    @staticmethod
    def _parse_goal(goal_text: str) -> Tuple[str, float | None, int | None]:
        """
        Extract core goal, target amount (RMB), and horizon (months) from freeform text.
        """
        normalized = goal_text.strip()
        if not normalized:
            return "未指定", None, None

        amount_match = re.search(r"(\d+(?:\.\d+)?)\s*(万|千|元|块)?", normalized)
        amount_value = None
        if amount_match:
            value = float(amount_match.group(1))
            unit = amount_match.group(2) or ""
            multiplier = 1.0
            if unit in {"万", "万元"}:
                multiplier = 10_000.0
            elif unit in {"千", "千元"}:
                multiplier = 1_000.0
            amount_value = value * multiplier

        horizon_match = re.search(r"(\d+)\s*(年|个月|月)", normalized)
        horizon_months = None
        if horizon_match:
            value = int(horizon_match.group(1))
            unit = horizon_match.group(2)
            horizon_months = value * 12 if unit == "年" else value

        return normalized, amount_value, horizon_months

    def _estimate_metrics(self, risk_profile: str) -> Dict[str, float]:
        return {
            "expected_return": self.EXPECTED_RETURN[risk_profile],
            "max_drawdown": self.MAX_DRAWDOWN[risk_profile],
        }

    def _transactions_dataframe(
        self, transactions: Iterable[Transaction]
    ) -> pd.DataFrame:
        rows = [
            {
                "date": pd.to_datetime(txn.date),
                "amount": float(txn.amount),
                "category": txn.category or "其他",
            }
            for txn in transactions
        ]
        if not rows:
            return pd.DataFrame(columns=["date", "amount", "category"])
        df = pd.DataFrame(rows)
        df.sort_values("date", inplace=True)
        return df

    @staticmethod
    def _monthly_average(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        monthly = df.copy()
        monthly["year_month"] = monthly["date"].dt.to_period("M")
        grouped = monthly.groupby("year_month")["amount"].sum()
        return float(grouped.mean()) if not grouped.empty else 0.0

    @staticmethod
    def _spending_volatility(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        monthly = df.copy()
        monthly["year_month"] = monthly["date"].dt.to_period("M")
        grouped = monthly.groupby("year_month")["amount"].sum()
        if len(grouped) < 2:
            return 0.0
        mean = grouped.mean()
        if mean == 0:
            return 0.0
        return float(grouped.std(ddof=0) / mean)

    @staticmethod
    def _category_breakdown(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
        totals = df.groupby("category")["amount"].sum()
        full = totals.sum()
        if full == 0:
            return {}
        shares = {cat: float(value / full) for cat, value in totals.items()}
        return dict(sorted(shares.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def _estimate_investable(monthly_avg: float) -> float:
        if monthly_avg <= 0:
            return 0.0
        if monthly_avg < 3_000:
            ratio = 0.1
        elif monthly_avg < 10_000:
            ratio = 0.2
        else:
            ratio = 0.3
        return round(monthly_avg * ratio, 2)

    def _format_allocation_desc(
        self, allocation: Dict[str, float], i18n: I18n
    ) -> Tuple[str, str]:
        allocation_desc = ", ".join(
            f"{i18n.t('recommendation.assets.' + asset)} {percentage*100:.0f}%"
            for asset, percentage in allocation.items()
        )
        allocation_rationale = "\n".join(
            f"- {i18n.t('recommendation.assets.' + asset)}: {percentage*100:.0f}%"
            for asset, percentage in allocation.items()
        )
        return allocation_desc, allocation_rationale

    def analyze_transactions(
        self, transactions: Iterable[Transaction]
    ) -> Dict[str, float | Dict[str, float]]:
        df = self._transactions_dataframe(transactions)
        monthly_avg = self._monthly_average(df)
        volatility = self._spending_volatility(df)
        breakdown = self._category_breakdown(df)
        investable = self._estimate_investable(monthly_avg)
        return {
            "monthly_average": monthly_avg,
            "spending_volatility": volatility,
            "category_breakdown": breakdown,
            "investable_amount": investable,
        }

    def _generate_llm_recommendations(
        self,
        metrics: Dict[str, float | Dict[str, float]],
        risk_profile: str,
        investment_goal: str,
        locale: str,
    ) -> List[Recommendation] | None:
        """
        使用LLM生成个性化投资推荐（基于真实消费数据）

        Returns None if LLM call fails, allowing fallback to rule-based recommendations
        """
        try:
            client = self._ensure_client()
        except RuntimeError as e:
            logger.warning(f"LLM client初始化失败: {e}")
            return None

        # 准备详细的用户财务画像
        monthly_avg = float(metrics.get("monthly_average", 0.0) or 0.0)
        volatility = float(metrics.get("spending_volatility", 0.0) or 0.0)
        investable = float(metrics.get("investable_amount", 0.0) or 0.0)
        breakdown = metrics.get("category_breakdown", {}) or {}

        # 构建详细的prompt
        risk_map = {
            "conservative": "保守型（不愿承受波动，追求稳健收益）",
            "balanced": "平衡型（可接受适度波动，追求收益与风险平衡）",
            "aggressive": "进取型（可承受较大波动，追求高收益）",
        }

        breakdown_str = (
            "\n".join(
                f"  - {cat}: ¥{amt:.2f} ({amt/monthly_avg*100:.1f}%)"
                for cat, amt in list(breakdown.items())[:5]
            )
            if breakdown and monthly_avg > 0
            else "  （暂无数据）"
        )

        system_prompt = f"""你是一位专业的理财顾问，根据用户的真实消费数据提供个性化投资建议。

用户财务画像：
- 月均消费：¥{monthly_avg:.2f}
- 消费波动率：{volatility:.2%}（越高说明消费越不稳定）
- 可投资金额：¥{investable:.2f}/月
- 风险偏好：{risk_map.get(risk_profile, risk_profile)}
- 投资目标：{investment_goal or '未指定具体目标'}

消费结构（Top 5类目）：
{breakdown_str}

请基于以上数据，生成2-3条具体的理财建议。每条建议需要包含：
1. 标题（简短有力，10字以内）
2. 摘要（一句话说明这条建议的核心内容，30字左右）
3. 推理步骤（2-4步，每步解释为什么这样建议，展示可解释性）
4. 风险等级（保守型/平衡型/进取型）

返回JSON格式：
{{
  "recommendations": [
    {{
      "title": "建议标题",
      "summary": "核心内容摘要",
      "rationale_steps": ["步骤1", "步骤2", "步骤3"],
      "risk_level": "风险等级"
    }}
  ]
}}

要求：
- 必须基于用户的真实数据（消费金额、结构、波动率）
- 推理步骤要体现因果关系（"因为你的XX情况，所以建议YY"）
- 避免通用建议，要个性化
- 如果可投资金额很少(<500元/月)，诚实告知并给出实际可行的建议
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,  # 稍高温度允许创造性，但保持合理性
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "请基于我的财务数据，生成个性化理财建议。",
                    },
                ],
                timeout=30,
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM返回空内容")
                return None

            # 解析JSON响应
            # 清理可能的markdown代码块
            content_cleaned = content.strip()
            if content_cleaned.startswith("```json"):
                content_cleaned = content_cleaned[7:]
            if content_cleaned.startswith("```"):
                content_cleaned = content_cleaned[3:]
            if content_cleaned.endswith("```"):
                content_cleaned = content_cleaned[:-3]
            content_cleaned = content_cleaned.strip()

            data = json.loads(content_cleaned)
            recs_data = data.get("recommendations", [])

            if not recs_data:
                logger.warning("LLM返回的JSON中没有recommendations")
                return None

            # 转换为Recommendation对象
            recommendations = []
            for rec in recs_data:
                recommendations.append(
                    Recommendation(
                        title=rec.get("title", ""),
                        summary=rec.get("summary", ""),
                        rationale_steps=rec.get("rationale_steps", []),
                        risk_level=rec.get(
                            "risk_level", risk_map.get(risk_profile, "平衡型")
                        ),
                    )
                )

            logger.info(f"LLM生成了{len(recommendations)}条个性化推荐")
            return recommendations

        except json.JSONDecodeError as e:
            logger.warning(
                f"LLM响应JSON解析失败: {e}, content={content[:200] if content else 'None'}"
            )
            return None
        except Exception as e:
            logger.warning(f"LLM推荐生成失败: {e}")
            return None

    def generate_recommendations(
        self,
        transactions: Iterable[Transaction],
        *,
        risk_profile: str = "balanced",
        investment_goal: str = "",
        locale: str = "zh_CN",
        metrics: Dict[str, float | Dict[str, float]] | None = None,
    ) -> List[Recommendation]:
        """Generate actionable recommendations grounded in real spending data."""

        metrics = metrics or self.analyze_transactions(transactions)
        monthly_avg = float(metrics.get("monthly_average", 0.0) or 0.0)
        volatility = float(metrics.get("spending_volatility", 0.0) or 0.0)
        investable = float(metrics.get("investable_amount", 0.0) or 0.0)
        breakdown = metrics.get("category_breakdown", {}) or {}

        i18n = I18n(locale)
        allocation = self.generate_allocation(risk_profile)
        allocation_desc, _ = self._format_allocation_desc(allocation, i18n)

        goal_name, _, _ = self._parse_goal(investment_goal)
        goal_text = goal_name or i18n.t("recommendation.goal_default")
        risk_name = i18n.t(f"recommendation.risk_name.{risk_profile}")
        invest_pct = (investable / monthly_avg * 100) if monthly_avg else 0.0
        buffer_low = monthly_avg * 3
        buffer_high = monthly_avg * 6

        recs: List[Recommendation] = []
        primary_summary = i18n.t(
            "recommendation.primary_summary",
            monthly=monthly_avg,
            investable=investable,
            invest_pct=invest_pct,
            allocation=allocation_desc,
            goal=goal_text,
        )
        primary_steps = [
            i18n.t(
                "recommendation.primary_step_spend",
                monthly=monthly_avg,
                volatility=volatility,
            ),
            i18n.t(
                "recommendation.primary_step_buffer",
                buffer_low=buffer_low,
                buffer_high=buffer_high,
            ),
            i18n.t(
                "recommendation.primary_step_invest",
                invest_pct=invest_pct,
                allocation=allocation_desc,
                risk=risk_name,
            ),
        ]

        recs.append(
            Recommendation(
                title=i18n.t("recommendation.primary_title"),
                summary=primary_summary,
                rationale_steps=primary_steps,
                risk_level=risk_name,
            )
        )

        if breakdown:
            top_category, share = next(iter(breakdown.items()))
            recs.append(
                Recommendation(
                    title=i18n.t(
                        "recommendation.category_tip_title", category=top_category
                    ),
                    summary=i18n.t(
                        "recommendation.category_tip_summary",
                        category=top_category,
                        share=share * 100,
                    ),
                    rationale_steps=[
                        i18n.t(
                            "recommendation.category_tip_step",
                            category=top_category,
                            investable=investable,
                        )
                    ],
                    risk_level=risk_name,
                )
            )

        return recs

    def create_plan(
        self,
        responses: Dict[str, int],
        investment_goal: str,
        transactions: Iterable[Transaction],
        locale: str,
    ) -> Tuple[List[Recommendation], Dict[str, float | Dict[str, float]], str]:
        """High-level orchestrator returning recommendations and derived metrics."""

        metrics = self.analyze_transactions(transactions)
        # 将metrics作为user_profile传递给LLM风险评估
        risk_key = self.conduct_risk_assessment(responses, user_profile=metrics)
        risk_name = I18n(locale).t(f"recommendation.risk_name.{risk_key}")
        recs = self.generate_recommendations(
            transactions,
            risk_profile=risk_key,
            investment_goal=investment_goal,
            locale=locale,
            metrics=metrics,
        )
        return recs, metrics, risk_name

    def generate(
        self,
        transactions: Iterable[Transaction],
        responses: Dict[str, int],
        investment_goal: str,
        *,
        locale: str = "zh_CN",
    ) -> Dict[str, object]:
        """Public API returning recommendation payload for UI consumption."""

        recommendations, metrics, risk_name = self.create_plan(
            responses=responses,
            investment_goal=investment_goal,
            transactions=transactions,
            locale=locale,
        )
        return {
            "recommendations": recommendations,
            "financial_profile": metrics,
            "risk_level": risk_name,
            "locale": locale,
        }
