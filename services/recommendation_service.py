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
        # 详细报告使用GPT-4o完整模型（更强大，适合长文本生成）
        self.report_model = "gpt-4o"
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

    @safe_call(timeout=30, fallback=None, error_message="个性化问题生成失败")
    def generate_personalized_questions(
        self,
        transactions: Iterable[Transaction],
        budget: float,
        locale: str = "zh_CN",
    ) -> List[Dict[str, object]] | None:
        """
        基于用户真实消费数据动态生成3-5个个性化风险评估问题

        Args:
            transactions: 交易记录
            budget: 月度预算
            locale: 语言区域

        Returns:
            问题列表，格式兼容原RISK_QUESTIONS，失败返回None
        """
        try:
            client = self._ensure_client()
        except RuntimeError as e:
            logger.warning(f"LLM client初始化失败: {e}")
            return None

        # 分析用户消费数据
        metrics = self.analyze_transactions(transactions)
        monthly_avg = float(metrics.get("monthly_average", 0.0) or 0.0)
        volatility = float(metrics.get("spending_volatility", 0.0) or 0.0)
        investable = float(metrics.get("investable_amount", 0.0) or 0.0)
        breakdown = metrics.get("category_breakdown", {}) or {}

        # 预算使用情况
        budget_usage_rate = (monthly_avg / budget * 100) if budget > 0 else 0

        # 消费特征分析
        top_category = next(iter(breakdown.keys())) if breakdown else "其他"
        top_category_share = next(iter(breakdown.values())) if breakdown else 0

        txn_list = list(transactions)
        total_amount = sum(t.amount for t in txn_list)

        if locale == "en_US":
            system_prompt = """You are a professional financial advisor. Generate 3-5 personalized risk assessment questions based on user's real spending data."""

            user_prompt = f"""Generate 3-5 personalized risk tolerance questions based on user's financial data:

User Financial Profile:
- Monthly spending: ¥{monthly_avg:,.2f}
- Monthly budget: ¥{budget:,.2f}
- Budget usage: {budget_usage_rate:.1f}%
- Spending volatility: {volatility:.2%}
- Investable amount: ¥{investable:,.2f}/month
- Top spending category: {top_category} ({top_category_share*100:.1f}%)
- Total transactions: {len(txn_list)} records

Question Generation Rules:
1. If volatility > 30%: Ask about income stability
2. If investable < 500: Ask about emergency fund
3. If budget_usage > 80%: Ask about debt situation
4. If top_category_share > 40%: Ask about category-specific plans
5. Always include: loss tolerance, investment horizon, volatility attitude

Return JSON format:
{{
  "questions": [
    {{
      "id": "custom_q1",
      "question": "Your question here",
      "options": [
        {{"label": "Option 1", "score": 1}},
        {{"label": "Option 2", "score": 2}},
        {{"label": "Option 3", "score": 3}}
      ]
    }}
  ]
}}

Requirements:
- 3-5 questions total
- Each question must have 3 options with scores 1-3
- Questions must be specific to user's situation
- Natural, conversational tone
"""
        else:  # zh_CN
            system_prompt = """你是专业的理财顾问，基于用户的真实消费数据生成个性化风险评估问题。"""

            user_prompt = f"""基于用户的真实财务数据，生成3-5个个性化的风险承受能力评估问题：

用户财务画像：
- 月均消费：¥{monthly_avg:,.2f}
- 月度预算：¥{budget:,.2f}
- 预算使用率：{budget_usage_rate:.1f}%
- 消费波动率：{volatility:.2%}
- 可投资金额：¥{investable:,.2f}/月
- 最大支出类目：{top_category}（占比{top_category_share*100:.1f}%）
- 交易记录：{len(txn_list)}笔

问题生成规则：
1. 如果消费波动率>30%：询问收入稳定性
2. 如果可投资金额<500元：询问是否有紧急备用金
3. 如果预算使用率>80%：询问债务情况
4. 如果某类目占比>40%：询问该类目的特殊计划
5. 必须包含：亏损承受能力、投资期限、波动态度

返回JSON格式：
{{
  "questions": [
    {{
      "id": "custom_q1",
      "question": "您的问题",
      "options": [
        {{"label": "选项1", "score": 1}},
        {{"label": "选项2", "score": 2}},
        {{"label": "选项3", "score": 3}}
      ]
    }}
  ]
}}

要求：
- 总共3-5个问题
- 每个问题必须有3个选项，分数1-3
- 问题必须贴合用户实际情况
- 语言自然、口语化
"""

        try:
            logger.info("开始生成个性化风险问题")
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=30,
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM返回空内容")
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
            questions = data.get("questions", [])

            if not questions or len(questions) < 3:
                logger.warning(f"生成的问题数量不足: {len(questions)}")
                return None

            # 验证格式
            for q in questions:
                if not all(key in q for key in ["id", "question", "options"]):
                    logger.warning("问题格式不完整")
                    return None
                if len(q["options"]) != 3:
                    logger.warning(f"选项数量不是3个: {len(q['options'])}")
                    return None

            logger.info(f"成功生成{len(questions)}个个性化问题")
            return questions

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"个性化问题生成失败: {e}")
            return None

    @safe_call(timeout=60, fallback="", error_message="详细报告生成失败")
    def generate_detailed_report(
        self,
        transactions: Iterable[Transaction],
        responses: Dict[str, int],
        investment_goal: str,
        risk_profile: str,
        metrics: Dict[str, float | Dict[str, float]],
        locale: str = "zh_CN",
    ) -> str:
        """
        生成详细的理财报告（Markdown格式，使用GPT-4o完整模型）

        Args:
            transactions: 交易记录
            responses: 风险问卷回答
            investment_goal: 投资目标
            risk_profile: 风险等级（conservative/balanced/aggressive）
            metrics: 财务画像指标
            locale: 语言区域

        Returns:
            Markdown格式的详细理财报告
        """
        try:
            client = self._ensure_client()
        except RuntimeError as e:
            logger.warning(f"LLM client初始化失败: {e}")
            return ""

        # 准备数据
        monthly_avg = float(metrics.get("monthly_average", 0.0) or 0.0)
        volatility = float(metrics.get("spending_volatility", 0.0) or 0.0)
        investable = float(metrics.get("investable_amount", 0.0) or 0.0)
        breakdown = metrics.get("category_breakdown", {}) or {}
        allocation = self.generate_allocation(risk_profile)

        # 交易数据统计
        txn_list = list(transactions)
        total_amount = sum(t.amount for t in txn_list)
        categories = {}
        for txn in txn_list:
            cat = txn.category or "其他"
            categories[cat] = categories.get(cat, 0) + txn.amount

        # 消费类别详情
        category_details = "\n".join(
            f"  - **{cat}**: ¥{amount:,.2f} ({amount/total_amount*100:.1f}%)"
            for cat, amount in sorted(categories.items(), key=lambda x: x[1], reverse=True)
        )

        # 完整交易明细（供LLM深入分析）
        transaction_details = "\n".join(
            f"  - {t.date} | {t.merchant or '未知商户'} | {t.category} | ¥{t.amount:.2f}"
            for t in sorted(txn_list, key=lambda x: x.date)
        )

        # 风险映射
        risk_map = {
            "conservative": "保守型",
            "balanced": "平衡型",
            "aggressive": "进取型",
        }
        risk_name_cn = risk_map.get(risk_profile, "平衡型")

        # 资产配置详情
        allocation_details = "\n".join(
            f"  - **{asset}**: {percentage*100:.0f}%"
            for asset, percentage in allocation.items()
        )

        # 构建详细的Prompt
        if locale == "en_US":
            system_prompt = """You are a senior financial advisor with 15+ years of experience in wealth management. Generate a comprehensive, professional financial advisory report based on real transaction data."""

            user_prompt = f"""Generate a detailed financial advisory report in Markdown format.

## User Financial Profile
- Monthly Spending: ¥{monthly_avg:,.2f}
- Spending Volatility: {volatility:.2%}
- Investable Amount: ¥{investable:,.2f}/month
- Risk Tolerance: {risk_profile} ({risk_name_cn})
- Investment Goal: {investment_goal or 'Not specified'}
- Total Transactions: {len(txn_list)} records, ¥{total_amount:,.2f}

## Spending Breakdown
{category_details}

## Complete Transaction History (for deep analysis)
{transaction_details}

## Recommended Asset Allocation
{allocation_details}

## Report Requirements
Generate a comprehensive report (4000-6000 words) with the following structure:

**Critical Requirements**:
- Must analyze based on the real transaction history above, not generic advice
- All data must be specific with amounts, percentages, timeframes
- Analyze spending trends, anomalies, optimization opportunities
- Provide actionable recommendations with real product names, codes, platforms

### 1. Executive Summary (400-500 words)
- Key findings and recommendations (insights from real transaction data)
- Financial health score (0-100, explain scoring criteria)
- Top 3 priority actions (each with specific amounts and timeline)

### 2. Financial Situation Deep Analysis (1200-1500 words)
- **Income & Spending Patterns**: Analyze monthly/weekly consumption patterns from transaction history, identify high-frequency spending periods
- **Spending Stability Assessment**: Based on actual volatility and transaction frequency
- **Category Breakdown Insights**:
  * Analyze specific merchants and amounts in each category
  * Identify non-essential spending that can be optimized (specific merchants)
  * Provide savings suggestions (specific amounts and alternatives)
- **Spending Trend Analysis**: Analyze expense changes from time series perspective
- **Peer/Income Level Comparison**: Provide specific comparison data and percentile rankings
- **Strengths & Weaknesses**: Each point must be supported by specific data

### 3. Risk Assessment (800-1000 words)
- Comprehensive risk tolerance evaluation (considering age, income, goal timeline)
- Risk questionnaire deep analysis (if available)
- Real behavior vs stated preferences comparison (based on actual spending behavior)
- Investment horizon suitability analysis (combining goals and risks)
- Risk capacity assessment (maximum acceptable loss amount)

### 4. Asset Allocation Strategy (1500-2000 words)
- **Recommended Allocation Rationale**: Explain why this allocation ratio
- **Expected Returns & Risk Calculations**:
  * Provide 3-5 year expected return curves
  * Best/worst/average scenario analysis
  * Cumulative effect of specific monthly investment amounts
- **Rebalancing Strategy**: When to adjust, adjustment magnitude
- **Tax Efficiency Considerations**: Specific tax avoidance strategies
- **Specific Product Recommendations**:
  * At least 3-5 funds/ETFs with complete names and codes
  * Risk level, historical returns, fees for each product
  * Purchase channels and platform recommendations

### 5. Execution Plan (1000-1200 words)
- **Step-by-Step Implementation Guide**:
  * Month 1-3: Account opening, initial allocation, auto-invest setup (specific steps)
  * Month 3-6: Monitoring and adjustment (specific indicators and thresholds)
  * Month 6-12: Optimization and upgrade (specific improvement directions)
- **Account Opening & Platform Selection**: Recommend 3-5 platforms, compare fees and features
- **Automation Setup**: Detailed auto-invest, take-profit and stop-loss settings
- **Performance Monitoring Plan**: Specific KPI indicators and monitoring frequency

### 6. Risk Warnings & Disclaimers (600-800 words)
- Market risks (specific potential loss magnitude)
- Liquidity risks (redemption restrictions and costs)
- Regulatory policy risks
- Important disclaimers
- When to adjust the plan (specific trigger conditions)

Format Requirements:
- Markdown format with clear heading hierarchy
- Use tables for complex data (product comparisons, return calculations)
- Bold important data and conclusions
- Avoid generic advice - all recommendations must have specific numbers and timelines
- Total report should be 4000-6000 words"""

        else:  # zh_CN
            system_prompt = """你是一位拥有15年财富管理经验的资深理财顾问，专注于为中国用户提供基于真实数据的个性化理财建议。你的报告专业、详细、可操作性强。"""

            user_prompt = f"""基于以下真实财务数据，生成一份详细的理财咨询报告（Markdown格式）。

## 用户财务画像
- 月均消费：¥{monthly_avg:,.2f}
- 消费波动率：{volatility:.2%}
- 可投资金额：¥{investable:,.2f}/月
- 风险偏好：{risk_profile} ({risk_name_cn})
- 投资目标：{investment_goal or '未明确指定'}
- 交易记录：{len(txn_list)}笔，累计¥{total_amount:,.2f}

## 消费结构详情
{category_details}

## 完整交易明细（供深度分析使用）
{transaction_details}

## 推荐资产配置
{allocation_details}

## 报告要求
生成一份详细的理财咨询报告（4000-6000字），包含以下结构：

**关键要求**:
- 必须基于上述真实交易明细进行深度分析，不要泛泛而谈
- 所有数据必须具体到金额、百分比、时间段
- 分析消费趋势、异常交易、优化机会
- 提供可操作的具体建议，包含真实产品名称、代码、平台

### 1. 报告摘要（400-500字）
- 核心发现与建议（基于真实交易数据的洞察）
- 财务健康评分（0-100分，详细说明评分依据）
- 三大优先行动建议（每条包含具体金额和时间表）

### 2. 财务状况深度分析（1200-1500字）
- **收支模式分析**: 从交易明细中分析每月/每周消费规律，识别高频消费时间段
- **消费稳定性评估**: 基于实际波动率和交易频次分析
- **类目结构洞察**:
  * 分析每个类别的具体商户和金额
  * 识别可优化的非必要支出（具体到商户名）
  * 提出节约建议（具体金额和替代方案）
- **消费趋势分析**: 从时间序列角度分析支出变化
- **同龄人/同收入层对比**: 提供具体的对比数据和百分位排名
- **财务优势与风险点**: 每项必须有具体数据支撑

### 3. 风险评估（800-1000字）
- 风险承受能力综合评估（结合年龄、收入、目标期限）
- 问卷答案深度分析（如有）
- 实际行为 vs 主观意愿对比（基于实际消费行为）
- 投资期限适配性分析（结合目标和风险）
- 风险容量评估（可承受的最大亏损金额）

### 4. 资产配置策略（1500-2000字）
- **推荐配置方案详解**: 解释为何选择该配置比例
- **预期收益与风险测算**:
  * 提供3-5年的预期回报曲线
  * 最好/最坏/平均场景分析
  * 具体到每月投资金额的累积效果
- **再平衡策略**: 何时调整，调整幅度
- **税务优化考虑**: 具体的避税策略
- **具体产品推荐**:
  * 至少3-5只基金/ETF，包含完整名称和代码
  * 每只产品的风险等级、历史收益、费率
  * 购买渠道和平台推荐

### 5. 执行计划（1000-1200字）
- **分步实施指南**:
  * 第1-3月: 开户、首次配置、定投设置（具体操作步骤）
  * 第3-6月: 监控调整（具体指标和阈值）
  * 第6-12月: 优化升级（具体改进方向）
- **开户与平台选择**: 推荐3-5个平台，比较费率和功能
- **自动化设置**: 详细的定投、止盈止损设置
- **业绩监控方案**: 具体的KPI指标和监控频率

### 6. 风险提示与免责声明（600-800字）
- 市场风险（具体到可能的亏损幅度）
- 流动性风险（赎回限制和成本）
- 监管政策风险
- 重要免责声明
- 何时需要调整方案（具体的触发条件）

格式要求：
- Markdown格式，标题层级清晰
- 使用表格展示复杂数据（如产品对比、收益测算）
- 加粗重要数据和结论
- 避免空洞的通用建议，所有建议必须有具体数字和时间表
- 报告总字数应在4000-6000字之间"""

        try:
            logger.info(f"开始生成详细报告，使用模型: {self.report_model}")
            response = client.chat.completions.create(
                model=self.report_model,
                temperature=0.7,  # 稍高温度允许更自然的写作风格
                max_tokens=12000,  # 支持4000-6000字的详细报告
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=90,  # 长文本生成需要更多时间
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM返回空内容")
                return ""

            logger.info(f"详细报告生成成功，长度: {len(content)} 字符")
            return content.strip()

        except Exception as e:
            logger.error(f"详细报告生成失败: {e}")
            return ""
