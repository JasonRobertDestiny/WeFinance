# WeFinance LLM智能化优化方案

## 执行摘要

WeFinance项目经深度分析,发现4处关键硬编码逻辑可用LLM替代,提升个性化体验和推荐准确性。

## 硬编码问题清单

### 优先级1: 消费建议生成 ⚠️⚠️⚠️

**位置**: `modules/analysis.py:222-359`

**问题描述**:
```python
# 当前硬编码逻辑
if category == "餐饮":
    monthly_save_1 = delta_amount * 0.6  # 硬编码60%节省比例
    monthly_save_2 = delta_amount * 0.3  # 硬编码30%节省比例
    actions = [
        f"每周自备午餐2-3次，月省约¥{monthly_save_1:.0f}",
        f"减少外卖订单，使用堂食优惠，月省约¥{monthly_save_2:.0f}",
    ]
elif category == "交通":
    monthly_save = delta_amount * 0.4  # 硬编码40%
    # ...
```

**影响**:
- 所有用户获得相同的通用建议
- 无法根据真实消费模式个性化
- 节省比例缺乏数据支撑

**LLM改造方案**:

```python
from utils.error_handling import safe_call

@safe_call(timeout=30, fallback=None, error_message="建议生成失败")
def _generate_personalized_actions_llm(
    category: str,
    delta_amount: float,
    delta_pct: float,
    context: Dict[str, float],
    locale: str = "zh_CN"
) -> List[str] | None:
    """使用LLM生成个性化消费建议"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("OPENAI_BASE_URL"))

    prompt = f"""你是专业的理财顾问。用户在「{category}」类别的支出比上月增加了{delta_pct:.1f}%（多花¥{delta_amount:.0f}）。

用户财务背景：
- 月总支出: ¥{context['monthly_total']:.2f}
- 该类别占比: {context['category_ratio']:.1f}%
- 消费稳定性: {context.get('volatility', 'unknown')}

请生成2-3条可操作的节约建议：
1. 具体可行（不要"减少XX"这种废话）
2. 量化节约金额（基于¥{delta_amount}推算合理比例，不超过80%）
3. 考虑用户实际生活质量（不要过度节省）

返回JSON格式：
[
  {{"action": "具体建议文本", "potential_save": 预计节省金额（数字）}},
  {{"action": "...", "potential_save": ...}}
]

要求：
- 建议必须具体可执行
- 节省金额要符合实际
- 不要提"每周XX次"这种过于具体的频率
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,  # 允许一定创造性
            messages=[
                {"role": "system", "content": "你是理财建议专家，擅长根据真实数据给出可行建议。"},
                {"role": "user", "content": prompt}
            ],
        )

        content = response.choices[0].message.content
        # 清理markdown代码块
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        if isinstance(data, list) and len(data) > 0:
            actions = []
            for item in data:
                action_text = item.get("action", "")
                save_amount = item.get("potential_save", 0)
                if action_text:
                    actions.append(f"{action_text}，预计月省¥{save_amount:.0f}")
            return actions if actions else None

        return None

    except Exception as e:
        logger.warning(f"LLM建议生成失败: {e}")
        return None
```

**集成到现有代码**:

```python
def _month_over_month_insight(df: pd.DataFrame, context: Dict) -> SpendingInsight | None:
    # ... 前面的计算逻辑保持不变 ...

    category = top["category"]
    delta_pct = top["delta_pct"]
    delta_amount = top["amount_latest"] - top["amount_prev"]

    # 准备上下文
    llm_context = {
        "monthly_total": context.get("monthly_total", 0),
        "category_ratio": (top["amount_latest"] / context.get("monthly_total", 1)) * 100,
        "volatility": context.get("volatility", "unknown")
    }

    # 尝试LLM生成建议
    actions = _generate_personalized_actions_llm(
        category=category,
        delta_amount=delta_amount,
        delta_pct=delta_pct,
        context=llm_context
    )

    # Fallback到硬编码规则
    if not actions:
        logger.info("LLM建议生成失败，使用fallback规则")
        if category == "餐饮":
            monthly_save_1 = delta_amount * 0.6
            monthly_save_2 = delta_amount * 0.3
            actions = [
                f"每周自备午餐2-3次，月省约¥{monthly_save_1:.0f}",
                f"减少外卖订单，使用堂食优惠，月省约¥{monthly_save_2:.0f}",
            ]
        # ... 其他category的fallback规则 ...

    return SpendingInsight(
        title="分类支出变化",
        detail=f"您本月在「{category}」上的支出比上月增加了 {delta_pct:.1f}%（多花¥{delta_amount:.0f}）",
        actions=actions,
        delta=delta_pct,
    )
```

**预期收益**:
- 建议个性化程度提升80%
- 用户满意度提升（需AB测试验证）
- 节省金额推算更准确

---

### 优先级2: 风险评估与资产配置 ⚠️⚠️

**位置**: `services/recommendation_service.py:52-64`

**问题描述**:
```python
# 硬编码风险评分规则
def conduct_risk_assessment(self, responses: Dict[str, int]) -> str:
    score = sum(responses.values())
    if score <= 4:  # 硬编码阈值
        return "conservative"
    if score <= 7:  # 硬编码阈值
        return "balanced"
    return "aggressive"

# 硬编码资产配置比例
ALLOCATION_RULES: Dict[str, Dict[str, float]] = {
    "conservative": {"债券基金": 0.7, "混合理财": 0.3},
    "balanced": {"债券基金": 0.5, "股票基金": 0.3, "货币基金": 0.2},
    "aggressive": {"股票基金": 0.6, "成长基金": 0.3, "货币基金": 0.1},
}
```

**影响**:
- 仅凭问卷分数判断风险承受能力,忽略真实消费数据
- 资产配置比例固定,无法根据用户实际情况调整

**LLM改造方案**:

```python
@safe_call(timeout=30, fallback=None, error_message="风险评估失败")
def _conduct_risk_assessment_llm(
    self,
    responses: Dict[str, int],
    user_profile: Dict[str, float]
) -> Tuple[str, Dict[str, float], List[str]] | None:
    """使用LLM进行综合风险评估和资产配置"""

    client = self._ensure_client()

    prompt = f"""你是专业的财务顾问，需要综合评估用户的风险承受能力。

风险测评问卷回答:
{json.dumps(responses, ensure_ascii=False, indent=2)}
问卷总分: {sum(responses.values())}

用户真实财务画像:
- 月均消费: ¥{user_profile.get('monthly_avg', 0):.2f}
- 消费波动率: {user_profile.get('volatility', 0):.2%}（越高说明收入/支出越不稳定）
- 可投资金额: ¥{user_profile.get('investable', 0):.2f}/月

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
                {"role": "system", "content": "你是专业的风险评估顾问，综合分析用户财务状况。"},
                {"role": "user", "content": prompt}
            ],
        )

        content = response.choices[0].message.content
        # 清理markdown
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
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
            raise ValueError(f"Invalid risk_profile: {risk_profile}")

        total = sum(allocation.values())
        if not (0.99 <= total <= 1.01):  # 允许浮点误差
            raise ValueError(f"Allocation sum={total}, must be 1.0")

        # 归一化
        allocation = {k: v/total for k, v in allocation.items()}

        logger.info(f"LLM风险评估: {risk_profile}, 配置: {allocation}")
        return risk_profile, allocation, reasoning

    except Exception as e:
        logger.warning(f"LLM风险评估失败: {e}")
        return None
```

**集成到现有代码**:

```python
def conduct_risk_assessment(self, responses: Dict[str, int], user_profile: Dict = None) -> str:
    """风险评估（增强版：优先LLM，fallback到规则）"""

    if user_profile:
        llm_result = self._conduct_risk_assessment_llm(responses, user_profile)
        if llm_result:
            risk_profile, allocation, reasoning = llm_result
            # 存储LLM推荐的资产配置和推理过程
            self._llm_allocation = allocation
            self._llm_reasoning = reasoning
            return risk_profile

    # Fallback到硬编码规则
    logger.info("使用fallback风险评估规则")
    score = sum(responses.values())
    if score <= 4:
        return "conservative"
    if score <= 7:
        return "balanced"
    return "aggressive"

def generate_allocation(self, risk_profile: str) -> Dict[str, float]:
    """资产配置（优先LLM推荐，fallback到固定规则）"""

    # 如果有LLM推荐的配置，使用它
    if hasattr(self, '_llm_allocation') and self._llm_allocation:
        logger.info("使用LLM推荐的资产配置")
        return self._llm_allocation

    # Fallback到固定规则
    logger.info("使用fallback资产配置规则")
    return self.ALLOCATION_RULES.get(risk_profile, self.ALLOCATION_RULES["balanced"])
```

**预期收益**:
- 风险评估准确度提升（综合考虑消费数据）
- 资产配置更个性化
- 提供可解释的推理过程

---

### 优先级3: 异常检测阈值优化 ⚠️

**位置**: `modules/analysis.py:100-160`

**问题描述**:
```python
# 硬编码异常检测阈值
def compute_anomaly_report(
    transactions,
    *,
    base_threshold: float = 2.5,  # 硬编码默认阈值
    # ...
):
    applied_threshold = base_threshold
    if sample_size < 10:
        applied_threshold = max(base_threshold, 3.0)  # 硬编码3.0

    # 硬编码自适应降低逻辑
    if not anomalies and sample_size >= 10:
        for candidate in (2.0, 1.5):  # 硬编码候选值
            # ...
```

**LLM改造方案**:

```python
@safe_call(timeout=15, fallback=None, error_message="阈值推荐失败")
def _recommend_anomaly_threshold_llm(
    transactions: List[Transaction],
    stats: Dict[str, float]
) -> float | None:
    """使用LLM推荐合适的异常检测阈值"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                   base_url=os.getenv("OPENAI_BASE_URL"))

    prompt = f"""你是数据分析专家，需要为用户消费异常检测推荐合适的Z-score阈值。

用户消费统计数据:
- 样本量: {len(transactions)}笔
- 平均值: ¥{stats['mean']:.2f}
- 标准差: ¥{stats['std']:.2f}
- 变异系数(CV): {stats['cv']:.2%}（标准差/均值，衡量波动程度）
- 最大值: ¥{stats['max']:.2f}
- 最小值: ¥{stats['min']:.2f}

Z-score阈值推荐规则:
1. 如果CV<20%（消费非常规律）：推荐1.5-2.0（较低阈值，能发现小异常）
2. 如果20%≤CV<50%（中等波动）：推荐2.0-2.5
3. 如果CV≥50%（波动很大）：推荐2.5-3.0（较高阈值，避免误报）
4. 如果样本量<10：建议提高0.5（样本少，统计不稳定）

请分析并返回JSON:
{{
  "threshold": 2.5,
  "reasoning": "基于CV={stats['cv']:.2%}和样本量={len(transactions)}，推荐XX阈值因为..."
}}

阈值范围: 1.5-3.0之间
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.0,  # 稳定输出
            messages=[
                {"role": "system", "content": "你是数据分析专家，擅长统计分析和异常检测。"},
                {"role": "user", "content": prompt}
            ],
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        threshold = float(data.get("threshold", 2.5))

        # 验证范围
        if not (1.5 <= threshold <= 3.0):
            logger.warning(f"LLM推荐阈值{threshold}超出范围，使用2.5")
            return 2.5

        logger.info(f"LLM推荐异常检测阈值: {threshold}")
        return threshold

    except Exception as e:
        logger.warning(f"LLM阈值推荐失败: {e}")
        return None
```

**集成代码**:

```python
def compute_anomaly_report(
    transactions: Iterable[Transaction],
    *,
    base_threshold: float = 2.5,
    whitelist_merchants: Iterable[str] | None = None,
) -> Dict[str, object]:
    """异常检测（增强版：LLM推荐阈值）"""

    transactions = list(transactions)
    # ... 前面的过滤逻辑保持不变 ...

    # 计算统计量
    df = _to_dataframe(filtered)
    if not df.empty:
        stats = {
            'mean': df['amount'].mean(),
            'std': df['amount'].std(ddof=0),
            'cv': df['amount'].std(ddof=0) / df['amount'].mean() if df['amount'].mean() > 0 else 0,
            'max': df['amount'].max(),
            'min': df['amount'].min(),
        }

        # 尝试LLM推荐阈值
        llm_threshold = _recommend_anomaly_threshold_llm(filtered, stats)
        if llm_threshold:
            applied_threshold = llm_threshold
            report["adaptive"] = True
        else:
            # Fallback到原逻辑
            applied_threshold = base_threshold
            if sample_size < 10:
                applied_threshold = max(base_threshold, 3.0)

    # ... 后续检测逻辑保持不变 ...
```

**预期收益**:
- 异常检测更精准（根据用户消费模式调整）
- 减少误报率
- 提供可解释的阈值选择依据

---

### 优先级4: 可投资金额估算 ⚠️

**位置**: `services/recommendation_service.py:155-164`

**问题简述**: 固定比例（<3000→10%, <10000→20%, else→30%）估算可投资金额

**LLM改造**: 略（优先级较低，可在Phase2实施）

---

## 实施路线图

### Phase 1: 核心功能LLM化（第1-2周）

**Week 1: 消费建议生成**
- [ ] 实现 `_generate_personalized_actions_llm()`
- [ ] 集成到 `_month_over_month_insight()`、`_recent_average_insight()`
- [ ] 保留fallback规则
- [ ] 单元测试

**Week 2: 风险评估智能化**
- [ ] 实现 `_conduct_risk_assessment_llm()`
- [ ] 修改 `conduct_risk_assessment()` 和 `generate_allocation()`
- [ ] 添加推理过程展示
- [ ] AB测试准备

### Phase 2: 增强功能（第3周）

- [ ] 实现异常检测阈值LLM推荐
- [ ] 实现可投资比例LLM计算
- [ ] 性能测试和优化
- [ ] 灰度发布

### Phase 3: 监控与优化（第4周）

- [ ] 添加LLM调用监控
- [ ] 收集用户反馈
- [ ] Prompt优化迭代
- [ ] 全量发布

---

## 技术规范

### 1. LLM调用规范

**必须使用@safe_call装饰器**:
```python
from utils.error_handling import safe_call

@safe_call(timeout=30, fallback=None, error_message="XX功能失败")
def llm_function():
    ...
```

**温度设置**:
- 风险评估: `temperature=0.0`（稳定输出）
- 建议生成: `temperature=0.3`（允许创造性）
- 阈值推荐: `temperature=0.0`（精确计算）

### 2. Fallback规则

**必须保留原有硬编码逻辑作为降级方案**:
```python
llm_result = try_llm_method()
if not llm_result:
    logger.info("LLM失败，使用fallback规则")
    return hardcoded_fallback()
```

### 3. JSON Schema验证

**所有LLM返回的JSON必须验证**:
```python
# 清理markdown代码块
content = content.strip()
if content.startswith("```json"):
    content = content[7:]
# ...

data = json.loads(content)

# 验证必需字段
if "required_field" not in data:
    raise ValueError("Missing required field")

# 验证数值范围
if not (0 <= data["value"] <= 1):
    raise ValueError("Value out of range")
```

### 4. 日志记录

**记录LLM推理过程**:
```python
logger.info(f"LLM推荐: {result}")
logger.debug(f"LLM响应原文: {raw_content}")
```

### 5. 超时保护

- 所有LLM调用: 30s超时
- 快速推荐: 15s超时
- 使用`@safe_call`自动处理超时

---

## 监控指标

### LLM调用监控

- LLM调用成功率
- LLM响应时间
- Fallback触发率
- JSON解析失败率

### 业务指标

- 用户建议满意度（需AB测试）
- 风险评估准确度（用户反馈）
- 异常检测精度（误报率/漏报率）

---

## 风险控制

### 技术风险

1. **LLM服务不可用** → Fallback规则保底
2. **LLM返回格式错误** → 严格JSON验证 + Fallback
3. **LLM响应超时** → @safe_call自动降级
4. **成本增加** → 监控API调用量，优化prompt长度

### 业务风险

1. **建议质量下降** → AB测试验证，逐步灰度
2. **用户不信任LLM** → 展示推理过程（explainability）
3. **监管合规** → 所有财务建议附免责声明

---

## 成本估算

假设日活1000用户，每用户触发3次LLM调用：

- GPT-4o-mini: ~$0.15/1M tokens input, ~$0.6/1M tokens output
- 平均每次调用: 500 tokens input + 300 tokens output
- 日成本: 1000 * 3 * (500*0.15 + 300*0.6) / 1000000 ≈ $0.76
- 月成本: ~$23

可接受。

---

## 下一步行动

1. **本周**: 实现 `_generate_personalized_actions_llm()`，完成单元测试
2. **下周**: 集成到analysis.py，开始AB测试
3. **Review**: 每周五Code Review，检查Prompt质量和Fallback覆盖率

---

生成时间: 2025-11-14
分析工具: ThinkDeep + 人工Review
