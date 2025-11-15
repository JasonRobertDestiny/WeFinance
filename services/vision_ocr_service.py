"""Vision LLM-based OCR service for accurate bill recognition."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from datetime import date
from typing import List, Optional

from dateutil import parser as date_parser
from openai import OpenAI

from models.entities import LineItem, Transaction
from utils.error_handling import safe_call
from utils.transactions import generate_transaction_id

logger = logging.getLogger(__name__)


def _t(key: str, fallback: str) -> str:
    """Best-effort translation helper for environments without Streamlit."""
    try:
        from utils.session import get_i18n  # Imported lazily to avoid heavy deps

        return get_i18n().t(key)
    except Exception:  # pylint: disable=broad-except
        return fallback


TYPO_FIELD_MAP = {
    "amout": "amount",
    "marchant": "merchant",
    "catagory": "category",
}


def _strip_markdown_fences(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _try_json_load(payload: str) -> List[dict] | None:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "transactions" in data and isinstance(data["transactions"], list):
            return data["transactions"]
        return [data]
    return None


def _robust_json_parse(content: str) -> List[dict]:
    """Robust JSON parsing with multiple fallback strategies."""

    text = _strip_markdown_fences(content or "")
    direct = _try_json_load(text)
    if direct is not None:
        return [
            _apply_typo_fix(dict(entry)) for entry in direct  # type: ignore[arg-type]
        ]

    array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if array_match:
        parsed = _try_json_load(array_match.group(0))
        if parsed is not None:
            return [_apply_typo_fix(dict(entry)) for entry in parsed]

    object_matches = re.findall(r"\{[^{}]+\}", text)
    if len(object_matches) > 1:
        joined = "[" + ",".join(object_matches) + "]"
        parsed = _try_json_load(joined)
        if parsed is not None:
            return [_apply_typo_fix(dict(entry)) for entry in parsed]

    typo_fixed = text
    for typo, correct in TYPO_FIELD_MAP.items():
        typo_fixed = typo_fixed.replace(f'"{typo}"', f'"{correct}"')
        typo_fixed = typo_fixed.replace(typo, correct)
    fallback = _try_json_load(typo_fixed)
    if fallback is not None:
        return [_apply_typo_fix(dict(entry)) for entry in fallback]

    logger.error("JSON解析失败，返回空数组。原始片段：%s", text[:200])
    return []


def _apply_typo_fix(entry: dict) -> dict:
    for typo, correct in TYPO_FIELD_MAP.items():
        if typo in entry and correct not in entry:
            entry[correct] = entry.pop(typo)
    return entry


def _validate_and_fix_transaction(
    item: dict,
    idx: int,
    source_hash: str,
) -> Transaction | None:
    """Validate transaction fields and attempt auto-fix."""

    payload = dict(item)
    for typo, correct in TYPO_FIELD_MAP.items():
        if typo in payload and correct not in payload:
            payload[correct] = payload.pop(typo)

    if "amount" not in payload:
        logger.warning("Transaction %s missing amount field", idx)
        return None

    if not payload.get("merchant"):
        payload["merchant"] = "Unknown Merchant"

    raw_date = payload.get("date")
    if raw_date:
        try:
            parsed_date = date_parser.parse(str(raw_date))
            payload["date"] = parsed_date.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            payload["date"] = date.today().isoformat()
    else:
        payload["date"] = date.today().isoformat()

    if "currency" not in payload:
        payload["currency"] = "CNY"

    line_items_data = payload.get("line_items", []) or []
    payload["line_items"] = [LineItem(**entry) for entry in line_items_data]

    receipt_time = payload.get("receipt_time")
    if isinstance(receipt_time, str):
        try:
            payload["receipt_time"] = date_parser.parse(receipt_time).isoformat()
        except (ValueError, TypeError):
            payload.pop("receipt_time", None)

    if not payload.get("id"):
        payload["id"] = generate_transaction_id(
            merchant=payload.get("merchant", ""),
            date_value=payload.get("date", ""),
            amount=float(payload.get("amount", 0)),
            currency=payload.get("currency", "CNY"),
            source_hash=source_hash,
            sequence=idx,
        )

    try:
        return Transaction(**payload)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to convert transaction %s: %s", idx, exc)
        return None


class VisionOCRService:
    """使用视觉大模型进行OCR识别，比传统OCR精度更高."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        初始化视觉OCR服务

        Args:
            model: 视觉模型名称，默认使用 gpt-4o（推荐），也支持 qwen3-vl-plus, gemini-2.5-pro
            api_key: OpenAI兼容API密钥
            base_url: API基础URL
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError(
                _t(
                    "errors.api_key_missing",
                    "OPENAI_API_KEY environment variable not set",
                )
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"初始化视觉OCR服务，使用模型：{self.model}")

    @safe_call(timeout=30, error_message="账单识别失败")
    def extract_transactions_from_image(self, image_bytes: bytes) -> List[Transaction]:
        """
        从图片中提取交易记录

        Args:
            image_bytes: 图片字节数据

        Returns:
            Transaction对象列表
        """
        try:
            # 将图片编码为base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            source_hash = hashlib.sha256(image_bytes).hexdigest()

            # 构造提示词（增强多语言支持和字段容错）
            prompt = """你是一个专业的财务账单识别助手。请仔细分析这张账单图片，提取所有交易记录。

【核心识别规则】：
★ 首先统计图片中有多少笔交易（有几行独立金额就有几笔交易）
★ 然后逐行提取每一笔的详细信息，确保 transactions 数组长度 = transaction_count
★ 如看到合计行，仅用于验证总额，不作为单独交易计数

多语言处理规则：
1. **语言识别**：
   - 如果账单为韩文/日文/泰文等非中英文：
     * 商户名保留原文（不要翻译）
     * 金额(amount)和分类(category)必须提取
     * 如果有英文字段，优先使用英文值
   - 如果账单为中文/英文：正常提取所有字段

2. **字段容错策略**：
   - date缺失 → 尝试从receipt_time推断，或设为null（但标记partial_data=true）
   - merchant缺失 → 从票据抬头/店铺名提取，找不到则设为"Unknown Merchant"
   - category缺失 → 根据商品明细智能推断（食品→餐饮，服装→购物，交通卡→交通）
   - **即使部分字段缺失，也要返回数据，不要直接返回空数组[]**

3. **货币识别增强**：
   - RM 或 MYR → "MYR"（马来西亚林吉特）
   - ฿ 或 THB → "THB"（泰铢）
   - ₩ 或 KRW → "KRW"（韩元）
   - ¥ → "CNY"（人民币）
   - $ → "USD"（美元，但S$为SGD新加坡元）
   - 无符号且无法判断 → 默认"CNY"

4. **提取字段**：
   - date: 日期（YYYY-MM-DD格式）或 null
   - merchant: 商户名称（保持原文）或 "Unknown Merchant"
   - category: 分类（餐饮、交通、购物、娱乐、医疗、教育、其他）
   - amount: 总金额（数字，不带货币符号，必需）
   - currency: 货币代码（见上述规则）
   - partial_data: 布尔值（如果有字段被推断，设为true）
   - inferred_fields: 数组（列出哪些字段是推断的，如 ["date", "merchant"]）

5. **详细收据字段**（可选）：
   - line_items: 商品明细数组
   - subtotal: 小计
   - total_discount: 总折扣金额
   - receipt_number: 收据编号

返回格式（纯JSON对象，不要markdown代码块）：
{
  "transaction_count": 4,  // 图片中的交易总数（必填）
  "transactions": [        // 交易详细列表（长度必须等于transaction_count）
    {
      "date": "2025-11-01",
      "merchant": "星巴克",
    "category": "餐饮",
    "amount": 45.0,
    "currency": "CNY",
    "partial_data": false,
    "inferred_fields": []
    }
  ]
}

部分字段缺失示例（韩文账单）：
{
  "transaction_count": 1,
  "transactions": [
    {
      "date": null,
      "merchant": "스타벅스",
    "category": "餐饮",
    "amount": 9000.0,
    "currency": "KRW",
    "partial_data": true,
    "inferred_fields": ["date"]
    }
  ]
}

详细收据示例：
{
  "transaction_count": 1,
  "transactions": [
    {
      "date": "2018-12-25",
    "merchant": "BOOK TA.K (TAMAN DAYA) SDN BHD",
    "category": "购物",
    "amount": 9.0,
    "currency": "MYR",
    "line_items": [
      {
        "description": "RF MODELLING CLAY KIDDY FISH",
        "quantity": 1,
        "unit_price": 9.0,
        "amount": 9.0
      }
    ],
    "receipt_number": "TD01167104",
    "partial_data": false,
    "inferred_fields": []
    }
  ]
}

如果图片中没有交易记录，返回：{"transaction_count": 0, "transactions": []}

重要：即使部分字段缺失，也要尝试返回部分数据，并标记inferred_fields。"""

            # 调用视觉模型
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=3000,
                temperature=0.0,  # 确定性输出，新数据结构已解决多行识别问题
            )

            # 解析响应
            content = response.choices[0].message.content
            logger.debug("视觉模型原始响应: %s", content)

            response_data = _robust_json_parse(content)

            # 新格式：{transaction_count, transactions}
            if isinstance(response_data, dict) and "transactions" in response_data:
                transaction_count = response_data.get("transaction_count", 0)
                transactions_data = response_data.get("transactions", [])
                logger.info(
                    f"LLM声明识别到 {transaction_count} 条交易，实际返回 {len(transactions_data)} 条"
                )
            # 兼容旧格式：直接返回数组
            elif isinstance(response_data, list):
                transactions_data = response_data
                logger.warning("LLM返回旧格式数组，未提供transaction_count")
            else:
                logger.error(f"无法识别的响应格式: {type(response_data)}")
                transactions_data = []

            transactions: List[Transaction] = []
            for idx, item in enumerate(transactions_data):
                txn = _validate_and_fix_transaction(item, idx, source_hash)
                if txn:
                    transactions.append(txn)

            logger.info(f"成功从图片中提取 {len(transactions)} 条交易记录")
            return transactions

        except json.JSONDecodeError as exc:
            logger.error("JSON解析失败: %s, 原始响应: %s", exc, content)
            raise
        except Exception as exc:
            logger.error("视觉OCR识别失败: %s", exc)
            raise
