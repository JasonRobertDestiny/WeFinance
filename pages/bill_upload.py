"""Streamlit page for uploading and parsing financial documents."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import streamlit as st

from models.entities import OCRParseResult, Transaction
from modules.analysis import generate_insights
from services.ocr_service import MAX_FILE_SIZE_BYTES, OCRService
from utils.error_handling import UserFacingError
from utils.session import get_i18n, get_transactions, set_analysis_summary, set_transactions
from utils.transactions import generate_transaction_id
from utils.ui_components import render_financial_health_card

STRUCTURED_FILE_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = MAX_FILE_SIZE_BYTES // (1024 * 1024)


def _is_structured_file(filename: str) -> bool:
    """判断是否为可直接导入的结构化文件（CSV或Excel）。"""

    suffix = Path(filename or "").suffix.lower()
    return suffix in STRUCTURED_FILE_EXTENSIONS


def _parse_excel_file(file_bytes: bytes, i18n=None) -> List[Transaction]:
    """Parse Excel file (.xlsx/.xls) into Transaction objects with smart column mapping."""
    i18n = i18n or get_i18n()

    # Column name mapping: Excel column → Transaction field
    COLUMN_MAPPINGS = {
        # Date fields
        "date": "date",
        "posting_date": "date",
        "transaction_date": "date",
        "clear_date": "date",
        "document_create_date": "date",
        # Merchant fields
        "merchant": "merchant",
        "name_customer": "merchant",
        "customer_name": "merchant",
        "vendor": "merchant",
        "supplier": "merchant",
        # Category field (less common, often needs manual input)
        "category": "category",
        "type": "category",
        "transaction_type": "category",
        # Amount fields
        "amount": "amount",
        "total_open_amount": "amount",
        "total_amount": "amount",
        "transaction_amount": "amount",
        "value": "amount",
        # Currency fields
        "currency": "currency",
        "invoice_currency": "currency",
        "transaction_currency": "currency",
    }

    try:
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        # Read Excel file using pandas
        df = pd.read_excel(io.BytesIO(file_bytes))

        if df.empty:
            raise ValueError(i18n.t("bill_upload.manual_error_no_rows"))

        # Map column names (case-insensitive)
        column_map = {}
        mapped_targets = set()  # Track which target fields are already mapped

        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in COLUMN_MAPPINGS:
                target_field = COLUMN_MAPPINGS[col_lower]
                # Only map if this target field hasn't been mapped yet
                if target_field not in mapped_targets:
                    column_map[col] = target_field
                    mapped_targets.add(target_field)

        # Check if we have minimum required fields
        mapped_fields = set(column_map.values())
        if "date" not in mapped_fields:
            raise ValueError(
                f"缺少日期列。Excel文件必须包含以下列之一: posting_date, date, transaction_date, clear_date"
            )
        if "merchant" not in mapped_fields:
            raise ValueError(
                f"缺少商户列。Excel文件必须包含以下列之一: merchant, name_customer, customer_name, vendor"
            )
        if "amount" not in mapped_fields:
            raise ValueError(
                f"缺少金额列。Excel文件必须包含以下列之一: amount, total_open_amount, total_amount"
            )

        # Rename columns according to mapping
        df_renamed = df.rename(columns=column_map)

        transactions: List[Transaction] = []
        for idx, row in enumerate(df_renamed.to_dict("records"), start=1):
            # Skip rows with empty date
            if pd.isna(row.get("date")) or not str(row.get("date", "")).strip():
                continue

            # Parse date field
            date_val = row.get("date")
            if isinstance(date_val, pd.Timestamp):
                date_str = date_val.strftime("%Y-%m-%d")
            elif hasattr(date_val, "isoformat"):
                date_str = date_val.isoformat()
            else:
                date_str = str(date_val).strip()

            # Skip if merchant is missing
            merchant = str(row.get("merchant", "")).strip()
            if not merchant:
                continue

            # Category is optional, use default if missing
            category = str(row.get("category", "")).strip() or "其他"

            try:
                amount = float(row.get("amount", 0))
            except (TypeError, ValueError):
                continue

            # Skip zero or negative amounts
            if amount <= 0:
                continue

            currency = str(row.get("currency", "CNY")).strip() or "CNY"
            txn_id = generate_transaction_id(
                merchant=merchant,
                date_value=date_str,
                amount=amount,
                currency=currency,
                source_hash=file_hash,
                sequence=idx,
            )

            transactions.append(
                Transaction(
                    id=row.get("id", "").strip() or txn_id,
                    date=date_str,
                    merchant=merchant,
                    category=category,
                    amount=amount,
                    currency=currency,
                    payment_method=str(row.get("payment_method", "")).strip() or None,
                )
            )

        if not transactions:
            raise ValueError(
                f"Excel文件中没有有效的交易记录。请确保数据行包含有效的日期、商户和金额。"
            )

        return transactions

    except Exception as exc:
        raise ValueError(
            f"Excel文件解析失败: {str(exc)}"
        ) from exc


def _parse_manual_input(raw_text: str, i18n=None) -> List[Transaction]:
    """Parse manual JSON/CSV input into Transaction objects."""
    i18n = i18n or get_i18n()
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []

    # Try JSON list first.
    if raw_text.startswith("["):
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise ValueError(i18n.t("bill_upload.manual_error_json_root"))

        transactions: List[Transaction] = []
        for idx, item in enumerate(data, start=1):
            payload = dict(item)
            merchant = str(payload.get("merchant", ""))
            date_str = payload.get("date", "")
            amount = float(payload.get("amount", 0))
            currency = payload.get("currency", "CNY")
            if not payload.get("id"):
                payload["id"] = generate_transaction_id(
                    merchant=merchant,
                    date_value=date_str,
                    amount=amount,
                    currency=currency,
                    source_hash="manual-json",
                    sequence=idx,
                )
            transactions.append(Transaction(**payload))
        return transactions

    # Otherwise assume CSV.
    csv_stream = io.StringIO(raw_text)
    reader = csv.DictReader(csv_stream)
    if not reader.fieldnames:
        raise ValueError(i18n.t("bill_upload.manual_error_csv_header"))

    transactions: List[Transaction] = []
    for row in reader:
        if not row:
            continue
        currency = row.get("currency", "CNY").strip() or "CNY"
        txn_id = row.get("id", "").strip() or generate_transaction_id(
            merchant=row.get("merchant", "").strip(),
            date_value=row.get("date", "").strip(),
            amount=float(row.get("amount", 0)),
            currency=currency,
            source_hash="manual-csv",
            sequence=idx,
        )

        transactions.append(
            Transaction(
                id=txn_id,
                date=row.get("date", "").strip(),
                merchant=row.get("merchant", "").strip(),
                category=row.get("category", "").strip(),
                amount=float(row.get("amount", 0)),
                currency=currency,
                payment_method=row.get("payment_method") or None,
            )
        )
    if not transactions:
        raise ValueError(i18n.t("bill_upload.manual_error_no_rows"))
    return transactions


def _render_manual_entry(i18n) -> None:
    """Render manual entry options for users who bypass OCR."""

    def _finalize_manual_transactions(transactions: List[Transaction]) -> None:
        """Persist manual transactions and surface insights."""
        set_transactions(transactions)
        st.session_state["ocr_raw_text"] = ""
        st.session_state["ocr_results"] = []
        st.session_state["uploaded_files_count"] = 0

        insights = generate_insights(transactions)
        insight_payload = [ins.model_dump() for ins in insights]
        set_analysis_summary(insight_payload)

        st.success(i18n.t("bill_upload.manual_success", count=len(transactions)))
        st.session_state.setdefault("manual_entries", [])
        st.session_state["manual_table_entries"] = []
        st.session_state["show_manual_entry"] = False
        _render_analysis(transactions, insights, [], i18n)

    st.subheader(i18n.t("bill_upload.manual_header"))
    st.caption(i18n.t("bill_upload.manual_subtitle"))

    tab_json, tab_table = st.tabs(
        [
            i18n.t("bill_upload.manual_tab_json"),
            i18n.t("bill_upload.manual_tab_table"),
        ]
    )

    with tab_json:
        st.caption(i18n.t("bill_upload.manual_json_desc"))
        st.code(i18n.t("bill_upload.manual_hint"), language="json")
        manual_input = st.text_area(
            i18n.t("bill_upload.manual_textarea_label"),
            placeholder=i18n.t("bill_upload.manual_placeholder"),
            height=200,
            key="manual_json_input",
        )
        if st.button(
            i18n.t("bill_upload.manual_json_button"), key="manual_json_submit"
        ):
            try:
                transactions = _parse_manual_input(manual_input, i18n)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(i18n.t("bill_upload.manual_invalid") + f" ({exc})")
            else:
                _finalize_manual_transactions(transactions)
                return

        csv_file = st.file_uploader(
            i18n.t("bill_upload.manual_csv_label"),
            type=["csv"],
            help=i18n.t("bill_upload.manual_csv_help"),
            key="manual_csv_upload",
        )
        if csv_file is not None:
            try:
                text = csv_file.read().decode("utf-8")
                transactions = _parse_manual_input(text, i18n)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(i18n.t("bill_upload.manual_csv_error", error=exc))
            else:
                _finalize_manual_transactions(transactions)
                return

    with tab_table:
        st.caption(i18n.t("bill_upload.manual_table_hint"))
        table_entries = st.session_state.get("manual_table_entries") or [
            {
                "date": date.today(),
                "merchant": "",
                "category": "",
                "amount": 0.0,
            }
        ]

        table_df = st.data_editor(
            pd.DataFrame(table_entries),
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "date": st.column_config.DateColumn(
                    i18n.t("bill_upload.manual_form_date")
                ),
                "merchant": st.column_config.TextColumn(
                    i18n.t("bill_upload.manual_form_merchant")
                ),
                "category": st.column_config.TextColumn(
                    i18n.t("bill_upload.manual_form_category")
                ),
                "amount": st.column_config.NumberColumn(
                    i18n.t("bill_upload.manual_form_amount"),
                    min_value=0.0,
                    step=0.1,
                ),
            },
            key="manual_table_editor",
        )
        current_rows = table_df.to_dict("records")
        st.session_state["manual_table_entries"] = current_rows

        if st.button(
            i18n.t("bill_upload.manual_table_submit"),
            key="manual_table_submit",
        ):
            transactions: List[Transaction] = []
            for idx, row in enumerate(current_rows, start=1):
                date_raw = row.get("date")
                if not date_raw:
                    continue
                if hasattr(date_raw, "isoformat"):
                    date_val = date_raw.isoformat()
                else:
                    date_val = str(date_raw).strip()

                merchant = str(row.get("merchant") or "").strip()
                category = str(row.get("category") or "").strip()
                if not (date_val and merchant and category):
                    continue

                try:
                    amount = float(row.get("amount") or 0.0)
                except (TypeError, ValueError):
                    st.warning(
                        i18n.t(
                            "bill_upload.manual_table_invalid_amount",
                            row=idx,
                        )
                    )
                    return

                txn_id = generate_transaction_id(
                    merchant=merchant,
                    date_value=date_val,
                    amount=amount,
                    currency="CNY",
                    source_hash="manual-table",
                    sequence=idx,
                )

                transactions.append(
                    Transaction(
                        id=txn_id,
                        date=date_val,
                        merchant=merchant,
                        category=category,
                        amount=amount,
                        currency="CNY",
                        payment_method=None,
                    )
                )

            if not transactions:
                st.warning(i18n.t("bill_upload.manual_table_warning"))
                return

            _finalize_manual_transactions(transactions)
            return


def _render_analysis(
    transactions: Iterable[Transaction],
    insights,
    serialized_results: List[dict],
    i18n,
) -> None:
    """Display analysis summary, insights, and raw text sections."""
    table_rows = [txn.model_dump() for txn in transactions]
    st.subheader(i18n.t("bill_upload.summary_header"))
    if table_rows:
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info(i18n.t("common.no_data"))

    if insights:
        st.subheader(i18n.t("bill_upload.insight_header"))
        for insight in insights:
            st.success(f"{insight.title}：{insight.detail}")

    st.subheader(i18n.t("bill_upload.ocr_header"))
    if serialized_results:
        for result in serialized_results:
            filename = result.get("filename", i18n.t("bill_upload.default_filename"))
            text = result.get("text", "")
            with st.expander(filename):
                st.text(text or i18n.t("bill_upload.no_text"))
    else:
        st.info(i18n.t("bill_upload.no_text"))


def render() -> None:
    """Render the bill upload workflow."""
    i18n = get_i18n()
    st.title(i18n.t("bill_upload.title"))
    st.write(i18n.t("bill_upload.subtitle"))

    # 显示财务健康卡片（整合预算与支出）
    transactions = get_transactions()
    if transactions:
        render_financial_health_card(transactions)

    uploaded_files = st.file_uploader(
        i18n.t("bill_upload.uploader_help", size=MAX_FILE_SIZE_MB),
        type=["png", "jpg", "jpeg", "pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(i18n.t("bill_upload.empty"))
        return

    ocr_service = OCRService()
    manual_mode = bool(st.session_state.get("show_manual_entry", False))
    structured_results: list[OCRParseResult] = []
    ocr_ready_files: list = []
    results: list[OCRParseResult | dict] = []
    total_transactions_detected = 0

    for uploaded_file in uploaded_files:
        filename = getattr(
            uploaded_file,
            "name",
            i18n.t("common.unnamed_file"),
        )
        file_size = getattr(uploaded_file, "size", None)
        if file_size and file_size > MAX_FILE_SIZE_BYTES:
            st.error(
                i18n.t(
                    "bill_upload.file_too_large",
                    filename=filename,
                    size=MAX_FILE_SIZE_MB,
                )
            )
            manual_mode = True
            st.session_state["show_manual_entry"] = True
            continue

        if _is_structured_file(filename):
            try:
                file_bytes = uploaded_file.read()
                if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                    raise ValueError(
                        i18n.t(
                            "bill_upload.file_too_large",
                            filename=filename,
                            size=MAX_FILE_SIZE_MB,
                        )
                    )

                # Try Excel first (handles misnamed .csv files that are actually Excel)
                structured_transactions = None
                file_text = ""
                parse_error = None

                try:
                    # Try parsing as Excel
                    structured_transactions = _parse_excel_file(file_bytes, i18n)
                    file_text = f"Excel file: {filename}"
                except Exception as excel_exc:
                    # If Excel parsing fails, try CSV
                    excel_error = str(excel_exc)
                    try:
                        csv_text = file_bytes.decode("utf-8")
                        structured_transactions = _parse_manual_input(csv_text, i18n)
                        file_text = csv_text
                    except Exception as csv_exc:
                        # Both failed, provide user-friendly error message
                        if "缺少" in excel_error or "missing" in excel_error.lower():
                            parse_error = (
                                f"Excel文件缺少必需的列。请确保包含：\n"
                                f"• 日期列（posting_date / date / transaction_date）\n"
                                f"• 商户列（merchant / name_customer / vendor）\n"
                                f"• 金额列（amount / total_amount）\n"
                                f"当前文件：{filename}"
                            )
                        elif "解析失败" in excel_error or "parse" in excel_error.lower():
                            parse_error = (
                                f"无法读取Excel文件格式。可能原因：\n"
                                f"• 文件已损坏或格式不正确\n"
                                f"• 文件被加密或受保护\n"
                                f"建议：尝试另存为新的.xlsx文件后再上传\n"
                                f"当前文件：{filename}"
                            )
                        else:
                            parse_error = (
                                f"文件导入失败。请检查：\n"
                                f"• 文件是否为有效的Excel (.xlsx/.xls) 或CSV格式\n"
                                f"• 文件内容是否包含有效的交易数据\n"
                                f"• 日期、商户、金额等字段是否完整\n"
                                f"当前文件：{filename}\n"
                                f"详细错误：{excel_error[:80]}"
                            )

                if parse_error or not structured_transactions:
                    raise ValueError(
                        parse_error or i18n.t("bill_upload.manual_error_no_rows")
                    )

            except Exception as exc:  # pylint: disable=broad-except
                st.error(
                    i18n.t(
                        "bill_upload.csv_import_error",
                        filename=filename,
                        error=str(exc),
                    )
                )
                manual_mode = True
                st.session_state["show_manual_entry"] = True
            else:
                structured_results.append(
                    OCRParseResult(
                        filename=filename,
                        text=file_text,
                        transactions=structured_transactions,
                    )
                )
                total_transactions_detected += len(structured_transactions)
                st.success(
                    i18n.t(
                        "bill_upload.csv_import_success",
                        filename=filename,
                        count=len(structured_transactions),
                    )
                )
            continue

        uploaded_file.seek(0)
        ocr_ready_files.append(uploaded_file)

    results.extend(structured_results)

    if not ocr_ready_files and not structured_results:
        st.warning(i18n.t("bill_upload.warning_no_txn"))
        manual_mode = True
        st.session_state["show_manual_entry"] = True
        if manual_mode:
            _render_manual_entry(i18n)
        return

    try:
        if ocr_ready_files:
            total_files = len(ocr_ready_files)
            with st.status(
                i18n.t("bill_upload.processing_status"), expanded=True
            ) as status:
                for idx, uploaded_file in enumerate(ocr_ready_files, 1):
                    filename = getattr(
                        uploaded_file,
                        "name",
                        i18n.t("common.unnamed_file"),
                    )
                    st.write(
                        f"📄 "
                        + i18n.t(
                            "bill_upload.processing_file",
                            current=idx,
                            total=total_files,
                            filename=filename,
                        )
                    )
                    try:
                        file_results = ocr_service.process_files([uploaded_file])
                        results.extend(file_results)
                        if file_results and file_results[0].transactions:
                            txn_list = file_results[0].transactions
                            total_transactions_detected += len(txn_list)
                            st.success(
                                i18n.t(
                                    "bill_upload.recognized_count",
                                    count=len(txn_list),
                                )
                            )
                            for txn in txn_list[:3]:
                                st.caption(
                                    i18n.t(
                                        "bill_upload.transaction_preview",
                                        date=txn.date,
                                        merchant=txn.merchant,
                                        amount=f"{txn.amount:.2f}",
                                    )
                                )
                            if len(txn_list) > 3:
                                st.caption(
                                    i18n.t(
                                        "bill_upload.and_more",
                                        count=len(txn_list) - 3,
                                    )
                                )
                        else:
                            st.warning(i18n.t("bill_upload.no_transactions_in_file"))
                            manual_mode = True
                            st.session_state["show_manual_entry"] = True
                    except UserFacingError:
                        raise
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(
                            i18n.t(
                                "bill_upload.file_process_error",
                                filename=filename,
                                error=str(exc),
                            )
                        )
                        manual_mode = True
                        st.session_state["show_manual_entry"] = True
                processed_total = len(structured_results) + total_files
                status.update(
                    label=i18n.t(
                        "bill_upload.all_files_processed",
                        total=processed_total,
                        transactions=total_transactions_detected,
                    ),
                    state="complete",
                    expanded=False,
                )
    except UserFacingError as err:
        st.error(f"❌ {err.message}")
        if err.suggestion:
            st.info(f"💡 {err.suggestion}")
        st.markdown("---")
        st.markdown(f"**{i18n.t('bill_upload.fallback_option')}**")

        # 3-option guidance for upload failure
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"### {i18n.t('bill_upload.fallback_option_1_title')}")
            st.caption(i18n.t('bill_upload.fallback_option_1_desc'))
            if st.button(
                i18n.t('bill_upload.fallback_option_1_title'),
                key="fallback_reupload",
                use_container_width=True
            ):
                # Clear state and force re-render uploader
                st.session_state["show_manual_entry"] = False
                st.session_state.pop("uploaded_files_count", None)
                st.rerun()

        with col2:
            st.markdown(f"### {i18n.t('bill_upload.fallback_option_2_title')}")
            st.caption(i18n.t('bill_upload.fallback_option_2_desc'))
            if st.button(
                i18n.t('bill_upload.fallback_option_2_title'),
                key="fallback_manual",
                type="primary",
                use_container_width=True
            ):
                st.session_state["show_manual_entry"] = True
                st.rerun()

        with col3:
            st.markdown(f"### {i18n.t('bill_upload.fallback_option_3_title')}")
            st.caption(i18n.t('bill_upload.fallback_option_3_desc'))
            st.caption("📄 上传 .xlsx / .csv 文件")

        if st.session_state.get("show_manual_entry"):
            st.markdown("---")
            _render_manual_entry(i18n)
        return

    if total_transactions_detected:
        st.success(i18n.t("bill_upload.ocr_success", count=total_transactions_detected))

    if not results:
        st.warning(i18n.t("bill_upload.warning_no_txn"))
        manual_mode = True
        st.session_state["show_manual_entry"] = True

    transactions: list[Transaction] = []
    raw_texts: list[str] = []
    serialized_results: list[dict] = []

    for result in results:
        if isinstance(result, OCRParseResult):
            serialized_results.append(result.model_dump())
            raw_texts.append(result.text)
            transactions.extend(result.transactions)
        elif isinstance(result, dict):
            serialized_results.append(result)
            raw_texts.append(result.get("text", ""))
            items = result.get("transactions", [])
            for item in items:
                if isinstance(item, Transaction):
                    transactions.append(item)
                elif isinstance(item, dict):
                    transactions.append(Transaction(**item))

    if transactions:
        # 追加到现有交易列表，而不是覆盖
        existing_transactions = get_transactions()
        all_transactions = list(existing_transactions) + list(transactions)
        set_transactions(all_transactions)
        st.session_state["ocr_raw_text"] = "\n\n".join(raw_texts)
        st.session_state["ocr_results"] = serialized_results
        st.session_state["uploaded_files_count"] = len(serialized_results)
        insights = generate_insights(transactions)
        insight_payload = [ins.model_dump() for ins in insights]
        set_analysis_summary(insight_payload)
        st.success(i18n.t("bill_upload.success", count=len(transactions)))
        st.session_state["show_manual_entry"] = False
        _render_analysis(transactions, insights, serialized_results, i18n)
    else:
        manual_mode = True
        st.session_state["show_manual_entry"] = True

    if manual_mode:
        _render_manual_entry(i18n)


if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    render()
