"""Streamlit page for uploading and parsing financial documents."""

from __future__ import annotations

import csv
import io
import json
from datetime import date
from typing import Iterable, List

import pandas as pd
import streamlit as st

from models.entities import OCRParseResult, Transaction
from modules.analysis import generate_insights
from services.ocr_service import OCRService
from utils.session import get_i18n, set_transactions


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
        return [Transaction(**item) for item in data]

    # Otherwise assume CSV.
    csv_stream = io.StringIO(raw_text)
    reader = csv.DictReader(csv_stream)
    if not reader.fieldnames:
        raise ValueError(i18n.t("bill_upload.manual_error_csv_header"))

    transactions: List[Transaction] = []
    for row in reader:
        if not row:
            continue
        transactions.append(
            Transaction(
                id=row.get("id", "").strip() or str(len(transactions) + 1),
                date=row.get("date", "").strip(),
                merchant=row.get("merchant", "").strip(),
                category=row.get("category", "").strip(),
                amount=float(row.get("amount", 0)),
                currency=row.get("currency", "CNY").strip() or "CNY",
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
        st.session_state["analysis_summary"] = [ins.model_dump() for ins in insights]

        st.success(i18n.t("bill_upload.manual_success", count=len(transactions)))
        st.session_state.setdefault("manual_entries", [])
        st.session_state["manual_table_entries"] = []
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

                transactions.append(
                    Transaction(
                        id=f"manual-table-{idx}",
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

    uploaded_files = st.file_uploader(
        i18n.t("bill_upload.uploader_help"),
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(i18n.t("bill_upload.empty"))
        return

    ocr_service = OCRService()
    manual_mode = False
    results: list[OCRParseResult | dict] = []
    total_files = len(uploaded_files)
    total_transactions_detected = 0

    with st.status(i18n.t("bill_upload.processing_status"), expanded=True) as status:
        for idx, uploaded_file in enumerate(uploaded_files, 1):
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
            except Exception as exc:  # pylint: disable=broad-except
                st.error(
                    i18n.t(
                        "bill_upload.file_process_error",
                        filename=filename,
                        error=str(exc),
                    )
                )
                manual_mode = True
        status.update(
            label=i18n.t(
                "bill_upload.all_files_processed",
                total=total_files,
                transactions=total_transactions_detected,
            ),
            state="complete",
            expanded=False,
        )

    if not results:
        st.warning(i18n.t("bill_upload.warning_no_txn"))
        manual_mode = True

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
        set_transactions(transactions)
        st.session_state["ocr_raw_text"] = "\n\n".join(raw_texts)
        st.session_state["ocr_results"] = serialized_results
        st.session_state["uploaded_files_count"] = len(serialized_results)
        insights = generate_insights(transactions)
        st.session_state["analysis_summary"] = [ins.model_dump() for ins in insights]
        st.success(i18n.t("bill_upload.success", count=len(transactions)))
        _render_analysis(transactions, insights, serialized_results, i18n)
    else:
        manual_mode = True

    if manual_mode:
        _render_manual_entry(i18n)
