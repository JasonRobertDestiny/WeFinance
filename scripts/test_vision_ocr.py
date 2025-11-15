#!/usr/bin/env python3
"""批量运行 Vision LLM OCR，回归测试示例账单。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


# 把仓库根目录加入 sys.path，方便直接 import services.*
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.vision_ocr_service import VisionOCRService  # noqa: E402  pylint: disable=wrong-import-position
from utils.error_handling import UserFacingError  # noqa: E402  pylint: disable=wrong-import-position

ASSETS_DIR = ROOT_DIR / "assets" / "sample_bills"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_TARGETS = [
    ASSETS_DIR / "real",
    ASSETS_DIR / "bill_dining.png",
    ASSETS_DIR / "bill_mixed.png",
    ASSETS_DIR / "bill_shopping.png",
]


def _load_expected_counts(metadata_path: Path) -> Dict[str, int]:
    """从 metadata.json 读取每张图片期望识别的交易条数。"""

    if not metadata_path.exists():
        return {}

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected: Dict[str, int] = {}
    for relative_path, info in payload.items():
        count = info.get("expected_transactions")
        if count is None:
            continue
        abs_path = (metadata_path.parent / relative_path).resolve()
        keys = {str(abs_path), abs_path.as_posix()}
        if abs_path.is_relative_to(ROOT_DIR):
            rel_path = abs_path.relative_to(ROOT_DIR)
            keys.add(str(rel_path))
            keys.add(rel_path.as_posix())
        for key in keys:
            expected[key] = count
    return expected


def _iter_image_files(targets: Sequence[Path]) -> List[Path]:
    """展开目录与文件列表，只保留图片路径。"""

    files: List[Path] = []
    for target in targets:
        if not target.exists():
            print(f"[WARN] 路径不存在：{target}")
            continue
        if target.is_dir():
            for path in sorted(target.rglob("*")):
                if path.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append(path)
        elif target.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(target)
        else:
            print(f"[SKIP] 非图片文件：{target}")
    return files


def _format_relative(path: Path) -> str:
    """将路径转换为相对仓库的友好展示。"""

    try:
        rel_path = path.resolve().relative_to(ROOT_DIR)
        return rel_path.as_posix()
    except ValueError:
        return path.as_posix()


def _get_expected_for(path: Path, expected_map: Dict[str, int]) -> int | None:
    """根据绝对/相对路径检索期望值。"""

    candidates = {
        str(path),
        path.as_posix(),
        str(path.resolve()),
        path.resolve().as_posix(),
    }
    for candidate in candidates:
        if candidate in expected_map:
            return expected_map[candidate]
    return None


def _dump_transactions(
    transactions: Iterable, output_dir: Path, file_path: Path
) -> None:
    """把识别结果写出到 JSON，便于人工核查。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    rel_name = _format_relative(file_path).replace("/", "__")
    output_file = output_dir / f"{rel_name}.json"
    data = [txn.model_dump(mode="json") for txn in transactions]
    output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="批量验证 Vision OCR 识别率")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[path.as_posix() for path in DEFAULT_TARGETS],
        help="要测试的图片路径或目录，默认覆盖 assets/sample_bills",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        help="使用的视觉模型名称，默认读取 OPENAI_MODEL",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="仅列出将被测试的图片，不执行OCR",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="打印每条交易明细，适合调试",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="把识别结果写入 artifacts/ocr_results 目录",
    )
    parser.add_argument(
        "--stop-on-mismatch",
        action="store_true",
        help="一旦条数不符立即停止，方便快速定位",
    )
    parser.add_argument(
        "--output-dir",
        default=(ROOT_DIR / "artifacts" / "ocr_results").as_posix(),
        help="--dump-json 的输出目录，默认 artifacts/ocr_results",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""

    args = parse_args()
    targets = _iter_image_files([Path(path) for path in args.paths])
    if not targets:
        print("未找到任何图片，请检查路径配置。")
        sys.exit(1)

    print("将要测试的图片：")
    for path in targets:
        print(f"  - {_format_relative(path)}")

    if args.list_only:
        return

    metadata_path = ASSETS_DIR / "metadata.json"
    expected_map = _load_expected_counts(metadata_path)

    try:
        service = VisionOCRService(model=args.model)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print("请先在 .env 中配置 OPENAI_API_KEY / OPENAI_BASE_URL 后再运行。")
        sys.exit(1)

    mismatches = 0
    failures = 0
    for path in targets:
        rel_name = _format_relative(path)
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:  # pragma: no cover - 仅用于提示文件权限问题
            print(f"[ERROR] 无法读取 {rel_name}: {exc}")
            failures += 1
            if args.stop_on_mismatch:
                break
            continue

        try:
            transactions = service.extract_transactions_from_image(image_bytes)
        except UserFacingError as exc:
            print(f"[FAIL] {rel_name}: {exc.message}")
            if exc.suggestion:
                print(f"       建议: {exc.suggestion}")
            failures += 1
            if args.stop_on_mismatch:
                break
            continue
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[FAIL] {rel_name}: {exc}")
            failures += 1
            if args.stop_on_mismatch:
                break
            continue

        count = len(transactions)
        expected = _get_expected_for(path, expected_map)
        status = "OK" if expected is None or expected == count else "MISMATCH"

        print(f"[{status}] {rel_name}: 识别 {count} 条", end="")
        if expected is not None:
            print(f"（期望 {expected} 条）")
        else:
            print()

        if args.show_details:
            for txn in transactions:
                print(
                    f"    - {txn.date} | {txn.merchant} | {txn.category} | ¥{txn.amount:.2f}"
                )

        if args.dump_json:
            _dump_transactions(transactions, Path(args.output_dir), path)

        if expected is not None and expected != count:
            mismatches += 1
            if args.stop_on_mismatch:
                break

    total = len(targets)
    passed = total - mismatches - failures
    print("\n测试完成：")
    print(f"  - 成功: {passed}")
    print(f"  - 条数不符: {mismatches}")
    print(f"  - 失败: {failures}")

    if mismatches or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
