"""测试Vision OCR服务（GPT-4o）识别账单图片"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from services.vision_ocr_service import VisionOCRService

# 初始化Vision OCR服务（使用gpt-4o）
vision_ocr = VisionOCRService(model="gpt-4o")

# 测试图片
test_images = [
    "assets/sample_bills/bill_dining.png",
    "assets/sample_bills/bill_mixed.png",
    "assets/sample_bills/bill_shopping.png",
]

for image_path in test_images:
    print(f"\n{'='*60}")
    print(f"测试图片: {image_path}")
    print('='*60)

    try:
        # 读取图片
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # 使用Vision OCR识别
        transactions = vision_ocr.extract_transactions_from_image(image_bytes)

        # 打印结果
        if transactions:
            print(f"\n✅ 成功识别到 {len(transactions)} 条交易记录：\n")
            for txn in transactions:
                print(f"  {txn.date} | {txn.merchant:15} | {txn.category:6} | ¥{txn.amount:.2f}")
        else:
            print("\n❌ 未识别到任何交易记录")

    except FileNotFoundError:
        print(f"\n❌ 文件不存在: {image_path}")
    except Exception as e:
        print(f"\n❌ 识别失败: {e}")

print(f"\n{'='*60}")
print("测试完成")
print('='*60)
