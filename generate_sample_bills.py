"""Generate sample bill images for OCR testing."""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_bill(filename, transactions):
    """Create a sample bill image with transaction data."""
    # Image size
    width, height = 800, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Use Noto Sans CJK for Chinese characters
    try:
        # Try Noto CJK fonts (installed via fonts-noto-cjk package)
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 24)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Title
    draw.text((50, 30), "消费账单 / Bill", fill='black', font=font)
    draw.line([(50, 70), (750, 70)], fill='black', width=2)

    # Headers
    y = 100
    draw.text((50, y), "日期", fill='black', font=font)
    draw.text((200, y), "商户", fill='black', font=font)
    draw.text((450, y), "类别", fill='black', font=font)
    draw.text((600, y), "金额", fill='black', font=font)

    y += 40
    draw.line([(50, y), (750, y)], fill='gray', width=1)

    # Transactions
    y += 20
    for txn in transactions:
        draw.text((50, y), txn['date'], fill='black', font=font)
        draw.text((200, y), txn['merchant'], fill='black', font=font)
        draw.text((450, y), txn['category'], fill='black', font=font)
        draw.text((600, y), f"¥{txn['amount']}", fill='black', font=font)
        y += 35

    # Total
    y += 20
    draw.line([(50, y), (750, y)], fill='black', width=2)
    y += 20
    total = sum(t['amount'] for t in transactions)
    draw.text((450, y), "合计/Total:", fill='black', font=font)
    draw.text((600, y), f"¥{total:.2f}", fill='black', font=font)

    # Save
    img.save(filename)
    print(f"Created: {filename}")

# Sample bill 1: Dining transactions
transactions1 = [
    {'date': '2025-11-01', 'merchant': '星巴克', 'category': '餐饮', 'amount': 45.0},
    {'date': '2025-11-02', 'merchant': '麦当劳', 'category': '餐饮', 'amount': 38.5},
    {'date': '2025-11-03', 'merchant': '美团外卖', 'category': '餐饮', 'amount': 52.0},
    {'date': '2025-11-04', 'merchant': '海底捞', 'category': '餐饮', 'amount': 268.0},
]

# Sample bill 2: Mixed transactions
transactions2 = [
    {'date': '2025-11-05', 'merchant': '地铁出行', 'category': '交通', 'amount': 6.0},
    {'date': '2025-11-05', 'merchant': '京东商城', 'category': '购物', 'amount': 199.0},
    {'date': '2025-11-06', 'merchant': '盒马鲜生', 'category': '餐饮', 'amount': 85.5},
    {'date': '2025-11-06', 'merchant': '滴滴出行', 'category': '交通', 'amount': 28.0},
]

# Sample bill 3: Shopping transactions
transactions3 = [
    {'date': '2025-11-03', 'merchant': '淘宝购物', 'category': '购物', 'amount': 156.8},
    {'date': '2025-11-04', 'merchant': '天猫超市', 'category': '购物', 'amount': 89.0},
    {'date': '2025-11-05', 'merchant': '京东数码', 'category': '购物', 'amount': 1299.0},
]

os.makedirs('assets/sample_bills', exist_ok=True)

create_sample_bill('assets/sample_bills/bill_dining.png', transactions1)
create_sample_bill('assets/sample_bills/bill_mixed.png', transactions2)
create_sample_bill('assets/sample_bills/bill_shopping.png', transactions3)

print("\n✅ Generated 3 sample bill images in assets/sample_bills/")
print("These images can be used for OCR testing.")
