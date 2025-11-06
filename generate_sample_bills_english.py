"""Generate English-only sample bill images for OCR testing."""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_bill(filename, transactions):
    """Create a sample bill image with transaction data."""
    # Image size
    width, height = 800, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Use default font (works for English/numbers)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Title
    draw.text((50, 30), "CONSUMPTION BILL", fill='black', font=font)
    draw.line([(50, 70), (750, 70)], fill='black', width=2)

    # Headers
    y = 100
    draw.text((50, y), "Date", fill='black', font=font)
    draw.text((200, y), "Merchant", fill='black', font=font)
    draw.text((450, y), "Category", fill='black', font=font)
    draw.text((600, y), "Amount", fill='black', font=font)

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
    draw.text((450, y), "Total:", fill='black', font=font)
    draw.text((600, y), f"¥{total:.2f}", fill='black', font=font)

    # Save
    img.save(filename)
    print(f"Created: {filename}")

# Sample bill 1: Dining transactions
transactions1 = [
    {'date': '2025-11-01', 'merchant': 'Starbucks', 'category': 'Dining', 'amount': 45.0},
    {'date': '2025-11-02', 'merchant': 'McDonald', 'category': 'Dining', 'amount': 38.5},
    {'date': '2025-11-03', 'merchant': 'Meituan', 'category': 'Dining', 'amount': 52.0},
    {'date': '2025-11-04', 'merchant': 'Haidilao', 'category': 'Dining', 'amount': 268.0},
]

# Sample bill 2: Mixed transactions
transactions2 = [
    {'date': '2025-11-05', 'merchant': 'Metro', 'category': 'Transport', 'amount': 6.0},
    {'date': '2025-11-05', 'merchant': 'JD.com', 'category': 'Shopping', 'amount': 199.0},
    {'date': '2025-11-06', 'merchant': 'Freshippo', 'category': 'Dining', 'amount': 85.5},
    {'date': '2025-11-06', 'merchant': 'Didi', 'category': 'Transport', 'amount': 28.0},
]

# Sample bill 3: Shopping transactions
transactions3 = [
    {'date': '2025-11-03', 'merchant': 'Taobao', 'category': 'Shopping', 'amount': 156.8},
    {'date': '2025-11-04', 'merchant': 'Tmall', 'category': 'Shopping', 'amount': 89.0},
    {'date': '2025-11-05', 'merchant': 'JD Digital', 'category': 'Shopping', 'amount': 1299.0},
]

os.makedirs('assets/sample_bills', exist_ok=True)

create_sample_bill('assets/sample_bills/bill_dining.png', transactions1)
create_sample_bill('assets/sample_bills/bill_mixed.png', transactions2)
create_sample_bill('assets/sample_bills/bill_shopping.png', transactions3)

print("\n✅ Generated 3 English sample bill images in assets/sample_bills/")
print("These images can be used for OCR testing without requiring Chinese fonts.")
