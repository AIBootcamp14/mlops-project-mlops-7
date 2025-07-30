import csv
import mysql.connector
from datetime import datetime

csv_path = 'backend/mysql/dataset/Discount_info_cleaned.csv'

def parse_date(date_str):
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None  # 날짜 형식 오류시 None 반환

coupons = []

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        valid_from = parse_date(row['valid_from'])
        valid_to = parse_date(row['valid_to'])
        if not valid_from:
            continue  # 유효하지 않은 날짜는 건너뜀

        try:
            discount_value = float(row['discount_value'])
        except ValueError:
            discount_value = 0.0

        coupons.append({
            'category_name': row['category_name'],
            'code': row['code'],
            'discount_value': discount_value,
            'valid_from': valid_from,
            'valid_to': valid_to,
            'discount_type': row['discount_type'],
        })

db_config = {
    'host': 'localhost',
    'port': 3307,
    'user': 'backend_notone',
    'password': 'notone0818',
    'database': 'mlops'
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 1) 카테고리 이름 -> id 매핑 미리 가져오기
cursor.execute("SELECT id, name FROM categories")
category_map = {name: cid for cid, name in cursor.fetchall()}

# 2) coupons 테이블에 INSERT or UPDATE
insert_coupon_sql = """
INSERT INTO coupons (code, discount_type, discount_value, valid_from, valid_to, category_id)
VALUES (%s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    discount_type = VALUES(discount_type),
    discount_value = VALUES(discount_value),
    valid_from = VALUES(valid_from),
    valid_to = VALUES(valid_to),
    category_id = VALUES(category_id);
"""

for c in coupons:
    category_id = category_map.get(c['category_name'])
    if not category_id:
        # 카테고리 없으면 스킵 또는 처리
        print(f"Warning: category '{c['category_name']}' not found. Skipping coupon {c['code']}")
        continue

    cursor.execute(insert_coupon_sql, (
        c['code'], c['discount_type'], c['discount_value'],
        c['valid_from'], c['valid_to'], category_id
    ))

conn.commit()
cursor.close()
conn.close()