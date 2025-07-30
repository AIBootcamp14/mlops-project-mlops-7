USE ml_data;

CREATE TABLE ml_data.transaction_features (
    customer_id INT,
    order_id INT,
    order_date DATETIME,
    product_id INT,
    product_category VARCHAR(100),
    quantity INT,
    avg_price_per_item DECIMAL(10,2),
    shipping_fee DECIMAL(10,2),
    coupon_used BOOLEAN,
    customer_city VARCHAR(100),
    membership_days INT,
    gst_rate DECIMAL(5,2),
    order_month INT,
    coupon_code VARCHAR(50),
    discount_value DECIMAL(10,2),
    order_amount DECIMAL(10,2),
    label INT, -- 예: 구매 여부 예측용 타겟 컬럼 등
    PRIMARY KEY (order_id, product_id)
);
