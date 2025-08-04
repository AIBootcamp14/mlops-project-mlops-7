# [file] model/preprocess.py
import os
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from infra.db_models.ml_features import TransactionFeatures
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SCALER_PATH = './scaler.pkl'  # To Do. 받기

def load_transaction_features(session):
    rows = session.query(TransactionFeatures).all()

    data = []
    for row in rows:
        data.append({
            "customer_id": f"USER_{row.customer_id}",         
            "order_id": f"Transaction_{row.order_id}",        
            "order_date": row.order_date,
            "product_id": f"Product_{row.product_id}",        
            "product_category": row.product_category,         
            "quantity": row.quantity,                          
            "avg_price_per_item": float(row.avg_price_per_item),  
            "shipping_fee": float(row.shipping_fee),          
            "coupon_used": "Used",                    
            "coupon_code": row.coupon_code,                  
            "customer_city": row.customer_city,                
            "gender": row.gender,                              
            "membership_days": row.membership_days,            
            "gst_rate": float(row.gst_rate),                   
            "order_month": row.order_month,                     
            "discount_value": float(row.discount_value),       
            "order_amount": float(row.order_amount),           
            "label": row.label,                                
            "user_id": f"{row.customer_id}"          
        })

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} rows from DB")
    logger.info(f"Sample data:\n{df.head(3)}")
    return df

def preprocess_for_inference(df, wide_features=None, deep_features=None):
    exclude_cols = ['user_id', 'label']
    df = df.drop(columns=exclude_cols, errors='ignore')

    df.fillna(0, inplace=True)

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['order_date'] = df['order_date'].dt.month + df['order_date'].dt.day / 31.0

    # 범주형, 수치형 분리
    wide_cols = [col for col in wide_features if col in df.columns]
    deep_cols = [col for col in deep_features if col in df.columns]

    # wide features
    for col in wide_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # deep features
    for col in deep_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # deep features 중 수치형에만 스케일링 적용
    numeric_deep_cols = [col for col in deep_cols if df[col].dtype != 'int' and df[col].dtype != 'object']
    try:
        scaler = joblib.load(SCALER_PATH)
        df[numeric_deep_cols] = scaler.transform(df[numeric_deep_cols])
    except FileNotFoundError:
        pass

    return df[wide_cols], df[deep_cols]