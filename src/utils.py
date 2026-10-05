"""
utils.py — Shared helper functions used across all modules.
"""
import pandas as pd
import numpy as np


def safe_empty(df) -> bool:
    """Return True if df is None or has no rows."""
    return df is None or df.empty


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an RFM (Recency, Frequency, Monetary) table per customer.

    Recency  = days since last purchase (lower = more recent)
    Frequency = number of orders
    Monetary  = total spending
    """
    if safe_empty(df):
        return pd.DataFrame()

    ref_date = df['Purchase Date'].max()

    rfm = df.groupby('Customer ID').agg(
        Recency=('Purchase Date', lambda x: (ref_date - x.max()).days),
        Frequency=('Customer ID', 'count'),
        Monetary=('Total Amount', 'sum'),
        Total_Quantity=('Quantity', 'sum'),
        Avg_Order_Value=('Total Amount', 'mean'),
        Last_Purchase=('Purchase Date', 'max'),
    ).reset_index()

    # Merge age / gender (first occurrence per customer)
    demo_cols = [c for c in ['Age', 'Gender', 'Age Group',
                              'Payment Method', 'Product Category']
                 if c in df.columns]
    if demo_cols:
        demo = (df[['Customer ID'] + demo_cols]
                .drop_duplicates(subset=['Customer ID']))
        rfm = rfm.merge(demo, on='Customer ID', how='left')

    # Preferred category — the one they purchased most
    if 'Product Category' in df.columns:
        pref_cat = (df.groupby(['Customer ID', 'Product Category'])
                      .size()
                      .reset_index(name='cnt')
                      .sort_values('cnt', ascending=False)
                      .drop_duplicates('Customer ID')
                      [['Customer ID', 'Product Category']]
                      .rename(columns={'Product Category': 'Preferred_Category'}))
        rfm = rfm.merge(pref_cat, on='Customer ID', how='left')

    return rfm


def encode_label(series: pd.Series) -> pd.Series:
    """Simple label-encoding for a categorical Series."""
    return series.astype('category').cat.codes


def month_label(year: int, month: int) -> str:
    """Return a 'YYYY-MM' string."""
    return f"{year}-{month:02d}"
