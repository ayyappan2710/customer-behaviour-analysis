"""
purchase_patterns.py — Category-level purchase pattern analysis.

NOTE: The dataset contains Product Category, not individual product names.
All analysis is therefore performed at the category level, which is clearly
communicated in the UI.
"""
import pandas as pd
import numpy as np
from src.utils import safe_empty


# ──────────────────────────────────────────────────────────────────────────────
# Category-level aggregations
# ──────────────────────────────────────────────────────────────────────────────

def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return revenue, order count, and quantity per category."""
    if safe_empty(df):
        return pd.DataFrame()

    grp = df.groupby('Product Category').agg(
        Total_Revenue=('Total Amount', 'sum'),
        Order_Count=('Customer ID', 'count'),
        Total_Quantity=('Quantity', 'sum'),
        Avg_Order_Value=('Total Amount', 'mean'),
        Unique_Customers=('Customer ID', 'nunique'),
    ).reset_index().sort_values('Total_Revenue', ascending=False)

    return grp


def gender_category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: Gender vs Product Category (revenue)."""
    if safe_empty(df) or 'Gender' not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        values='Total Amount',
        index='Gender',
        columns='Product Category',
        aggfunc='sum',
        fill_value=0,
    )
    return pivot


def age_category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: Age Group vs Product Category (order count)."""
    if safe_empty(df) or 'Age Group' not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        values='Customer ID',
        index='Age Group',
        columns='Product Category',
        aggfunc='count',
        fill_value=0,
        observed=True,
    )
    return pivot


def payment_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Payment method breakdown — count and revenue."""
    if safe_empty(df) or 'Payment Method' not in df.columns:
        return pd.DataFrame()

    grp = df.groupby('Payment Method').agg(
        Order_Count=('Customer ID', 'count'),
        Total_Revenue=('Total Amount', 'sum'),
    ).reset_index().sort_values('Order_Count', ascending=False)

    return grp


def repeat_category_customers(df: pd.DataFrame, min_purchases: int = 3) -> pd.DataFrame:
    """
    Identify customers who buy from the same category repeatedly.
    Returns customer-category pairs with purchase count >= min_purchases.
    """
    if safe_empty(df):
        return pd.DataFrame()

    grp = (df.groupby(['Customer ID', 'Product Category'])
             .size()
             .reset_index(name='Purchase_Count')
             .query('Purchase_Count >= @min_purchases')
             .sort_values('Purchase_Count', ascending=False))

    return grp


def customer_category_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Customer × Category order-count matrix
    (sampled to top-30 customers by order volume for readability).
    """
    if safe_empty(df):
        return pd.DataFrame()

    top_customers = (df.groupby('Customer ID')
                       .size()
                       .nlargest(30)
                       .index.tolist())

    sub = df[df['Customer ID'].isin(top_customers)]

    pivot = sub.pivot_table(
        values='Quantity',
        index='Customer ID',
        columns='Product Category',
        aggfunc='sum',
        fill_value=0,
    )
    return pivot
