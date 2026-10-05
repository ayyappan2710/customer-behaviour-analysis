"""
recommendations.py — Business insights + personalised category recommendations.

Personalised logic (category-level, no individual product data):
  Score = 0.4 × customer's own purchase frequency for that category
        + 0.3 × how popular the category is among customers in the same segment
        + 0.3 × normalised revenue rank of the category globally

All scores are normalised 0-100.  Recommendations exclude the customer's
single most-purchased category (they already know about it) unless it also
ranks highly in the segment/global mix.
"""
import pandas as pd
import numpy as np
from src.utils import safe_empty


# ──────────────────────────────────────────────────────────────────────────────
# Business insights (kept from original, slightly extended)
# ──────────────────────────────────────────────────────────────────────────────

def generate_insights(df, customer_summary):
    """Generate business recommendations based on dynamic data."""
    insights = []

    if safe_empty(df) or safe_empty(customer_summary):
        return ["Not enough data to generate insights."]

    # 1. Product Category insights
    category_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
    if not category_sales.empty:
        top_cat = category_sales.index[0]
        bot_cat = category_sales.index[-1]
        insights.append({
            'type': 'Top Category',
            'insight': f"'{top_cat}' is the highest-revenue category (${category_sales.iloc[0]:,.2f}).",
            'recommendation': "Increase inventory and run targeted ads for this category."
        })
        insights.append({
            'type': 'Underperforming Category',
            'insight': f"'{bot_cat}' generates the least revenue (${category_sales.iloc[-1]:,.2f}).",
            'recommendation': "Consider promotional discounts or review its pricing strategy."
        })

    # 2. Customer Segmentation
    if 'Customer Segment' in customer_summary.columns:
        high_val = customer_summary[customer_summary['Customer Segment'].str.contains('High|VIP', case=False, na=False)]
        if not high_val.empty:
            avg_hv = high_val['Total_Spending'].mean()
            insights.append({
                'type': 'High-Value Customers',
                'insight': f"High-Value customers average ${avg_hv:,.2f} in total spending.",
                'recommendation': "Launch a loyalty programme with exclusive perks to retain these customers."
            })

    # 3. Age Group
    if 'Age Group' in df.columns:
        age_sales = df.groupby('Age Group', observed=True)['Total Amount'].sum().sort_values(ascending=False)
        if not age_sales.empty:
            top_age = age_sales.index[0]
            insights.append({
                'type': 'Top Age Segment',
                'insight': f"The '{top_age}' age group contributes the most to total sales.",
                'recommendation': "Tailor marketing creatives and product recommendations for this demographic."
            })

    # 4. Payment method
    if 'Payment Method' in df.columns:
        pay = df['Payment Method'].value_counts()
        top_pay = pay.index[0]
        insights.append({
            'type': 'Payment Preference',
            'insight': f"'{top_pay}' is the most preferred payment method ({pay.iloc[0]:,} transactions).",
            'recommendation': "Ensure zero downtime for this payment channel and offer exclusive cashback."
        })

    return insights


# ──────────────────────────────────────────────────────────────────────────────
# Personalised recommendations
# ──────────────────────────────────────────────────────────────────────────────

def personalised_recommendations(
    customer_id: str,
    df: pd.DataFrame,
    customer_summary: pd.DataFrame,
    top_n: int = 4,
) -> pd.DataFrame:
    """
    Return a ranked DataFrame of category recommendations for one customer.

    Columns: Category | Score (0-100) | Reason
    """
    if safe_empty(df):
        return pd.DataFrame(columns=['Category', 'Score', 'Reason'])

    # ── Customer's own purchase frequency per category ──
    cust_df = df[df['Customer ID'] == customer_id]
    if cust_df.empty:
        return pd.DataFrame(columns=['Category', 'Score', 'Reason'])

    cust_cat_freq = (cust_df.groupby('Product Category')
                             .size()
                             .rename('Own_Freq'))

    # ── Segment popularity ──
    segment = None
    seg_popularity = pd.Series(dtype=float)
    if customer_summary is not None and 'Customer Segment' in customer_summary.columns:
        row = customer_summary[customer_summary['Customer ID'] == customer_id]
        if not row.empty:
            segment = row.iloc[0]['Customer Segment']
            seg_customers = customer_summary[
                customer_summary['Customer Segment'] == segment
            ]['Customer ID'].tolist()
            seg_df = df[df['Customer ID'].isin(seg_customers)]
            seg_popularity = (seg_df.groupby('Product Category')
                                     .size()
                                     .rename('Seg_Freq'))

    # ── Global category revenue rank ──
    global_revenue = df.groupby('Product Category')['Total Amount'].sum()
    global_rev_norm = (global_revenue / global_revenue.max()).rename('Global_Rev_Norm')

    # ── Combine all categories present in the dataset ──
    all_cats = df['Product Category'].unique()
    score_df = pd.DataFrame(index=all_cats)
    score_df.index.name = 'Category'

    score_df['Own_Freq'] = cust_cat_freq.reindex(score_df.index).fillna(0)
    score_df['Seg_Freq'] = seg_popularity.reindex(score_df.index).fillna(0)
    score_df['Global_Rev_Norm'] = global_rev_norm.reindex(score_df.index).fillna(0)

    # Normalise own & seg freq to 0-1
    if score_df['Own_Freq'].max() > 0:
        score_df['Own_Freq_N'] = score_df['Own_Freq'] / score_df['Own_Freq'].max()
    else:
        score_df['Own_Freq_N'] = 0

    if score_df['Seg_Freq'].max() > 0:
        score_df['Seg_Freq_N'] = score_df['Seg_Freq'] / score_df['Seg_Freq'].max()
    else:
        score_df['Seg_Freq_N'] = 0

    # Weighted score
    score_df['Raw_Score'] = (
        0.40 * score_df['Own_Freq_N'] +
        0.30 * score_df['Seg_Freq_N'] +
        0.30 * score_df['Global_Rev_Norm']
    )

    # Scale to 0-100
    if score_df['Raw_Score'].max() > 0:
        score_df['Score'] = (score_df['Raw_Score'] / score_df['Raw_Score'].max() * 100).round(1)
    else:
        score_df['Score'] = 0

    score_df = score_df.reset_index().sort_values('Score', ascending=False)

    # ── Build reason text ──
    def build_reason(row):
        parts = []
        if row['Own_Freq'] > 0:
            parts.append(f"customer purchased this {int(row['Own_Freq'])}× personally")
        if row['Seg_Freq'] > 0 and segment:
            parts.append(f"popular in '{segment}' segment")
        rev_rank = global_revenue.rank(ascending=False)[row['Category']]
        parts.append(f"global revenue rank #{int(rev_rank)}")
        return '; '.join(parts).capitalize() + '.'

    score_df['Reason'] = score_df.apply(build_reason, axis=1)

    return score_df[['Category', 'Score', 'Reason']].head(top_n).reset_index(drop=True)
