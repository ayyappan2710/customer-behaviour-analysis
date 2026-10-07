def generate_category_insights(category_sales):
    """Generate insights based on product category performance."""
    insights = []

    if not category_sales:
        return insights

    highest_cat = max(category_sales, key=category_sales.get)
    lowest_cat = min(category_sales, key=category_sales.get)

    insights.append({
        'type': 'Category Insight',
        'finding': f"'{highest_cat}' is the most popular category generating the highest revenue.",
        'recommendation': f"Ensure sufficient stock for '{highest_cat}' and consider expanding this product line."
    })

    insights.append({
        'type': 'Category Insight',
        'finding': f"'{lowest_cat}' has the lowest sales performance.",
        'recommendation': f"Investigate pricing or marketing for '{lowest_cat}'. Consider promotional bundles to boost its sales."
    })

    return insights

def generate_segment_insights(df_clustered):
    """Generate insights based on customer segments."""
    insights = []

    if df_clustered is None or 'Customer_Segment' not in df_clustered.columns:
        return insights

    segment_counts = df_clustered['Customer_Segment'].value_counts()

    for segment in segment_counts.index:
        count = segment_counts[segment]
        pct = (count / len(df_clustered)) * 100

        if 'High-Value' in segment:
            insights.append({
                'type': 'Segment Strategy',
                'finding': f"{pct:.1f}% of customers are in the '{segment}' segment.",
                'recommendation': "Implement a VIP loyalty program to retain these high-spending customers. Offer exclusive early access to new products."
            })
        elif 'Occasional' in segment or 'Low-Value' in segment:
            insights.append({
                'type': 'Segment Strategy',
                'finding': f"{pct:.1f}% of customers are in the '{segment}' segment.",
                'recommendation': "Send targeted re-engagement campaigns with discount codes to increase their purchase frequency."
            })
        elif 'Regular' in segment or 'Medium' in segment:
             insights.append({
                'type': 'Segment Strategy',
                'finding': f"{pct:.1f}% of customers are in the '{segment}' segment.",
                'recommendation': "Use cross-selling techniques and personalized recommendations to increase their average order value."
            })

    return insights

def generate_demographic_insights(df):
    """Generate insights based on demographics."""
    insights = []

    # Age insights
    if 'Age Group' in df.columns:
        age_sales = df.groupby('Age Group')['Total Amount'].sum()
        top_age = age_sales.idxmax()
        insights.append({
            'type': 'Demographic Insight',
            'finding': f"The '{top_age}' age group contributes the most to total sales.",
            'recommendation': f"Tailor marketing messages and preferred ad platforms to target the '{top_age}' demographic effectively."
        })

    # Payment method insights
    if 'Payment Method' in df.columns:
        top_payment = df['Payment Method'].value_counts().idxmax()
        insights.append({
            'type': 'Payment Insight',
            'finding': f"'{top_payment}' is the most widely used payment method.",
            'recommendation': f"Ensure the '{top_payment}' gateway is always optimized for seamless checkout experiences."
        })

    return insights

def get_all_recommendations(df_clean, df_clustered, category_sales):
    """Aggregate all insights and recommendations."""
    all_insights = []

    all_insights.extend(generate_category_insights(category_sales))
    all_insights.extend(generate_demographic_insights(df_clean))
    all_insights.extend(generate_segment_insights(df_clustered))

    return all_insights

import pandas as pd

CANDIDATE_PRODUCTS = {
    'Electronics': ['Laptop Bag', 'USB Hub', 'Wireless Headphones', 'Webcam', 'Cooling Pad', 'Smartphone Stand', 'Portable Charger'],
    'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Socks', 'Hat', 'Scarf'],
    'Home & Garden': ['Plant Pot', 'Table Lamp', 'Cushion', 'Vase', 'Wall Art', 'Candles', 'Storage Box'],
    'Sports': ['Yoga Mat', 'Dumbbells', 'Water Bottle', 'Jump Rope', 'Resistance Bands', 'Gym Towel', 'Protein Shaker'],
    'Books': ['Fiction Novel', 'Self-Help Book', 'Biography', 'Cookbook', 'Notebook', 'Planner', 'Bookmarks'],
    'Beauty': ['Moisturizer', 'Face Wash', 'Sunscreen', 'Lip Balm', 'Perfume', 'Hand Cream', 'Makeup Brush']
}

def get_customer_purchase_history(df, customer_id):
    return df[df['Customer ID'] == customer_id]

def calculate_frequency(history, category):
    if history.empty: return 0
    return len(history[history['Product Category'] == category])

def calculate_quantity(history, category):
    if history.empty: return 0
    cat_purchases = history[history['Product Category'] == category]
    return cat_purchases['Quantity'].sum() if not cat_purchases.empty else 0

def calculate_recency(history, category):
    if history.empty: return 0
    cat_purchases = history[history['Product Category'] == category]
    if cat_purchases.empty:
        return 0

    latest_date = pd.to_datetime(cat_purchases['Purchase Date']).max()
    today = pd.to_datetime(history['Purchase Date']).max()
    days_diff = (today - latest_date).days
    return max(0, 365 - days_diff)

def calculate_category_preference(history, category):
    if history.empty: return 0
    total_purchases = len(history)
    cat_purchases = len(history[history['Product Category'] == category])
    return cat_purchases / total_purchases if total_purchases > 0 else 0

def get_recommendation_reason(f_norm, q_norm, r_norm, p_norm, category):
    if f_norm > 0.7 or p_norm > 0.7:
        return f"Recommended based on your frequent purchases in the {category} category."
    elif r_norm > 0.7:
        return f"Recommended based on your recent purchasing behaviour in {category}."
    elif q_norm > 0.7:
        return f"Recommended because you often buy large quantities of {category}."
    else:
        return "Recommended based on similar purchasing patterns."

def get_product_recommendations(df, customer_id, top_n=5):
    history = get_customer_purchase_history(df, customer_id)

    if history.empty:
        return get_popular_products(df, top_n)

    purchased_products = history['Product Name'].unique().tolist()

    cat_freq, cat_qty, cat_recency, cat_pref = {}, {}, {}, {}
    for category in CANDIDATE_PRODUCTS.keys():
        cat_freq[category] = calculate_frequency(history, category)
        cat_qty[category] = calculate_quantity(history, category)
        cat_recency[category] = calculate_recency(history, category)
        cat_pref[category] = calculate_category_preference(history, category)

    max_f = max(cat_freq.values()) if max(cat_freq.values()) > 0 else 1
    max_q = max(cat_qty.values()) if max(cat_qty.values()) > 0 else 1
    max_r = max(cat_recency.values()) if max(cat_recency.values()) > 0 else 1
    max_p = max(cat_pref.values()) if max(cat_pref.values()) > 0 else 1

    products_scored = []

    for category, products in CANDIDATE_PRODUCTS.items():
        f_norm = cat_freq[category] / max_f
        q_norm = cat_qty[category] / max_q
        r_norm = cat_recency[category] / max_r
        p_norm = cat_pref[category] / max_p

        score = (f_norm * 0.35) + (q_norm * 0.20) + (r_norm * 0.25) + (p_norm * 0.20)
        score_val = round(score * 100, 2)

        if score_val == 0:
            continue

        reason = get_recommendation_reason(f_norm, q_norm, r_norm, p_norm, category)

        for p in products:
            products_scored.append({
                'Product': p,
                'Category': category,
                'Score': score_val,
                'Reason': reason
            })

    # Filter out already purchased products if alternatives exist
    unseen_products = [p for p in products_scored if p['Product'] not in purchased_products]
    unseen_products = sorted(unseen_products, key=lambda x: x['Score'], reverse=True)

    seen_products = [p for p in products_scored if p['Product'] in purchased_products]
    seen_products = sorted(seen_products, key=lambda x: x['Score'], reverse=True)

    final_recommendations = unseen_products
    if len(final_recommendations) < top_n:
        final_recommendations.extend(seen_products)

    if len(final_recommendations) < top_n:
        fallback = get_popular_products(df, top_n)
        for item in fallback:
            if item['Product'] not in [p['Product'] for p in final_recommendations]:
                final_recommendations.append(item)

    return final_recommendations[:top_n]

def get_popular_products(df, top_n=5):
    cat_counts = df['Product Category'].value_counts()
    popular_cats = cat_counts.index.tolist()
    fallback = []
    for cat in popular_cats:
        for p in CANDIDATE_PRODUCTS.get(cat, []):
            fallback.append({
                'Product': p,
                'Category': cat,
                'Score': 'N/A',
                'Reason': "Popular products you may like"
            })
            if len(fallback) >= top_n:
                return fallback
    return fallback
