import pandas as pd

def analyze_demographics(df):
    """Analyze customer demographics like age and gender."""
    if df is None or df.empty:
        return {}
    
    # Unique customers demographics
    unique_customers = df.drop_duplicates(subset=['Customer ID'])
    
    gender_dist = unique_customers['Gender'].value_counts().to_dict() if 'Gender' in unique_customers.columns else {}
    age_group_dist = unique_customers['Age Group'].value_counts().to_dict() if 'Age Group' in unique_customers.columns else {}
    
    return {
        'gender_distribution': gender_dist,
        'age_group_distribution': age_group_dist
    }

def analyze_purchases(df):
    """Analyze overall purchasing trends."""
    if df is None or df.empty:
        return {}
    
    total_sales = df['Total Amount'].sum() if 'Total Amount' in df.columns else 0
    total_quantity = df['Quantity'].sum() if 'Quantity' in df.columns else 0
    avg_order_val = df['Total Amount'].mean() if 'Total Amount' in df.columns else 0
    
    category_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False).to_dict() if 'Product Category' in df.columns else {}
    category_qty = df.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False).to_dict() if 'Product Category' in df.columns else {}
    
    return {
        'total_sales': total_sales,
        'total_quantity_sold': total_quantity,
        'average_order_value': avg_order_val,
        'sales_by_category': category_sales,
        'quantity_by_category': category_qty
    }

def customer_behaviour_metrics(df):
    """Calculate per-customer metrics."""
    if df is None or df.empty:
        return None
    
    # Aggregate data per customer
    customer_summary = df.groupby('Customer ID').agg(
        Purchase_Frequency=('Customer ID', 'count'),
        Total_Spending=('Total Amount', 'sum'),
        Average_Purchase_Value=('Total Amount', 'mean'),
        Total_Quantity=('Quantity', 'sum')
    ).reset_index()
    
    # Merge demographics back
    demographics = df[['Customer ID', 'Age', 'Gender', 'Age Group']].drop_duplicates(subset=['Customer ID'])
    customer_summary = pd.merge(customer_summary, demographics, on='Customer ID', how='left')
    
    return customer_summary
