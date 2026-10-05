def generate_insights(df, customer_summary):
    """Generate business recommendations based on dynamic data."""
    insights = []
    
    if df is None or df.empty or customer_summary is None or customer_summary.empty:
        return ["Not enough data to generate insights."]
        
    # 1. Product Category insights
    category_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
    if not category_sales.empty:
        top_category = category_sales.index[0]
        bottom_category = category_sales.index[-1]
        
        insights.append({
            'type': 'Data-Driven Insight',
            'insight': f"'{top_category}' is the highest-selling category generating ${category_sales.iloc[0]:.2f}.",
            'recommendation': "Allocate more marketing budget and ensure sufficient stock for this category."
        })
        
        insights.append({
            'type': 'Data-Driven Insight',
            'insight': f"'{bottom_category}' is the lowest-selling category generating only ${category_sales.iloc[-1]:.2f}.",
            'recommendation': "Investigate pricing or run promotional campaigns to boost sales, or consider phasing it out."
        })
        
    # 2. Customer Segmentation insights
    if 'Customer Segment' in customer_summary.columns:
        segment_counts = customer_summary['Customer Segment'].value_counts()
        high_value = customer_summary[customer_summary['Customer Segment'] == 'High-Value Customers']
        
        if not high_value.empty:
            avg_high_val = high_value['Total_Spending'].mean()
            insights.append({
                'type': 'Data-Driven Insight',
                'insight': f"High-Value Customers spend on average ${avg_high_val:.2f}.",
                'recommendation': "Create an exclusive loyalty program offering premium rewards to retain these customers."
            })
            
    # 3. Demographics insights
    if 'Age Group' in df.columns:
        age_sales = df.groupby('Age Group', observed=True)['Total Amount'].sum().sort_values(ascending=False)
        if not age_sales.empty:
            top_age = age_sales.index[0]
            insights.append({
                'type': 'Data-Driven Insight',
                'insight': f"The '{top_age}' age group contributes the most to total sales.",
                'recommendation': "Tailor ad creatives and product recommendations targeting this specific age demographic."
            })
            
    return insights
