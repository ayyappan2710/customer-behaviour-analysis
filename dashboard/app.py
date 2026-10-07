import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import preprocess_pipeline
from src.analysis import analyze_purchases, customer_behaviour_metrics
from src.segmentation import perform_clustering
from src.recommendations import get_all_recommendations, get_customer_purchase_history, get_product_recommendations

st.set_page_config(page_title="Customer Behaviour Analysis", layout="wide", page_icon="📊")

# ---------------- Custom CSS for UI/UX Redesign ----------------
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #f3f4f6;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #1f2937;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
        line-height: 1.2;
    }

    /* Recommendation Cards */
    .rec-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border-left: 5px solid #3b82f6;
        border-top: 1px solid #f3f4f6;
        border-right: 1px solid #f3f4f6;
        border-bottom: 1px solid #f3f4f6;
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.1);
        border-left-color: #2563eb;
    }
    .rec-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .rec-rank {
        background-color: #eff6ff;
        color: #2563eb;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    .rec-score-badge {
        background-color: #f0fdf4;
        color: #16a34a;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .rec-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        color: #111827;
    }
    .rec-category {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 4px;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .rec-reason {
        font-size: 0.95rem;
        font-style: italic;
        margin-top: 12px;
        color: #4b5563;
        background-color: #f9fafb;
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #d1d5db;
    }

    /* Progress Bar */
    .progress-container {
        width: 100%;
        background-color: #e5e7eb;
        border-radius: 9999px;
        height: 8px;
        margin-top: 8px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Hero Section */
    .hero-container {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #4b5563;
        font-weight: 400;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f3f4f6;
    }

    /* Dark Mode Adjustments */
    @media (prefers-color-scheme: dark) {
        .metric-card, .rec-card {
            background-color: #1f2937;
            border-color: #374151;
            color: #f3f4f6;
        }
        .rec-card:hover { border-left-color: #60a5fa; }
        .metric-title, .rec-category { color: #9ca3af; }
        .metric-value, .rec-title { color: #f9fafb; }
        .rec-rank { background-color: #1e3a8a; color: #bfdbfe; }
        .rec-score-badge { background-color: #064e3b; color: #a7f3d0; }
        .rec-reason { background-color: #111827; color: #d1d5db; border-left-color: #4b5563; }
        .progress-container { background-color: #374151; }
        .hero-title { background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero-subtitle { color: #9ca3af; }
        .section-header { color: #f3f4f6; border-bottom-color: #374151; }
        .hero-container { border-bottom-color: #374151; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Header / Hero Section ----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📊 Customer Behaviour Analysis</div>
    <div class="hero-subtitle">Analyze customer patterns, purchase behaviour, and personalized product recommendations.</div>
</div>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_and_process_data(filepath):
    if not os.path.exists(filepath):
        return None, None

    df_clean = preprocess_pipeline(filepath)
    if df_clean is not None:
        customer_summary = customer_behaviour_metrics(df_clean)
        df_clustered, kmeans_model = perform_clustering(customer_summary, n_clusters=3)
        return df_clean, df_clustered
    return None, None

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'customer_data.csv')

with st.spinner("Loading and processing data..."):
    df, customer_summary = load_and_process_data(DATA_PATH)

if df is not None and customer_summary is not None:

    # ---------------- Sidebar UX ----------------
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        st.markdown("Use the filters below to slice the data.")

        gender_filter = st.multiselect("👥 Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
        age_filter = st.multiselect("📅 Select Age Group", options=df['Age Group'].dropna().unique(), default=df['Age Group'].dropna().unique())
        category_filter = st.multiselect("📦 Select Product Category", options=df['Product Category'].unique(), default=df['Product Category'].unique())

    # Apply filters to main dataframe
    mask = (df['Gender'].isin(gender_filter)) & \
           (df['Age Group'].isin(age_filter)) & \
           (df['Product Category'].isin(category_filter))

    filtered_df = df[mask]
    metrics = analyze_purchases(filtered_df)

    # ---------------- Overview (KPIs) ----------------
    st.markdown('<div class="section-header">📈 Business Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    def metric_html(icon, title, value):
        return f"""
        <div class="metric-card">
            <div class="metric-title">{icon} {title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    col1.markdown(metric_html("👥", "Customers", f"{len(filtered_df['Customer ID'].unique()):,}"), unsafe_allow_html=True)
    col2.markdown(metric_html("💰", "Total Revenue", f"₹{metrics.get('total_sales', 0):,.2f}"), unsafe_allow_html=True)
    col3.markdown(metric_html("🛒", "Total Orders", f"{metrics.get('total_orders', 0):,}"), unsafe_allow_html=True)
    col4.markdown(metric_html("📊", "Avg Order Value", f"₹{metrics.get('average_order_value', 0):,.2f}"), unsafe_allow_html=True)
    col5.markdown(metric_html("📦", "Units Sold", f"{metrics.get('total_quantity_sold', 0):,}"), unsafe_allow_html=True)

    # ---------------- Analytics Section ----------------
    st.markdown('<div class="section-header">📊 Customer Behaviour Analytics</div>', unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        if not filtered_df.empty:
            cat_sales = filtered_df.groupby('Product Category')['Total Amount'].sum().reset_index()
            fig1 = px.bar(cat_sales, x='Total Amount', y='Product Category', orientation='h',
                          title="Revenue by Category", color='Total Amount', color_continuous_scale='Blues')
            fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        if not filtered_df.empty:
            unique_cust = filtered_df.drop_duplicates(subset=['Customer ID'])
            fig2 = px.histogram(unique_cust, x='Age', nbins=20, title="Customer Age Distribution", color_discrete_sequence=['#8b5cf6'])
            fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Customer Segmentation ----------------
    st.markdown('<div class="section-header">🎯 Customer Segmentation</div>', unsafe_allow_html=True)

    if 'Customer_Segment' in customer_summary.columns:
        col_seg1, col_seg2 = st.columns([2, 1])

        with col_seg1:
            fig_cluster = px.scatter(
                customer_summary,
                x='Total_Spending',
                y='Purchase_Frequency',
                color='Customer_Segment',
                size='Average_Purchase_Value',
                hover_data=['Customer ID'],
                title="Customer Segments: Spending vs Frequency",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_cluster.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cluster, use_container_width=True)

        with col_seg2:
            seg_counts = customer_summary['Customer_Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig_pie = px.pie(seg_counts, values='Count', names='Segment', hole=0.4, title="Segment Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Not enough data to perform clustering.")

    # ---------------- Personalized Recommendations ----------------
    st.markdown('<div class="section-header">🎯 Personalized Product Recommendations</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;'>Select a customer to view their previous purchases and receive dynamically generated product recommendations based on their unique behaviour.</p>", unsafe_allow_html=True)

    # Customer selector with distinct styling
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        customer_list = df['Customer ID'].unique()
        selected_customer = st.selectbox("👤 Select Customer:", customer_list, index=0)

    st.markdown("<br>", unsafe_allow_html=True)

    col_hist, col_rec = st.columns([1.2, 1])

    with col_hist:
        st.markdown("### 📜 Previous Purchase History")
        history = get_customer_purchase_history(df, selected_customer)
        if not history.empty:
            cols_to_show = ['Purchase Date', 'Product Category']
            if 'Product Name' in history.columns:
                cols_to_show.append('Product Name')
            cols_to_show.extend(['Quantity', 'Total Amount', 'Payment Method'])
            st.dataframe(history[cols_to_show].sort_values('Purchase Date', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No purchase history found for this customer.")

    with col_rec:
        st.markdown("### ✨ Recommended For You")
        recommendations = get_product_recommendations(df, selected_customer, top_n=5)

        if recommendations:
            for idx, item in enumerate(recommendations):
                # Clean and parse score
                raw_score = item['Score']
                if raw_score != 'N/A':
                    try:
                        score_val = float(raw_score)
                    except:
                        score_val = 0.0
                else:
                    score_val = 0.0

                score_str = f"{score_val:.1f} / 100" if raw_score != 'N/A' else "N/A"
                progress_width = score_val if raw_score != 'N/A' else 0

                # Render Recommendation Card HTML
                card_html = f"""
                <div class="rec-card">
                    <div class="rec-header">
                        <span class="rec-rank">#{idx+1}</span>
                        <span class="rec-score-badge">⭐ {score_str}</span>
                    </div>
                    <h3 class="rec-title">{item['Product']}</h3>
                    <div class="rec-category">Category: {item.get('Category', 'General')}</div>

                    <div class="progress-container">
                        <div class="progress-bar" style="width: {progress_width}%;"></div>
                    </div>

                    <div class="rec-reason">
                        "{item['Reason']}"
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.info("No recommendations available.")

    # ---------------- Data-Driven Insights ----------------
    st.markdown('<div class="section-header">💡 Business Insights & Recommendations</div>', unsafe_allow_html=True)

    # Generate insights based on the unfiltered metrics to give overall business advice
    overall_metrics = analyze_purchases(df)
    insights = get_all_recommendations(df, customer_summary, overall_metrics.get('revenue_by_category', {}))

    for idx, insight in enumerate(insights):
        with st.expander(f"{insight['type']}: {insight['finding']}", expanded=(idx<2)):
            st.markdown(f"**Recommendation:** {insight['recommendation']}")

    # ---------------- Customer Details Table ----------------
    st.markdown('<div class="section-header">📋 Customer Details</div>', unsafe_allow_html=True)

    st.dataframe(
        customer_summary[['Customer ID', 'Age', 'Gender', 'Purchase_Frequency', 'Total_Spending', 'Average_Purchase_Value', 'Total_Quantity', 'Customer_Segment']],
        use_container_width=True,
        hide_index=True
    )

    # ---------------- Export Data ----------------
    st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)

    # Convert dataframe to CSV for download
    @st.cache_data
    def convert_df(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv_clean = convert_df(df)
    csv_summary = convert_df(customer_summary)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="⬇️ Download Cleaned Dataset (CSV)",
            data=csv_clean,
            file_name='cleaned_customer_data.csv',
            mime='text/csv',
        )
    with col_dl2:
        st.download_button(
            label="⬇️ Download Customer Segmentation Summary (CSV)",
            data=csv_summary,
            file_name='customer_segmentation_summary.csv',
            mime='text/csv',
        )

else:
    st.error("Dataset could not be loaded. Please ensure the file exists at 'data/customer_data.csv'.")
    st.info("Run: `python src/generate_data.py` to generate the dataset.")
