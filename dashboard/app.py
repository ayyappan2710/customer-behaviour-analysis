"""
app.py — Customer Behaviour Analysis & Purchase Prediction System
Multi-page Streamlit dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import preprocess_pipeline
from src.analysis import analyze_purchases, customer_behaviour_metrics
from src.segmentation import perform_clustering, perform_rfm_clustering, segment_statistics
from src.recommendations import generate_insights, personalised_recommendations
from src.purchase_patterns import (
    category_summary, gender_category_matrix, age_category_matrix,
    payment_pattern, repeat_category_customers, customer_category_heatmap_data
)
from src.trend_prediction import daily_trend, monthly_trend, category_monthly_trend, build_forecast
from src.ml_prediction import train_model, predict_customer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Behaviour Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0E1117; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #9CA3AF; }
.stMetric {
    background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.rec-card {
    background: #1F2937;
    border-left: 4px solid #636EFA;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.score-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #636EFA, #EF553B);
}
h1 { font-size: 2rem !important; }
h2 { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'customer_data.csv')

if not os.path.exists(DATA_PATH):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_data", os.path.join(PROJECT_ROOT, 'data', 'generate_data.py'))
        gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_mod)
        gen_mod.generate_customer_data()
    except Exception as _e:
        st.error(f"Could not auto-generate dataset: {_e}")


@st.cache_data(show_spinner="Loading & processing data…")
def load_data():
    return preprocess_pipeline(DATA_PATH)


@st.cache_data(show_spinner="Building customer profiles…")
def load_customer_summary(_df):
    cs = customer_behaviour_metrics(_df)
    cs, _ = perform_clustering(cs, n_clusters=3)
    return cs


@st.cache_data(show_spinner="Running RFM segmentation…")
def load_rfm(_df):
    rfm, _ = perform_rfm_clustering(_df, n_clusters=4)
    return rfm


@st.cache_resource(show_spinner="Training ML model…")
def load_ml_model(_df):
    return train_model(_df)


df = load_data()

if df is None:
    st.error(f"❌ Cannot load dataset from `{DATA_PATH}`. Please run `python data/generate_data.py` first.")
    st.stop()

customer_summary = load_customer_summary(df)
rfm_df = load_rfm(df)
ml_model, ml_scaler, ml_encoders, ml_metrics, ml_features, ml_X_df, ml_importance = load_ml_model(df)

ALL_CUSTOMERS = sorted(df['Customer ID'].unique().tolist())

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=72)
    st.markdown("## 📈 Analytics Hub")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 Customer Behaviour",
            "👥 Customer Segmentation",
            "🛒 Purchase Patterns",
            "📈 Trend Prediction",
            "🎯 Recommendations",
            "🤖 AI/ML Prediction",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"📁 Dataset: {len(df):,} records · {df['Customer ID'].nunique():,} customers")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD (Home)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("📈 Customer Behaviour Analysis & Purchase Prediction")
    st.markdown("*A professional analytics system built on real customer transaction data.*")
    st.markdown("---")

    purchases = analyze_purchases(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Customers", f"{df['Customer ID'].nunique():,}")
    c2.metric("💰 Total Revenue", f"${purchases.get('total_sales',0):,.0f}")
    c3.metric("🛒 Total Orders", f"{len(df):,}")
    c4.metric("📊 Avg Order Value", f"${purchases.get('average_order_value',0):,.2f}")
    c5.metric("📦 Units Sold", f"{purchases.get('total_quantity_sold',0):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        cat_rev = df.groupby('Product Category', as_index=False)['Total Amount'].sum()
        fig = px.bar(cat_rev, x='Product Category', y='Total Amount',
                     color='Product Category', text_auto='.2s',
                     template='plotly_dark', title='Revenue by Category')
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        monthly = monthly_trend(df)
        fig2 = px.line(monthly, x='YearMonth', y='Monthly_Revenue',
                       markers=True, template='plotly_dark',
                       title='Monthly Revenue Trend')
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        gender_rev = df.groupby('Gender', as_index=False)['Total Amount'].sum()
        fig3 = px.pie(gender_rev, names='Gender', values='Total Amount',
                      hole=0.4, template='plotly_dark',
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      title='Revenue by Gender')
        st.plotly_chart(fig3, use_container_width=True)
    with col_d:
        if not rfm_df.empty and 'Customer Segment' in rfm_df.columns:
            seg_cnt = rfm_df['Customer Segment'].value_counts().reset_index()
            seg_cnt.columns = ['Segment', 'Count']
            fig4 = px.pie(seg_cnt, names='Segment', values='Count',
                          hole=0.4, template='plotly_dark',
                          title='Customer Segment Distribution')
            st.plotly_chart(fig4, use_container_width=True)

    # Quick insights
    st.markdown("### 💡 Quick Insights")
    insights = generate_insights(df, customer_summary)
    cols = st.columns(2)
    for i, ins in enumerate(insights):
        if isinstance(ins, dict):
            with cols[i % 2]:
                st.info(f"**{ins['type']}**\n\n{ins['insight']}\n\n*{ins['recommendation']}*")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER BEHAVIOUR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Customer Behaviour":
    st.title("📊 Customer Behaviour Analysis")
    st.markdown("Explore purchasing behaviour across demographics, categories, and time.")
    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            sel_gender = st.multiselect("Gender", df['Gender'].unique(),
                                        default=df['Gender'].unique())
        with fc2:
            sel_cat = st.multiselect("Product Category",
                                     df['Product Category'].unique(),
                                     default=df['Product Category'].unique())
        with fc3:
            age_groups = df['Age Group'].cat.categories.tolist() if hasattr(df['Age Group'], 'cat') else df['Age Group'].unique().tolist()
            sel_age = st.multiselect("Age Group", age_groups, default=age_groups)
        with fc4:
            sel_pay = st.multiselect("Payment Method",
                                     df['Payment Method'].unique(),
                                     default=df['Payment Method'].unique())
        date_min = df['Purchase Date'].min().date()
        date_max = df['Purchase Date'].max().date()
        d1, d2 = st.columns(2)
        sel_start = d1.date_input("From", value=date_min, min_value=date_min, max_value=date_max)
        sel_end = d2.date_input("To", value=date_max, min_value=date_min, max_value=date_max)

    fdf = df[
        df['Gender'].isin(sel_gender) &
        df['Product Category'].isin(sel_cat) &
        df['Age Group'].isin(sel_age) &
        df['Payment Method'].isin(sel_pay) &
        (df['Purchase Date'].dt.date >= sel_start) &
        (df['Purchase Date'].dt.date <= sel_end)
    ]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    p = analyze_purchases(fdf)
    top_cat_qty = fdf.groupby('Product Category')['Quantity'].sum().idxmax()
    top_cat_rev = fdf.groupby('Product Category')['Total Amount'].sum().idxmax()
    top_cust = fdf.groupby('Customer ID')['Total Amount'].sum().idxmax()

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("👥 Customers", f"{fdf['Customer ID'].nunique():,}")
    k2.metric("🛒 Orders", f"{len(fdf):,}")
    k3.metric("💰 Revenue", f"${p.get('total_sales',0):,.0f}")
    k4.metric("📊 Avg Order", f"${p.get('average_order_value',0):,.2f}")
    k5.metric("📦 Units", f"{p.get('total_quantity_sold',0):,}")

    k6,k7,k8,k9,k10 = st.columns(5)
    k6.metric("🏆 Top Category (Qty)", top_cat_qty)
    k7.metric("💎 Top Category (Rev)", top_cat_rev)
    k8.metric("⭐ Most Active Customer", top_cust)
    k9.metric("🧮 Avg Qty/Order", f"{fdf['Quantity'].mean():.2f}")
    k10.metric("💵 Avg Customer Spend", f"${fdf.groupby('Customer ID')['Total Amount'].sum().mean():,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    tab_a, tab_b, tab_c = st.tabs(["📦 Category Analysis", "👤 Demographics", "📅 Time Trends"])

    with tab_a:
        ca1, ca2 = st.columns(2)
        with ca1:
            cat_r = fdf.groupby('Product Category', as_index=False)['Total Amount'].sum()
            fig = px.bar(cat_r, x='Product Category', y='Total Amount',
                         color='Product Category', text_auto='.2s',
                         template='plotly_dark', title='Revenue by Category')
            st.plotly_chart(fig, use_container_width=True)
        with ca2:
            cat_q = fdf.groupby('Product Category', as_index=False)['Quantity'].sum()
            fig = px.bar(cat_q, x='Product Category', y='Quantity',
                         color='Product Category', text_auto='d',
                         template='plotly_dark', title='Quantity by Category')
            st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        cb1, cb2 = st.columns(2)
        with cb1:
            gen_r = fdf.groupby('Gender', as_index=False)['Total Amount'].sum()
            fig = px.pie(gen_r, names='Gender', values='Total Amount',
                         hole=0.4, template='plotly_dark',
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title='Revenue by Gender')
            st.plotly_chart(fig, use_container_width=True)
        with cb2:
            pay_r = fdf.groupby('Payment Method', as_index=False)['Customer ID'].count()
            fig = px.bar(pay_r, x='Payment Method', y='Customer ID',
                         color='Payment Method', text_auto='d',
                         template='plotly_dark', title='Payment Method Distribution',
                         labels={'Customer ID': 'Order Count'})
            st.plotly_chart(fig, use_container_width=True)

        cb3, cb4 = st.columns(2)
        with cb3:
            spend = fdf.groupby('Customer ID')['Total Amount'].sum()
            fig = px.histogram(spend, nbins=30, template='plotly_dark',
                               title='Customer Spending Distribution',
                               labels={'value': 'Total Spending', 'count': 'Customers'})
            st.plotly_chart(fig, use_container_width=True)
        with cb4:
            if 'Age Group' in fdf.columns:
                age_cat = fdf.groupby(['Age Group', 'Product Category'], observed=True,
                                      as_index=False)['Total Amount'].sum()
                fig = px.bar(age_cat, x='Age Group', y='Total Amount',
                             color='Product Category', barmode='group',
                             template='plotly_dark', title='Age Group vs Category Revenue')
                st.plotly_chart(fig, use_container_width=True)

    with tab_c:
        mon = monthly_trend(fdf)
        ct1, ct2 = st.columns(2)
        with ct1:
            fig = px.line(mon, x='YearMonth', y='Monthly_Revenue',
                          markers=True, template='plotly_dark',
                          title='Monthly Revenue Trend')
            st.plotly_chart(fig, use_container_width=True)
        with ct2:
            fig = px.bar(mon, x='YearMonth', y='Monthly_Orders',
                         template='plotly_dark', title='Monthly Order Count')
            st.plotly_chart(fig, use_container_width=True)

        cat_mon = category_monthly_trend(fdf)
        fig = px.line(cat_mon, x='YearMonth', y='Revenue',
                      color='Product Category', markers=True,
                      template='plotly_dark', title='Category-wise Monthly Revenue')
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("Customers are grouped using **K-Means clustering** on RFM (Recency, Frequency, Monetary) features.")
    st.markdown("---")

    if rfm_df.empty:
        st.warning("Not enough data for segmentation.")
        st.stop()

    seg_stats = segment_statistics(rfm_df)

    # ── Segment overview ──────────────────────────────────────────────────────
    st.subheader("📊 Segment Overview")
    ov1, ov2 = st.columns(2)
    with ov1:
        fig = px.bar(seg_stats, x='Customer Segment', y='Customer_Count',
                     color='Customer Segment', text='Customer_Count',
                     template='plotly_dark', title='Customer Count per Segment')
        st.plotly_chart(fig, use_container_width=True)
    with ov2:
        fig = px.bar(seg_stats, x='Customer Segment', y='Total_Revenue',
                     color='Customer Segment', text_auto='.2s',
                     template='plotly_dark', title='Revenue per Segment')
        st.plotly_chart(fig, use_container_width=True)

    # Stats table
    st.subheader("📋 Segment Statistics")
    st.dataframe(
        seg_stats.style.background_gradient(cmap='Blues', subset=['Total_Revenue']),
        use_container_width=True,
    )

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.subheader("🔵 Cluster Scatter — Spending vs. Frequency")
    fig = px.scatter(
        rfm_df, x='Monetary', y='Frequency',
        color='Customer Segment', size='Avg_Order_Value',
        hover_data=['Customer ID', 'Recency'],
        template='plotly_dark',
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title='RFM Clusters: Spending vs. Purchase Frequency',
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # ── Spending distribution per segment ─────────────────────────────────────
    st.subheader("💰 Spending Distribution by Segment")
    fig = px.box(rfm_df, x='Customer Segment', y='Monetary',
                 color='Customer Segment', template='plotly_dark',
                 title='Monetary Distribution per Segment')
    st.plotly_chart(fig, use_container_width=True)

    # ── Customer lookup ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Customer Profile Lookup")
    sel_cust = st.selectbox("Select a Customer ID", ALL_CUSTOMERS)
    cust_row = rfm_df[rfm_df['Customer ID'] == sel_cust]
    if not cust_row.empty:
        r = cust_row.iloc[0]
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Segment", r.get('Customer Segment', 'N/A'))
        p2.metric("💰 Total Spending", f"${r.get('Monetary',0):,.2f}")
        p3.metric("🛒 Orders", int(r.get('Frequency', 0)))
        p4.metric("📦 Quantity", int(r.get('Total_Quantity', 0)))
        p5.metric("📊 Avg Order", f"${r.get('Avg_Order_Value',0):,.2f}")
        p6.metric("📅 Last Purchase", str(r.get('Last_Purchase', 'N/A'))[:10])
    else:
        st.warning("Customer not found in RFM data.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PURCHASE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Purchase Patterns":
    st.title("🛒 Purchase Pattern Analysis")
    st.info("ℹ️ This dataset contains **Product Category** data. All pattern analysis is performed at the **category level**.")
    st.markdown("---")

    cat_sum = category_summary(df)
    pay_sum = payment_pattern(df)

    # Category summary
    st.subheader("📦 Category Summary")
    st.dataframe(cat_sum.style.background_gradient(cmap='Blues', subset=['Total_Revenue']),
                 use_container_width=True)

    pp1, pp2 = st.columns(2)
    with pp1:
        fig = px.bar(cat_sum, x='Product Category', y='Total_Revenue',
                     color='Product Category', text_auto='.2s',
                     template='plotly_dark', title='Revenue by Category')
        st.plotly_chart(fig, use_container_width=True)
    with pp2:
        fig = px.bar(cat_sum, x='Product Category', y='Unique_Customers',
                     color='Product Category', text_auto='d',
                     template='plotly_dark', title='Unique Customers per Category')
        st.plotly_chart(fig, use_container_width=True)

    # Payment patterns
    st.subheader("💳 Payment Method Patterns")
    pp3, pp4 = st.columns(2)
    with pp3:
        fig = px.bar(pay_sum, x='Payment Method', y='Order_Count',
                     color='Payment Method', text_auto='d',
                     template='plotly_dark', title='Orders by Payment Method')
        st.plotly_chart(fig, use_container_width=True)
    with pp4:
        fig = px.pie(pay_sum, names='Payment Method', values='Total_Revenue',
                     hole=0.4, template='plotly_dark',
                     title='Revenue by Payment Method')
        st.plotly_chart(fig, use_container_width=True)

    # Gender × Category
    st.subheader("👤 Gender × Category Preference")
    gc_mat = gender_category_matrix(df)
    if not gc_mat.empty:
        fig = px.imshow(gc_mat, text_auto='.0f', aspect='auto',
                        color_continuous_scale='Blues', template='plotly_dark',
                        title='Revenue Heatmap: Gender vs Category')
        st.plotly_chart(fig, use_container_width=True)

    # Age × Category
    st.subheader("👶 Age Group × Category Preference")
    ac_mat = age_category_matrix(df)
    if not ac_mat.empty:
        fig = px.imshow(ac_mat, text_auto='d', aspect='auto',
                        color_continuous_scale='Purples', template='plotly_dark',
                        title='Order Count Heatmap: Age Group vs Category')
        st.plotly_chart(fig, use_container_width=True)

    # Repeat buyers
    st.subheader("🔁 Repeat Category Purchasers")
    repeat = repeat_category_customers(df, min_purchases=3)
    if not repeat.empty:
        st.dataframe(repeat.head(30), use_container_width=True)
    else:
        st.info("No customer-category pair with 3+ purchases found.")

    # Customer-Category heatmap
    st.subheader("🔥 Customer × Category Purchase Heatmap (Top 30 Customers)")
    hm = customer_category_heatmap_data(df)
    if not hm.empty:
        fig = px.imshow(hm, aspect='auto', color_continuous_scale='YlOrRd',
                        template='plotly_dark',
                        title='Quantity Heatmap: Top Customers vs Categories')
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — TREND PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Prediction":
    st.title("📈 Trend Prediction & Forecasting")
    st.warning("""
⚠️ **Model Limitation Notice**
This forecast uses a Random Forest Regressor trained on historical monthly data.
Predictions are **indicative trend estimates only** — not financial forecasts.
Accuracy depends on data volume and seasonality.
    """)
    st.markdown("---")

    n_forecast = st.slider("Months to forecast ahead", 1, 6, 3)

    with st.spinner("Building forecast…"):
        history_df, forecast_df, fc_metrics = build_forecast(df, periods_ahead=n_forecast)

    if 'error' in fc_metrics:
        st.error(fc_metrics['error'])
    else:
        # Model metrics
        m1, m2 = st.columns(2)
        m1.metric("📉 MAE (Mean Absolute Error)", f"${fc_metrics.get('MAE', 'N/A'):,}")
        m2.metric("📐 R² Score", fc_metrics.get('R2', 'N/A'))

        # Historical + Forecast chart
        st.subheader("📊 Historical Revenue & Forecast")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['YearMonth'], y=history_df['Monthly_Revenue'],
            mode='lines+markers', name='Historical Revenue',
            line=dict(color='#636EFA', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=history_df['YearMonth'], y=history_df['Predicted_Revenue'],
            mode='lines', name='Model Fit',
            line=dict(color='#EF553B', width=2, dash='dot')
        ))
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df['YearMonth'], y=forecast_df['Predicted_Revenue'],
                mode='lines+markers', name='Forecast',
                line=dict(color='#00CC96', width=2, dash='dash')
            ))
        fig.update_layout(template='plotly_dark', hovermode='x unified',
                          xaxis_title='Month', yaxis_title='Revenue ($)')
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        if not forecast_df.empty:
            st.subheader(f"📋 Forecast for Next {n_forecast} Month(s)")
            fc_display = forecast_df[['YearMonth', 'Predicted_Revenue']].copy()
            fc_display.columns = ['Month', 'Predicted Revenue ($)']
            fc_display['Trend'] = fc_display['Predicted Revenue ($)'].diff().apply(
                lambda x: '📈 Up' if x > 0 else ('📉 Down' if x < 0 else '➡️ Flat')
            ).fillna('—')
            st.dataframe(fc_display, use_container_width=True)

    # Daily trend
    st.subheader("📅 Daily Sales Trend")
    daily = daily_trend(df)
    fig = px.line(daily, x='Date', y='Daily_Revenue', template='plotly_dark',
                  title='Daily Revenue')
    st.plotly_chart(fig, use_container_width=True)

    # Category-wise monthly
    st.subheader("🗂️ Category-Wise Monthly Revenue")
    cat_mon = category_monthly_trend(df)
    fig = px.line(cat_mon, x='YearMonth', y='Revenue',
                  color='Product Category', markers=True,
                  template='plotly_dark', title='Category Monthly Revenue Trend')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.title("🎯 Personalised Recommendations")
    st.markdown("""
Recommendations are generated from **actual customer purchase behaviour** using a
transparent weighted scoring system:

| Component | Weight | Source |
|---|---|---|
| Customer's own purchase frequency | **40%** | Customer transaction history |
| Popularity in customer's segment | **30%** | RFM cluster peer behaviour |
| Global category revenue rank | **30%** | Overall dataset revenue |
    """)
    st.markdown("---")

    # Business-wide insights
    st.subheader("💡 Business-Wide Insights")
    insights = generate_insights(df, customer_summary)
    for ins in insights:
        if isinstance(ins, dict):
            with st.expander(f"📌 {ins['type']}", expanded=True):
                st.write(f"**Insight:** {ins['insight']}")
                st.write(f"**Recommendation:** {ins['recommendation']}")

    st.markdown("---")

    # Personalised
    st.subheader("🔍 Personalised Category Recommendations")
    sel_cust = st.selectbox("Select Customer ID", ALL_CUSTOMERS, key='rec_cust')

    cust_history = df[df['Customer ID'] == sel_cust]
    if not cust_history.empty:
        with st.expander("📜 This customer's purchase history", expanded=False):
            hist_cat = (cust_history.groupby('Product Category')
                                    .agg(Orders=('Customer ID','count'),
                                         Total_Spent=('Total Amount','sum'))
                                    .reset_index()
                                    .sort_values('Orders', ascending=False))
            st.dataframe(hist_cat, use_container_width=True)

    recs = personalised_recommendations(sel_cust, df, rfm_df, top_n=4)
    if recs.empty:
        st.warning("Could not generate recommendations for this customer.")
    else:
        for _, row in recs.iterrows():
            score = row['Score']
            bar_width = int(score)
            st.markdown(f"""
<div class="rec-card">
  <strong>📦 {row['Category']}</strong>
  &nbsp;&nbsp;<span style="color:#636EFA; font-size:1.1rem; font-weight:700;">{score}/100</span><br>
  <div style="background:#374151; border-radius:4px; height:8px; margin:8px 0;">
    <div style="width:{bar_width}%; height:8px; border-radius:4px;
         background:linear-gradient(90deg,#636EFA,#EF553B);"></div>
  </div>
  <small style="color:#9CA3AF;">💡 {row['Reason']}</small>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AI/ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI/ML Prediction":
    st.title("🤖 AI/ML Purchase Prediction")
    st.info("""
**Model:** Random Forest Classifier  
**Target variable (proxy):** A customer is classified as *High Activity* (likely to purchase again) if their historical
purchase frequency exceeds the dataset median. This is derived **entirely from past behaviour** — no future data is used.  
**Features used:** Age, Recency, Frequency, Monetary value, Avg order value, Total quantity, Gender, Preferred category, Preferred payment method.
    """)
    st.markdown("---")

    if ml_model is None:
        st.error("ML model could not be trained. Check that the dataset has enough diverse records.")
        st.stop()

    # ── Model Performance ─────────────────────────────────────────────────────
    st.subheader("📊 Model Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Accuracy", f"{ml_metrics.get('Accuracy','N/A')}%")
    m2.metric("🎯 Precision", f"{ml_metrics.get('Precision','N/A')}%")
    m3.metric("🔁 Recall", f"{ml_metrics.get('Recall','N/A')}%")
    m4.metric("⚖️ F1 Score", f"{ml_metrics.get('F1 Score','N/A')}%")

    split_col1, split_col2 = st.columns(2)
    split_col1.caption(f"🏋️ Train samples: {ml_metrics.get('Train_Size','N/A')}")
    split_col2.caption(f"🧪 Test samples: {ml_metrics.get('Test_Size','N/A')}")

    # Confusion matrix
    cm = ml_metrics.get('Confusion_Matrix', None)
    if cm:
        st.subheader("🔲 Confusion Matrix")
        cm_arr = np.array(cm)
        labels = ['Low Activity', 'High Activity']
        fig_cm = px.imshow(
            cm_arr, x=labels, y=labels,
            text_auto=True, color_continuous_scale='Blues',
            template='plotly_dark',
            title='Confusion Matrix (Test Set)',
            labels=dict(x='Predicted', y='Actual'),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importances
    if ml_importance is not None and not ml_importance.empty:
        st.subheader("📌 Feature Importances")
        fig_fi = px.bar(ml_importance.head(10), x='Importance', y='Feature',
                        orientation='h', template='plotly_dark',
                        color='Importance', color_continuous_scale='Blues',
                        title='Top Feature Importances')
        fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Per-customer prediction ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Predict for a Specific Customer")
    sel_cust = st.selectbox("Select Customer ID", ALL_CUSTOMERS, key='ml_cust')

    result = predict_customer(sel_cust, df, ml_model, ml_scaler, ml_encoders, ml_features)
    if 'error' in result:
        st.error(result['error'])
    elif result:
        r1, r2, r3 = st.columns(3)
        prob = result['probability']
        color = "#00CC96" if prob >= 60 else ("#EF553B" if prob < 40 else "#FFA15A")

        r1.markdown(f"""
<div style="background:#1F2937; border-radius:12px; padding:20px; text-align:center;">
  <div style="font-size:0.85rem; color:#9CA3AF;">Purchase Probability</div>
  <div style="font-size:3rem; font-weight:700; color:{color};">{prob}%</div>
  <div style="font-size:1rem;">{result['label']}</div>
</div>
""", unsafe_allow_html=True)

        r2.metric("🛒 Historical Orders", result['frequency'])
        r2.metric("💰 Total Spending", f"${result['total_spending']:,.2f}")
        r3.metric("📅 Days Since Last Purchase", result['recency_days'])

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Purchase Probability (%)'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': '#374151'},
                    {'range': [40, 70], 'color': '#1F2937'},
                    {'range': [70, 100], 'color': '#111827'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 3},
                    'thickness': 0.75, 'value': 50,
                },
            }
        ))
        fig_gauge.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)