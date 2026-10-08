"""
app.py  —  Customer Intelligence Platform
Premium Glassmorphism Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os

# ── Path setup ────────────────────────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(_DIR, ".."))
for p in [ROOT, _DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Backend imports ───────────────────────────────────────────────────────────
from src.data_preprocessing import preprocess_pipeline
from src.analysis           import analyze_purchases, customer_behaviour_metrics
from src.segmentation       import perform_clustering, perform_rfm_clustering, segment_statistics
from src.recommendations    import generate_insights, personalised_recommendations
from src.purchase_patterns  import (category_summary, gender_category_matrix,
                                     age_category_matrix, payment_pattern,
                                     repeat_category_customers, customer_category_heatmap_data)
from src.trend_prediction   import daily_trend, monthly_trend, category_monthly_trend, build_forecast
from src.ml_prediction      import train_model, predict_customer

# ── Component imports ─────────────────────────────────────────────────────────
from dashboard.components.theme  import apply_theme, page_header, section_header, kpi, divider
from dashboard.components.ui     import (insight_card, rec_card, html_table,
                                          customer_profile_card, probability_display)
from dashboard.components.charts import theme as ct, colors as chart_colors

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(ROOT, "data", "customer_data.csv")
if not os.path.exists(DATA_PATH):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_data", os.path.join(ROOT, "data", "generate_data.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.generate_customer_data()
    except Exception as e:
        st.error(f"Could not generate dataset: {e}")

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    return preprocess_pipeline(DATA_PATH)

@st.cache_data(show_spinner="Building profiles…")
def load_summary(_df):
    cs = customer_behaviour_metrics(_df)
    cs, _ = perform_clustering(cs, n_clusters=3)
    return cs

@st.cache_data(show_spinner="Running RFM…")
def load_rfm(_df):
    rfm, _ = perform_rfm_clustering(_df, n_clusters=4)
    return rfm

@st.cache_resource(show_spinner="Training model…")
def load_ml(_df):
    return train_model(_df)

df = load_data()
if df is None:
    st.error(f"Cannot load dataset from `{DATA_PATH}`. Run `python data/generate_data.py` first.")
    st.stop()

summary_df = load_summary(df)
rfm_df     = load_rfm(df)
ml_model, ml_scaler, ml_encoders, ml_metrics, ml_features, ml_X_df, ml_imp = load_ml(df)
ALL_CUSTS  = sorted(df["Customer ID"].unique().tolist())
COLORS     = chart_colors()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:24px 0 20px;">
  <div style="font-size:1.4rem;font-weight:800;color:#EAEDF2;letter-spacing:-0.03em;">
    <span style="color:#6C63FF;">◆</span> Customer
  </div>
  <div style="font-size:1rem;font-weight:300;color:#8B92A5;margin-top:2px;">Intelligence Platform</div>
</div>
<div style="height:1px;background:rgba(255,255,255,0.08);margin-bottom:16px;"></div>
""", unsafe_allow_html=True)

    PAGE = st.radio("nav", [
        "🏠  Dashboard",
        "📊  Customer Behaviour",
        "👥  Segmentation",
        "🛒  Purchase Patterns",
        "📈  Trend Prediction",
        "🎯  Recommendations",
        "🤖  AI / ML Prediction",
    ], label_visibility="collapsed")

    st.markdown("""
<div style="height:1px;background:rgba(255,255,255,0.08);margin:16px 0;"></div>
<div style="font-size:0.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
            color:#8B92A5;margin-bottom:10px;">System</div>
""", unsafe_allow_html=True)
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        '<span class="dot-green"></span>'
        '<span style="font-size:0.85rem;color:#EAEDF2;">Data Connected</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">'
        '<span class="dot-green"></span>'
        '<span style="font-size:0.85rem;color:#EAEDF2;">Models Ready</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
<div class="gc" style="padding:14px 16px;">
  <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:.08em;color:#8B92A5;margin-bottom:6px;">Dataset</div>
  <div style="font-size:1rem;font-weight:600;color:#EAEDF2;">{len(df):,} records</div>
  <div style="font-size:0.82rem;color:#8B92A5;">{df['Customer ID'].nunique():,} unique customers</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠  Dashboard":
    page_header("Overview", "Executive Dashboard",
                "High-level snapshot of revenue, orders, and customer health.")

    p = analyze_purchases(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Customers",    f"{df['Customer ID'].nunique():,}")
    with c2: kpi("Total Revenue",f"${p.get('total_sales',0):,.0f}")
    with c3: kpi("Total Orders", f"{len(df):,}")
    with c4: kpi("Avg Order",    f"${p.get('average_order_value',0):,.2f}")
    with c5: kpi("Units Sold",   f"{p.get('total_quantity_sold',0):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    r1a, r1b = st.columns([3, 2])
    with r1a:
        cat = df.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=list(cat.index), y=list(cat.values),
                               marker_color=COLORS[:len(cat)],
                               text=[f"${v:,.0f}" for v in cat.values],
                               textposition="auto", marker_cornerradius=5))
        st.plotly_chart(ct(fig, "Revenue by Category"), use_container_width=True)

    with r1b:
        grev = df.groupby("Gender")["Total Amount"].sum()
        fig = go.Figure(go.Pie(labels=list(grev.index), values=list(grev.values),
                               hole=0.62, marker_colors=["#6C63FF","#3B82F6","#22C55E"],
                               textinfo="label+percent"))
        fig.update_traces(textfont_color="#EAEDF2")
        st.plotly_chart(ct(fig, "Revenue by Gender"), use_container_width=True)

    # Row 2
    r2a, r2b = st.columns([3, 2])
    with r2a:
        mon = monthly_trend(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(mon["YearMonth"].astype(str)),
                                 y=list(mon["Monthly_Revenue"]),
                                 mode="lines", line=dict(color="#6C63FF", width=3, shape="spline"),
                                 fill="tozeroy", fillcolor="rgba(108,99,255,0.08)"))
        st.plotly_chart(ct(fig, "Monthly Revenue Trend"), use_container_width=True)

    with r2b:
        if not rfm_df.empty and "Customer Segment" in rfm_df.columns:
            seg = rfm_df["Customer Segment"].value_counts()
            fig = go.Figure(go.Pie(labels=list(seg.index), values=list(seg.values),
                                   hole=0.62, marker_colors=COLORS,
                                   textinfo="label+percent"))
            fig.update_traces(textfont_color="#EAEDF2")
            st.plotly_chart(ct(fig, "Segment Distribution"), use_container_width=True)

    # Insights
    divider()
    section_header("Business Insights")
    insights = generate_insights(df, summary_df)
    ia, ib = st.columns(2)
    for i, ins in enumerate(insights):
        if isinstance(ins, dict):
            with (ia if i % 2 == 0 else ib):
                insight_card(ins["type"], ins["insight"], ins["recommendation"])


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — CUSTOMER BEHAVIOUR
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📊  Customer Behaviour":
    page_header("Exploration", "Customer Behaviour",
                "Explore purchasing behaviour across demographics, categories and time.")

    with st.expander("⚙️  Filters", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        sel_g   = fc1.multiselect("Gender",  df["Gender"].unique(),  default=list(df["Gender"].unique()))
        sel_cat = fc2.multiselect("Category",df["Product Category"].unique(), default=list(df["Product Category"].unique()))
        ag_opts = df["Age Group"].cat.categories.tolist() if hasattr(df["Age Group"], "cat") else list(df["Age Group"].unique())
        sel_age = fc3.multiselect("Age Group", ag_opts, default=ag_opts)
        sel_pay = fc4.multiselect("Payment",df["Payment Method"].unique(), default=list(df["Payment Method"].unique()))
        d1, d2, _ = st.columns([1,1,2])
        dmin = df["Purchase Date"].min().date(); dmax = df["Purchase Date"].max().date()
        sel_s = d1.date_input("From", dmin, dmin, dmax)
        sel_e = d2.date_input("To",   dmax, dmin, dmax)

    fdf = df[
        df["Gender"].isin(sel_g) &
        df["Product Category"].isin(sel_cat) &
        df["Age Group"].isin(sel_age) &
        df["Payment Method"].isin(sel_pay) &
        (df["Purchase Date"].dt.date >= sel_s) &
        (df["Purchase Date"].dt.date <= sel_e)
    ]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    fp = analyze_purchases(fdf)
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi("Filtered Revenue", f"${fp.get('total_sales',0):,.0f}")
    with k2: kpi("Orders", f"{len(fdf):,}")
    with k3: kpi("Top Category", fdf.groupby("Product Category")["Total Amount"].sum().idxmax())
    with k4: kpi("Avg Customer Spend", f"${fdf.groupby('Customer ID')['Total Amount'].sum().mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab_cat, tab_dem, tab_time = st.tabs(["📦  Category Analysis", "👤  Demographics", "📅  Time Trends"])

    with tab_cat:
        c1, c2 = st.columns(2)
        with c1:
            d = fdf.groupby("Product Category", as_index=False)["Total Amount"].sum()
            fig = px.bar(d, x="Product Category", y="Total Amount",
                         color="Product Category", color_discrete_sequence=COLORS)
            st.plotly_chart(ct(fig,"Revenue by Category"), use_container_width=True)
        with c2:
            d = fdf.groupby("Product Category", as_index=False)["Quantity"].sum()
            fig = px.bar(d, x="Product Category", y="Quantity",
                         color="Product Category", color_discrete_sequence=COLORS)
            st.plotly_chart(ct(fig,"Units by Category"), use_container_width=True)

    with tab_dem:
        c1, c2 = st.columns(2)
        with c1:
            d = fdf.groupby("Gender", as_index=False)["Total Amount"].sum()
            fig = px.pie(d, names="Gender", values="Total Amount", hole=0.6, color_discrete_sequence=COLORS)
            st.plotly_chart(ct(fig,"Revenue by Gender"), use_container_width=True)
        with c2:
            d = fdf.groupby("Payment Method", as_index=False)["Customer ID"].count()
            fig = px.bar(d, x="Payment Method", y="Customer ID", color="Payment Method",
                         color_discrete_sequence=COLORS, labels={"Customer ID":"Orders"})
            st.plotly_chart(ct(fig,"Payment Methods"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            spend = fdf.groupby("Customer ID")["Total Amount"].sum()
            fig = px.histogram(spend, nbins=40, color_discrete_sequence=["#6C63FF"],
                               labels={"value":"Total Spending","count":"Customers"})
            st.plotly_chart(ct(fig,"Spending Distribution"), use_container_width=True)
        with c4:
            d = fdf.groupby(["Age Group","Product Category"], observed=True, as_index=False)["Total Amount"].sum()
            fig = px.bar(d, x="Age Group", y="Total Amount", color="Product Category",
                         barmode="group", color_discrete_sequence=COLORS)
            st.plotly_chart(ct(fig,"Age Group × Category"), use_container_width=True)

    with tab_time:
        mon = monthly_trend(fdf)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(mon, x="YearMonth", y="Monthly_Revenue", markers=True,
                          color_discrete_sequence=["#6C63FF"])
            st.plotly_chart(ct(fig,"Monthly Revenue"), use_container_width=True)
        with c2:
            fig = px.bar(mon, x="YearMonth", y="Monthly_Orders",
                         color_discrete_sequence=["#3B82F6"])
            st.plotly_chart(ct(fig,"Monthly Orders"), use_container_width=True)
        cm = category_monthly_trend(fdf)
        fig = px.line(cm, x="YearMonth", y="Revenue", color="Product Category",
                      markers=True, color_discrete_sequence=COLORS)
        st.plotly_chart(ct(fig,"Category Monthly Revenue"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "👥  Segmentation":
    page_header("Intelligence", "Customer Segmentation",
                "Understand the behavioural groups hidden inside your customer base using K-Means RFM clustering.")

    if rfm_df.empty:
        st.warning("Not enough data for segmentation."); st.stop()

    seg_stats = segment_statistics(rfm_df)

    # Segment KPI row
    scols = st.columns(len(seg_stats))
    for i, (_, row) in enumerate(seg_stats.iterrows()):
        with scols[i]:
            kpi(row["Customer Segment"],
                f"{row['Customer_Count']:,}",
                f"${row['Total_Revenue']:,.0f} revenue")

    st.markdown("<br>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        fig = px.scatter(rfm_df, x="Monetary", y="Frequency", color="Customer Segment",
                         size="Avg_Order_Value", hover_data=["Customer ID","Recency"],
                         color_discrete_sequence=COLORS)
        st.plotly_chart(ct(fig,"RFM Scatter: Spending vs Frequency"), use_container_width=True)
    with r2:
        fig = px.box(rfm_df, x="Customer Segment", y="Monetary", color="Customer Segment",
                     color_discrete_sequence=COLORS)
        st.plotly_chart(ct(fig,"Spending Distribution by Segment"), use_container_width=True)

    divider()
    html_table(seg_stats.round(2), "Segment Statistics")

    divider()
    section_header("Customer Profile Lookup")
    sel_c = st.selectbox("Search Customer ID", ALL_CUSTS, key="seg_cust")
    row = rfm_df[rfm_df["Customer ID"] == sel_c]
    if not row.empty:
        r = row.iloc[0]
        customer_profile_card(
            cust_id      = sel_c,
            segment      = r.get("Customer Segment","N/A"),
            monetary     = r.get("Monetary",0),
            frequency    = r.get("Frequency",0),
            recency      = r.get("Recency",0),
            avg_order    = r.get("Avg_Order_Value",0),
            last_purchase= r.get("Last_Purchase","N/A"),
        )
    else:
        st.warning("Customer not found in RFM data.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — PURCHASE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🛒  Purchase Patterns":
    page_header("Behaviour", "Purchase Patterns",
                "Discover category affinity, repeat purchases and cross-dimensional buying behaviour.")

    cat_sum = category_summary(df)
    pay_sum = payment_pattern(df)

    # Top row KPIs
    top_cat  = cat_sum.sort_values("Total_Revenue", ascending=False).iloc[0]
    top_pay  = pay_sum.sort_values("Order_Count", ascending=False).iloc[0]
    k1, k2, k3 = st.columns(3)
    with k1: kpi("Top Category",      top_cat["Product Category"], f"${top_cat['Total_Revenue']:,.0f}")
    with k2: kpi("Top Payment Method",top_pay["Payment Method"],   f"{top_pay['Order_Count']:,} orders")
    with k3: kpi("Total Categories",  str(len(cat_sum)))

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue heatmap — full width
    hm = customer_category_heatmap_data(df)
    if not hm.empty:
        fig = px.imshow(hm, aspect="auto", color_continuous_scale="Blues",
                        text_auto=True)
        st.plotly_chart(ct(fig,"Quantity: Top 30 Customers × Category"), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        gc = gender_category_matrix(df)
        if not gc.empty:
            fig = px.imshow(gc, aspect="auto", color_continuous_scale="Purples", text_auto=".0f")
            st.plotly_chart(ct(fig,"Gender × Category Revenue"), use_container_width=True)
    with r2:
        ac = age_category_matrix(df)
        if not ac.empty:
            fig = px.imshow(ac, aspect="auto", color_continuous_scale="Blues", text_auto="d")
            st.plotly_chart(ct(fig,"Age Group × Category Orders"), use_container_width=True)

    divider()
    r3, r4 = st.columns([2,3])
    with r3:
        section_header("Repeat Purchasers")
        repeat = repeat_category_customers(df, min_purchases=3)
        if not repeat.empty:
            for i, (_, row) in enumerate(repeat.head(8).iterrows()):
                st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
  <div>
    <div style="font-size:0.9rem;font-weight:600;color:#EAEDF2;">{row['Customer ID']}</div>
    <div style="font-size:0.78rem;color:#8B92A5;">{row['Product Category']}</div>
  </div>
  <span style="background:rgba(108,99,255,0.15);color:#6C63FF;font-size:0.82rem;font-weight:600;
               padding:3px 10px;border-radius:20px;">{row['Purchases']} orders</span>
</div>""", unsafe_allow_html=True)
    with r4:
        html_table(cat_sum.round(2), "Category Summary")


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — TREND PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📈  Trend Prediction":
    page_header("Forecasting", "Revenue Forecast",
                "Machine-learning powered revenue projection using a Random Forest Regressor.")

    n = st.slider("Forecast Horizon (months)", 1, 6, 3)
    with st.spinner("Building forecast…"):
        hist, fc, fc_m = build_forecast(df, periods_ahead=n)

    if "error" in fc_m:
        st.error(fc_m["error"])
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("MAE", f"${fc_m.get('MAE',0):,}")
        with k2: kpi("R² Score", str(fc_m.get("R2","N/A")))
        with k3: kpi("Forecast Horizon", f"{n} month{'s' if n>1 else ''}")
        with k4: kpi("Model", "Random Forest")

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(hist["YearMonth"].astype(str)),
                                 y=list(hist["Monthly_Revenue"]),
                                 mode="lines", name="Historical",
                                 line=dict(color="#6C63FF", width=3, shape="spline")))
        fig.add_trace(go.Scatter(x=list(hist["YearMonth"].astype(str)),
                                 y=list(hist["Predicted_Revenue"]),
                                 mode="lines", name="Model Fit",
                                 line=dict(color="#3B82F6", width=2, dash="dot")))
        if not fc.empty:
            fig.add_trace(go.Scatter(x=list(fc["YearMonth"].astype(str)),
                                     y=list(fc["Predicted_Revenue"]),
                                     mode="lines+markers", name="Forecast",
                                     line=dict(color="#22C55E", width=3, dash="dash")))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(ct(fig,"Historical → Model Fit → Forecast"), use_container_width=True)

        if not fc.empty:
            divider()
            fc_disp = fc[["YearMonth","Predicted_Revenue"]].copy()
            fc_disp.columns = ["Month","Predicted Revenue"]
            fc_disp["Predicted Revenue"] = fc_disp["Predicted Revenue"].map(lambda x: f"${x:,.0f}")
            fc_disp["Trend"] = ["📈 Up" if i > 0 else "📉 Down" if i < 0 else "—"
                                 for i in fc["Predicted_Revenue"].diff().fillna(0)]
            html_table(fc_disp, f"Forecast — Next {n} Month(s)")

        divider()
        cat_mon = category_monthly_trend(df)
        fig2 = px.line(cat_mon, x="YearMonth", y="Revenue", color="Product Category",
                       markers=True, color_discrete_sequence=COLORS)
        st.plotly_chart(ct(fig2,"Category-wise Monthly Revenue"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🎯  Recommendations":
    page_header("Targeting", "Personalized Recommendations",
                "Product subcategories ranked using customer behaviour, peer segment intelligence and global revenue rank.")

    # Scoring breakdown
    with st.expander("How are scores calculated?", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Purchase Frequency","40%", "Customer's own history")
        with c2: kpi("Segment Popularity","30%", "Peer behaviour in cluster")
        with c3: kpi("Global Revenue Rank","30%", "Overall dataset performance")

    divider()

    sel_c = st.selectbox("Select Customer ID", ALL_CUSTS, key="rec_cust")
    recs  = personalised_recommendations(sel_c, df, rfm_df, top_n=5)

    # Customer summary
    seg_row = rfm_df[rfm_df["Customer ID"] == sel_c]
    if not seg_row.empty:
        r = seg_row.iloc[0]
        s1, s2, s3, s4 = st.columns(4)
        with s1: kpi("Segment",   r.get("Customer Segment","N/A"))
        with s2: kpi("Orders",    f"{int(r.get('Frequency',0)):,}")
        with s3: kpi("Total Spend",f"${r.get('Monetary',0):,.0f}")
        with s4: kpi("Recency",   f"{int(r.get('Recency',0))} days ago")

    st.markdown("<br>", unsafe_allow_html=True)

    if recs.empty:
        st.warning("Could not generate recommendations for this customer.")
    else:
        section_header("Recommended Subcategories")
        for i, (_, row) in enumerate(recs.iterrows()):
            rec_card(i+1, row["Category"], row["Score"], row["Reason"])

    # Business-wide insights
    divider()
    section_header("Business-Wide Insights")
    ins_all = generate_insights(df, summary_df)
    ia, ib = st.columns(2)
    for i, ins in enumerate(ins_all):
        if isinstance(ins, dict):
            with (ia if i % 2 == 0 else ib):
                insight_card(ins["type"], ins["insight"], ins["recommendation"])


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — AI / ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🤖  AI / ML Prediction":
    page_header("Machine Learning", "AI Customer Activity Prediction",
                "Predict the likelihood of customers remaining highly active using a Random Forest Classifier.")

    if ml_model is None:
        st.error("ML model could not be trained."); st.stop()

    # Model metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi("Accuracy",  f"{ml_metrics.get('Accuracy','N/A')}%")
    with m2: kpi("Precision", f"{ml_metrics.get('Precision','N/A')}%")
    with m3: kpi("Recall",    f"{ml_metrics.get('Recall','N/A')}%")
    with m4: kpi("F1 Score",  f"{ml_metrics.get('F1 Score','N/A')}%")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cm = ml_metrics.get("Confusion_Matrix")
        if cm:
            labels = ["Low Activity","High Activity"]
            fig = px.imshow(np.array(cm), x=labels, y=labels, text_auto=True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(ct(fig,"Confusion Matrix"), use_container_width=True)
    with c2:
        if ml_imp is not None and not ml_imp.empty:
            fig = px.bar(ml_imp.head(8), x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
            fig.update_layout(yaxis={"categoryorder":"total ascending"}, showlegend=False)
            st.plotly_chart(ct(fig,"Feature Importance"), use_container_width=True)

    divider()
    section_header("Single Customer Prediction")
    sel_c = st.selectbox("Select Customer ID", ALL_CUSTS, key="ml_cust")
    result = predict_customer(sel_c, df, ml_model, ml_scaler, ml_encoders, ml_features)

    if "error" in result:
        st.error(result["error"])
    elif result:
        prob = result["probability"]
        pr1, pr2 = st.columns([1, 2])
        with pr1:
            probability_display(prob, result["label"])
        with pr2:
            section_header("Customer Details")
            d1, d2, d3 = st.columns(3)
            with d1: kpi("Historical Orders",        str(result["frequency"]))
            with d2: kpi("Total Spending",            f"${result['total_spending']:,.0f}")
            with d3: kpi("Days Since Last Purchase",  str(result["recency_days"]))

            st.markdown("<br>", unsafe_allow_html=True)
            # Gauge chart
            clr = "#22C55E" if prob >= 60 else ("#EF4444" if prob < 40 else "#F59E0B")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Activity Score", "font":{"size":13,"color":"#8B92A5"}},
                number={"suffix":"%","font":{"size":32,"color":clr}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#8B92A5"},
                    "bar":{"color":clr,"thickness":0.25},
                    "bgcolor":"rgba(0,0,0,0)",
                    "borderwidth":0,
                    "steps":[
                        {"range":[0,40],"color":"rgba(239,68,68,0.15)"},
                        {"range":[40,70],"color":"rgba(245,158,11,0.12)"},
                        {"range":[70,100],"color":"rgba(34,197,94,0.12)"},
                    ],
                    "threshold":{"line":{"color":"white","width":2},"value":50},
                },
            ))
            fig.update_layout(height=220, margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(ct(fig), use_container_width=True)