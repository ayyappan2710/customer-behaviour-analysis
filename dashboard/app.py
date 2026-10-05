import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.data_preprocessing import preprocess_pipeline
from src.analysis import analyze_purchases, customer_behaviour_metrics
from src.segmentation import perform_clustering
from src.recommendations import generate_insights


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Behaviour Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    h1, h2, h3 {
        color: #FAFAFA !important;
        font-family: 'Inter', sans-serif;
    }

    hr {
        border-top: 1px solid #4B4B4B;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.title("📈 Customer Behaviour Analytics Dashboard")

st.markdown(
    "### *A premium, data-driven system for extracting customer insights and business intelligence.*"
)

st.markdown("---")


# =========================================================
# DATASET PATH
# =========================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "customer_data.csv"
)


# =========================================================
# LOAD AND PROCESS DATA
# =========================================================

def load_and_process():

    try:

        df = preprocess_pipeline(DATA_PATH)

        if df is None or df.empty:
            return None, None

        # Ensure numeric values
        df["Quantity"] = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        ).fillna(0)

        df["Price"] = pd.to_numeric(
            df["Price"],
            errors="coerce"
        ).fillna(0)

        # Always calculate revenue
        df["Total Amount"] = (
            df["Quantity"] * df["Price"]
        )

        # Customer summary
        cust_summary = customer_behaviour_metrics(df)

        # Customer segmentation
        if (
            cust_summary is not None
            and not cust_summary.empty
        ):

            cust_summary, _ = perform_clustering(
                cust_summary,
                n_clusters=3
            )

        return df, cust_summary

    except Exception as e:

        st.error(
            f"Error while processing data: {e}"
        )

        return None, None


df, customer_summary = load_and_process()


# =========================================================
# ERROR CHECK
# =========================================================

if df is None or df.empty:

    st.error(
        f"Error loading dataset from {DATA_PATH}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=80
    )

    st.header("Control Panel")

    st.markdown(
        "Filter the global dataset here."
    )

    selected_gender = st.multiselect(
        "👥 Target Gender",
        options=sorted(
            df["Gender"].dropna().unique()
        ),
        default=sorted(
            df["Gender"].dropna().unique()
        )
    )

    selected_category = st.multiselect(
        "📦 Product Category",
        options=sorted(
            df["Product Category"].dropna().unique()
        ),
        default=sorted(
            df["Product Category"].dropna().unique()
        )
    )

    st.markdown("---")

    st.markdown("**About**")

    st.markdown(
        "Developed for advanced analytics and behavior segmentation."
    )


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df["Gender"].isin(selected_gender))
    &
    (df["Product Category"].isin(selected_category))
].copy()


# =========================================================
# RECALCULATE FILTERED REVENUE
# =========================================================

filtered_df["Quantity"] = pd.to_numeric(
    filtered_df["Quantity"],
    errors="coerce"
).fillna(0)

filtered_df["Price"] = pd.to_numeric(
    filtered_df["Price"],
    errors="coerce"
).fillna(0)

filtered_df["Total Amount"] = (
    filtered_df["Quantity"]
    * filtered_df["Price"]
)


# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

total_customers = filtered_df["Customer ID"].nunique()

total_revenue = float(
    filtered_df["Total Amount"].sum()
)

total_orders = len(filtered_df)

average_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)

total_units = int(
    filtered_df["Quantity"].sum()
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "👥 Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "💰 Total Revenue",
    f"₹{total_revenue:,.2f}"
)

col3.metric(
    "🛒 Total Orders",
    f"{total_orders:,}"
)

col4.metric(
    "📊 Avg Order Value",
    f"₹{average_order_value:,.2f}"
)

col5.metric(
    "📦 Total Units Sold",
    f"{total_units:,}"
)


st.markdown(
    "<br>",
    unsafe_allow_html=True
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Demographics & Sales",
        "🤖 Customer Segmentation",
        "💡 Actionable Insights",
        "🗄️ Raw Data Base"
    ]
)


# =========================================================
# TAB 1 - SALES
# =========================================================

with tab1:

    st.subheader(
        "Sales Breakdown Analysis"
    )

    col1, col2 = st.columns(2)


    # =====================================================
    # PRODUCT CATEGORY REVENUE
    # =====================================================

    with col1:

        st.markdown(
            "### Revenue by Product Category"
        )

        category_data = (
            filtered_df
            .groupby(
                "Product Category"
            )["Total Amount"]
            .sum()
            .reset_index()
        )

        # Explicitly create Python lists
        category_names = [
            str(x)
            for x in category_data[
                "Product Category"
            ].tolist()
        ]

        category_values = [
            float(x)
            for x in category_data[
                "Total Amount"
            ].tolist()
        ]

        category_text = [
            f"₹{x:,.0f}"
            for x in category_values
        ]

        fig_cat = go.Figure()

        fig_cat.add_trace(
            go.Bar(
                x=category_names,
                y=category_values,
                text=category_text,
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Revenue: ₹%{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )

        fig_cat.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Product Category",
            yaxis_title="Revenue (₹)",
            yaxis=dict(
                tickformat=",",
                rangemode="tozero"
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                t=40,
                b=80,
                l=60,
                r=20
            )
        )

        st.plotly_chart(
            fig_cat,
            use_container_width=True
        )


    # =====================================================
    # GENDER REVENUE
    # =====================================================

    with col2:

        st.markdown(
            "### Revenue by Gender"
        )

        gender_data = (
            filtered_df
            .groupby(
                "Gender"
            )["Total Amount"]
            .sum()
            .reset_index()
        )

        gender_names = [
            str(x)
            for x in gender_data[
                "Gender"
            ].tolist()
        ]

        gender_values = [
            float(x)
            for x in gender_data[
                "Total Amount"
            ].tolist()
        ]

        fig_gender = go.Figure()

        fig_gender.add_trace(
            go.Pie(
                labels=gender_names,
                values=gender_values,
                hole=0.45,
                textinfo="label+percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Revenue: ₹%{value:,.2f}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )
        )

        fig_gender.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                t=40,
                b=40,
                l=20,
                r=20
            )
        )

        st.plotly_chart(
            fig_gender,
            use_container_width=True
        )


# =========================================================
# TAB 2 - CUSTOMER SEGMENTATION
# =========================================================

with tab2:

    st.subheader(
        "AI-Powered Customer Segmentation (K-Means)"
    )

    if (
        customer_summary is not None
        and not customer_summary.empty
    ):

        st.markdown(
            "Customers have been automatically grouped into "
            "distinct segments based on spending habits, "
            "purchase frequency, and overall value."
        )

        fig_seg = go.Figure()

        for segment in sorted(
            customer_summary["Customer Segment"].dropna().unique()
        ):

            segment_df = customer_summary[
                customer_summary["Customer Segment"] == segment
            ]

            fig_seg.add_trace(
                go.Scatter(
                    x=segment_df["Total_Spending"],
                    y=segment_df["Purchase_Frequency"],
                    mode="markers",
                    name=str(segment),
                    marker=dict(
                        size=(
                            segment_df[
                                "Average_Purchase_Value"
                            ].fillna(0)
                            .clip(lower=5)
                            .tolist()
                        )
                    ),
                    text=segment_df["Customer ID"],
                    customdata=segment_df["Age"],
                    hovertemplate=(
                        "<b>Customer:</b> %{text}<br>"
                        "<b>Age:</b> %{customdata}<br>"
                        "<b>Spending:</b> ₹%{x:,.2f}<br>"
                        "<b>Frequency:</b> %{y}<br>"
                        "<extra></extra>"
                    )
                )
            )

        fig_seg.update_layout(
            title="Customer Clusters: Spending vs. Frequency",
            template="plotly_dark",
            xaxis_title="Total Spending",
            yaxis_title="Purchase Frequency",
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_seg,
            use_container_width=True
        )

    else:

        st.warning(
            "Customer segmentation data is not available."
        )


# =========================================================
# TAB 3 - ACTIONABLE INSIGHTS
# =========================================================

with tab3:

    st.subheader(
        "Automated Business Recommendations"
    )

    st.markdown(
        "Based on the data and calculated segments, "
        "our system suggests the following strategies:"
    )

    insights = generate_insights(
        df,
        customer_summary
    )

    for ins in insights:

        if isinstance(ins, dict):

            with st.expander(
                f"📌 {ins['type']}",
                expanded=True
            ):

                st.write(
                    f"**Insight:** {ins['insight']}"
                )

                st.write(
                    f"**Recommendation:** "
                    f"{ins['recommendation']}"
                )

        else:

            st.warning(ins)


# =========================================================
# TAB 4 - RAW DATA
# =========================================================

with tab4:

    st.subheader(
        "Customer Master Record"
    )

    if (
        customer_summary is not None
        and not customer_summary.empty
    ):

        st.dataframe(
            customer_summary,
            use_container_width=True,
            height=400
        )

        st.download_button(
            label="⬇️ Download Segmented Customer Summary (CSV)",
            data=customer_summary.to_csv(
                index=False
            ),
            file_name="customer_summary_segments.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.warning(
            "Customer summary is not available."
        )