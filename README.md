# CUSTOMER BEHAVIOUR ANALYSIS USING PYTHON

## Project Overview
This project provides a comprehensive data-driven system to analyze customer purchasing behaviour. It converts raw transaction data into meaningful customer-level insights, enabling businesses to understand customer preferences, spending patterns, and distinct customer groups.

## Problem Statement
Businesses collect large amounts of customer purchase data, but raw transaction data alone does not clearly reveal customer preferences, spending patterns, purchasing frequency, or customer groups. Therefore, a data-driven system is required to analyze customer behaviour, identify meaningful customer segments, and generate actionable insights.

## Project Objective
Create a complete customer behaviour analysis system that analyzes customer purchasing data and provides meaningful business insights, identifying:
- Which products customers prefer
- Customer spending patterns
- Frequent and high-spending customers
- Age and gender-wise purchasing behaviour
- Payment-method preferences
- Customer segments

## Features
- **Data Preprocessing**: Robust pipeline for cleaning data, handling missing values, and engineering new features.
- **Exploratory Data Analysis (EDA)**: Detailed analysis of demographics, purchasing behaviour, and revenue.
- **Customer Segmentation**: Unsupervised machine learning using K-Means clustering to group customers based on their spending and frequency.
- **Interactive Dashboard**: Streamlit dashboard for dynamic data visualization and exploration.
- **Actionable Insights**: Rule-based recommendation module generating strategies based on real data calculations.

### UI/UX Features
- **Modern analytics dashboard** with a professional, clean SaaS aesthetic.
- **Responsive layout** suitable for desktop, laptop, and tablet viewing.
- **Professional card-based UI** for metrics, insights, and recommendations.
- **Interactive charts** powered by Plotly with improved visual hierarchy.
- **Personalized recommendation cards** with dynamic progress bar scores and ranked layouts.
- **Hover effects** and micro-interactions on metric cards and recommendations.
- **Smooth animations** for a premium presentation feel.
- **Improved accessibility** with readable typography, clear spacing, and sufficient contrast.

## Personalized Product Recommendation System
"The system analyzes a customer's previous product purchases and generates personalized product recommendations using a weighted scoring model based on frequency, quantity, recency, and category preference."

### Purpose
To increase sales by recommending the most relevant individual products dynamically based on the customer's actual product-level purchasing history.

### How it Works
The recommendation model evaluates potential products across all categories based on the following weighted formula:
- **Frequency (35%)**: How frequently the customer purchased the product or related category.
- **Recency (25%)**: How recently the customer made a purchase related to the product/category.
- **Quantity (20%)**: How many items the customer usually buys in the relevant category.
- **Category Preference (20%)**: The proportion of total purchases that align with the product's category.

The Top 5 products with the highest overall score are presented to the user.

**Previously Purchased Product Filtering**:
The system actively avoids recommending products the customer has already purchased, provided there are enough alternative unseen products available. Already-purchased products are only used as a fallback if necessary.

**Fallback Logic**:
If a customer has no purchase history or insufficient data, the system falls back to recommending popular products globally from the dataset.
## Technologies Used
- **Python 3.x**
- **Pandas & NumPy** (Data manipulation)
- **Matplotlib & Seaborn** (Static visualization)
- **Plotly** (Interactive visualization for dashboard)
- **Scikit-learn** (Machine learning / K-Means clustering)
- **Streamlit** (Web dashboard)
- **Jupyter Notebook** (Interactive analysis)

## Machine Learning Approach (K-Means)
We use the **K-Means clustering algorithm** to segment customers based on behavioral features:
- Purchase Frequency
- Total Spending
- Average Purchase Value

**Important Clarification**: K-Means is used strictly for *Customer Segmentation* (grouping customers with similar traits). It does not predict future purchases.

The optimal number of clusters is determined using the **Elbow Method**. The resulting clusters are dynamically analyzed to assign labels such as "High-Value / Frequent", "Regular / Medium-Value", or "Occasional / Low-Value".

## Project Structure
```
customer-behaviour-analysis/
│
├── data/
│   └── customer_data.csv                 # The dataset (generated synthetically)
│
├── notebooks/
│   └── customer_behaviour_analysis.ipynb # Complete Jupyter Notebook analysis
│
├── src/
│   ├── generate_data.py                  # Script to generate synthetic dataset
│   ├── data_preprocessing.py             # Data cleaning pipeline
│   ├── analysis.py                       # Metric calculations
│   ├── visualization.py                  # Static plotting functions
│   ├── segmentation.py                   # K-Means clustering logic
│   └── recommendations.py                # Rule-based insights
│
├── dashboard/
│   └── app.py                            # Streamlit web application
│
├── outputs/
│   ├── charts/                           # Saved visualization PNGs
│   └── reports/
│
├── README.md                             # Project documentation
└── requirements.txt                      # Required Python packages
```

## Installation Steps
1. Clone or download this repository.
2. Ensure you have Python 3.8+ installed.
3. Open a terminal/command prompt in the project root directory.
4. (Optional) Create a virtual environment: `python -m venv venv` and activate it.
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Step 1: Generate Dataset
Before running the analysis or dashboard, generate the synthetic dataset:
```bash
python src/generate_data.py
```
This will create a `customer_data.csv` file in the `data/` folder.

### Step 2: Run Streamlit Dashboard
To launch the interactive web dashboard:
```bash
streamlit run dashboard/app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

### Step 3: Run Jupyter Notebook
To view the step-by-step academic analysis:
```bash
jupyter notebook notebooks/customer_behaviour_analysis.ipynb
```

## Expected Output
- A generated CSV file containing synthetic transaction data.
- A functional Streamlit dashboard showing KPIs, demographic analysis, category sales, and K-Means segmentation scatter plots.
- Generated static charts saved in `outputs/charts/`.
- Downloadable CSVs for the cleaned dataset and the final customer summary with assigned clusters.

## Future Enhancements
- Integrate a genuine predictive model (e.g., Random Forest or XGBoost) to predict Customer Lifetime Value (CLV) or churn probability.
- Add Market Basket Analysis (Apriori algorithm) to find frequently bought together products.
- Connect to a live SQL database instead of static CSVs.

## Conclusion
This system successfully converts raw transaction data into meaningful customer-level insights that can help businesses understand customer behaviour, support data-driven decisions, and formulate targeted marketing strategies.
