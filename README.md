# CUSTOMER BEHAVIOUR ANALYSIS USING PYTHON

## Project Overview
Businesses collect large amounts of customer purchase data, but raw transaction data alone does not clearly reveal customer preferences, spending patterns, purchasing frequency, or customer groups. Therefore, a data-driven system is required to analyze customer behaviour, identify meaningful customer segments, and generate actionable insights.

## Objective
Create a complete customer behaviour analysis system that analyzes customer purchasing data and provides meaningful business insights. The system identifies product preferences, high/low spending customers, demographic purchasing behaviour, and customer segments using K-Means clustering.

## Features
- **Data Preprocessing Pipeline:** Cleans dataset, handles missing/duplicate values, and engineers features (e.g., Total Amount, Age Groups).
- **Exploratory Data Analysis:** Analyzes age, gender, category preferences, and purchase frequency.
- **Machine Learning Segmentation:** Uses Scikit-learn's K-Means clustering to dynamically segment customers based on behaviour.
- **Visualizations:** Professional charts using Matplotlib and Seaborn.
- **Dashboard:** Interactive Streamlit web application.
- **Actionable Insights:** Rule-based recommendation module.

## Technologies Used
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

## Project Structure
```text
customer-behaviour-analysis/
├── data/
│   ├── customer_data.csv
│   └── generate_data.py
├── notebooks/
│   └── customer_behaviour_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── analysis.py
│   ├── visualization.py
│   ├── segmentation.py
│   └── recommendations.py
├── dashboard/
│   └── app.py
├── outputs/
│   ├── charts/
│   └── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation & How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate Synthetic Data:** (If no dataset is provided)
   ```bash
   cd data
   python generate_data.py
   cd ..
   ```
3. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook notebooks/customer_behaviour_analysis.ipynb
   ```
4. **Run Streamlit Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

## Machine Learning Approach
- **K-Means Clustering:** Used for grouping customers based on Total Spending, Purchase Frequency, and Average Order Value.
- **Elbow Method:** Used to find the optimal number of clusters.
- **Note:** K-Means does *not* predict future purchases; it segments customers into behavioural groups (e.g., High-Value, Regular, Low-Value) to support targeted marketing.

## Future Enhancements
- Integrate a real-time database.
- Build predictive models to forecast churn rate.
- Add product-level collaborative filtering recommendations.
