# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as plt_sns
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

def plot_sales_by_category(df, save_dir=None):
    if df is None or df.empty: return
    plt.figure(figsize=(10, 6))
    category_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
    sns.barplot(x=category_sales.values, y=category_sales.index, palette='viridis')
    plt.title('Total Sales by Product Category')
    plt.xlabel('Total Sales ($)')
    plt.ylabel('Category')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sales_by_category.png'), bbox_inches='tight')
    plt.close()

def plot_gender_purchases(df, save_dir=None):
    if df is None or df.empty: return
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Gender', palette='pastel')
    plt.title('Number of Purchases by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'gender_purchases.png'), bbox_inches='tight')
    plt.close()

def plot_age_distribution(unique_customers, save_dir=None):
    if unique_customers is None or unique_customers.empty: return
    plt.figure(figsize=(10, 6))
    sns.histplot(unique_customers['Age'], bins=15, kde=True, color='skyblue')
    plt.title('Age Distribution of Customers')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'age_distribution.png'), bbox_inches='tight')
    plt.close()

def plot_customer_segments(customer_summary, save_dir=None):
    if customer_summary is None or 'Customer Segment' not in customer_summary.columns: return
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=customer_summary, 
        x='Total_Spending', 
        y='Purchase_Frequency', 
        hue='Customer Segment',
        palette='Set1',
        s=100
    )
    plt.title('Customer Segmentation')
    plt.xlabel('Total Spending ($)')
    plt.ylabel('Purchase Frequency')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'customer_segments.png'), bbox_inches='tight')
    plt.close()

def generate_all_plots(df, customer_summary, save_dir='outputs/charts'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    unique_cust = df.drop_duplicates(subset=['Customer ID']) if df is not None else None
    
    plot_sales_by_category(df, save_dir)
    plot_gender_purchases(df, save_dir)
    plot_age_distribution(unique_cust, save_dir)
    plot_customer_segments(customer_summary, save_dir)
    print(f"All plots saved to {save_dir}")
