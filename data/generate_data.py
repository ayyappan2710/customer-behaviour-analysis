import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_customer_data(num_records=10000):
    np.random.seed(42)
    random.seed(42)
    
    customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, 1001)]  # 1000 unique customers
    genders = ['Male', 'Female', 'Other']
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Sports', 'Books']
    subcategories_map = {
        'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Audio', 'Accessories'],
        'Clothing': ['Men', 'Women', 'Kids', 'Activewear', 'Footwear'],
        'Home & Kitchen': ['Furniture', 'Decor', 'Appliances', 'Cookware', 'Bedding'],
        'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrance', 'Bath & Body'],
        'Sports': ['Fitness Equipment', 'Outdoor Gear', 'Athletic Clothing', 'Footwear', 'Accessories'],
        'Books': ['Personal Development', 'Finance', 'College Books', 'Programming Books', 'Story Books']
    }
    payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Cash']
    
    data = []
    
    start_date = datetime(2025, 1, 1)
    
    for _ in range(num_records):
        cust_id = random.choice(customer_ids)
        age = random.randint(16, 75)
        
        # Ensure age consistency per customer roughly by linking age to customer ID hash
        # (A bit simplified here, but we'll try to keep one age/gender per customer)
        # Actually let's create a customer dict first
        pass
    
    customer_profiles = {}
    for cid in customer_ids:
        customer_profiles[cid] = {
            'Age': random.randint(18, 70),
            'Gender': random.choices(genders, weights=[48, 48, 4])[0]
        }
        
    for i in range(num_records):
        cid = random.choice(customer_ids)
        age = customer_profiles[cid]['Age']
        gender = customer_profiles[cid]['Gender']
        
        category = random.choice(categories)
        
        # Some missing values randomly (5% chance)
        if random.random() < 0.05:
            category = np.nan
        
        quantity = random.randint(1, 5)
        
        price_ranges = {
            'Electronics': (50, 1500),
            'Clothing': (10, 200),
            'Home & Kitchen': (20, 500),
            'Beauty': (5, 100),
            'Sports': (15, 300),
            'Books': (5, 50)
        }
        
        if pd.notna(category):
            price = round(random.uniform(*price_ranges[category]), 2)
            subcategory = random.choice(subcategories_map[category])
        else:
            price = round(random.uniform(10, 500), 2)
            subcategory = np.nan
            
        # 5% chance missing price
        if random.random() < 0.05:
            price = np.nan
            
        date = start_date + timedelta(days=random.randint(0, 365), hours=random.randint(8, 22))
        payment = random.choice(payment_methods)
        
        data.append({
            'Customer ID': cid,
            'Age': age,
            'Gender': gender,
            'Product Category': category,
            'Product Subcategory': subcategory,
            'Quantity': quantity,
            'Price': price,
            'Purchase Date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'Payment Method': payment
        })
        
    df = pd.DataFrame(data)
    
    # Introduce some duplicates
    duplicates = df.sample(n=20)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'customer_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Dataset with {len(df)} records generated successfully.")

if __name__ == '__main__':
    generate_customer_data(10000)
