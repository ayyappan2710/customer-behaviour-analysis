import pandas as pd

def load_data(file_path):
    """Load dataset and return pandas dataframe."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_exploration(df):
    """Print basic dataset information."""
    if df is None:
        return
    print("--- Dataset Shape ---")
    print(df.shape)
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Column Information ---")
    print(df.info())

def clean_data(df):
    """Clean the dataset: handle missing values, duplicates, and formats."""
    if df is None or df.empty:
        return None
    
    df_cleaned = df.copy()
    
    # 0. Validate required columns
    required_cols = ['Customer ID', 'Age', 'Gender', 'Product Category', 'Quantity', 'Price', 'Purchase Date', 'Payment Method']
    missing_required = [col for col in required_cols if col not in df_cleaned.columns]
    if missing_required:
        print(f"Error: Missing required columns: {missing_required}")
        return None

    # 1. Check and handle duplicates
    duplicates = df_cleaned.duplicated().sum()
    print(f"\nFound {duplicates} duplicate rows. Removing them...")
    df_cleaned = df_cleaned.drop_duplicates()
    
    # 2. Convert Purchase Date to datetime
    if 'Purchase Date' in df_cleaned.columns:
        df_cleaned['Purchase Date'] = pd.to_datetime(df_cleaned['Purchase Date'], errors='coerce')
        
    # 3. Ensure numeric columns are numeric
    for col in ['Quantity', 'Price', 'Total Amount']:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # 4. Handle missing values
    missing_cols = df_cleaned.isnull().sum()
    print("\n--- Missing Values Before Cleaning ---")
    print(missing_cols[missing_cols > 0])
    
    # Fill categorical with mode or 'Unknown'
    if 'Product Category' in df_cleaned.columns:
        df_cleaned['Product Category'] = df_cleaned['Product Category'].fillna('Unknown')
        
    # Fill numerical with median
    if 'Price' in df_cleaned.columns:
        df_cleaned['Price'] = df_cleaned['Price'].fillna(df_cleaned['Price'].median())
    if 'Quantity' in df_cleaned.columns:
        df_cleaned['Quantity'] = df_cleaned['Quantity'].fillna(df_cleaned['Quantity'].median())
        
    # Drop rows where critical info like Customer ID is missing
    if 'Customer ID' in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=['Customer ID'])
        
    print("\nData cleaning complete.")
    return df_cleaned

def feature_engineering(df):
    """Create derived columns for analysis."""
    if df is None or df.empty:
        return None

    df_features = df.copy()

    # Always calculate Total Amount from Quantity and Price
    if 'Quantity' in df_features.columns and 'Price' in df_features.columns:
        df_features['Total Amount'] = (
            df_features['Quantity'] * df_features['Price']
        )
    else:
        print("Warning: Cannot calculate Total Amount. Missing Quantity or Price.")

    # Time-based features
    if 'Purchase Date' in df_features.columns:
        df_features['Purchase Month'] = df_features['Purchase Date'].dt.month
        df_features['Purchase Year'] = df_features['Purchase Date'].dt.year

    # Age Groups
    if 'Age' in df_features.columns:
        bins = [0, 18, 25, 35, 50, 100]
        labels = ['Below 18', '18-25', '26-35', '36-50', '51+']
        df_features['Age Group'] = pd.cut(
            df_features['Age'],
            bins=bins,
            labels=labels,
            right=False
        )

    return df_features

def preprocess_pipeline(file_path):
    """Run the full preprocessing pipeline."""
    df = load_data(file_path)
    if df is not None:
        basic_exploration(df)
        df_clean = clean_data(df)
        df_final = feature_engineering(df_clean)
        return df_final
    return None
