"""
ml_prediction.py — Binary classification: will a customer purchase again?

TARGET VARIABLE (proxy — clearly documented in UI):
  A customer is labelled "High Activity" (1) if their purchase frequency
  is above the median for the entire customer base; otherwise "Low Activity" (0).
  This is a purely historical behavioural label — it does NOT use future data.

FEATURES used:
  - Age
  - Gender (encoded)
  - Preferred_Category (encoded)
  - Preferred_Payment (encoded)
  - Recency (days since last purchase)
  - Frequency (total orders)
  - Monetary (total spending)
  - Avg_Order_Value
  - Total_Quantity

MODEL: Random Forest Classifier
SPLIT : 80 / 20 stratified train-test split
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from src.utils import safe_empty, build_rfm


# ──────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_ml_features(df: pd.DataFrame):
    """
    Build the feature matrix X and binary target y from the transaction DataFrame.

    Returns
    -------
    X_df  : feature DataFrame (before scaling) — used for display
    X     : numpy array ready for sklearn
    y     : numpy array (0 / 1)
    feature_names : list[str]
    encoders : dict of fitted LabelEncoders (for later single-customer prediction)
    scaler   : fitted StandardScaler
    """
    if safe_empty(df):
        return None, None, None, [], {}, None

    rfm = build_rfm(df)
    if rfm.empty:
        return None, None, None, [], {}, None

    # Also bring in preferred payment method
    if 'Payment Method' in df.columns:
        pay = (df.groupby(['Customer ID', 'Payment Method'])
                 .size()
                 .reset_index(name='cnt')
                 .sort_values('cnt', ascending=False)
                 .drop_duplicates('Customer ID')
                 [['Customer ID', 'Payment Method']]
                 .rename(columns={'Payment Method': 'Preferred_Payment'}))
        rfm = rfm.merge(pay, on='Customer ID', how='left')

    # ── Target: high activity = frequency > median ──
    median_freq = rfm['Frequency'].median()
    rfm['Target'] = (rfm['Frequency'] > median_freq).astype(int)

    feature_cols = ['Age', 'Recency', 'Frequency', 'Monetary',
                    'Avg_Order_Value', 'Total_Quantity']

    cat_cols = []
    if 'Gender' in rfm.columns:
        cat_cols.append('Gender')
    if 'Preferred_Category' in rfm.columns:
        cat_cols.append('Preferred_Category')
    if 'Preferred_Payment' in rfm.columns:
        cat_cols.append('Preferred_Payment')

    rfm = rfm.dropna(subset=feature_cols + cat_cols + ['Target'])

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        rfm[col + '_enc'] = le.fit_transform(rfm[col].astype(str))
        encoders[col] = le

    all_features = feature_cols + [c + '_enc' for c in cat_cols]
    X_df = rfm[all_features].copy()
    y = rfm['Target'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    return X_df, X, y, all_features, encoders, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier and evaluate on the test split.

    Returns
    -------
    model, scaler, encoders, metrics, feature_names, rfm_df
    """
    X_df, X, y, feature_names, encoders, scaler = prepare_ml_features(df)

    if X is None or len(np.unique(y)) < 2:
        return None, None, {}, [], {}, pd.DataFrame(), None

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'F1 Score': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'Confusion_Matrix': confusion_matrix(y_test, y_pred).tolist(),
        'Test_Size': len(y_test),
        'Train_Size': len(y_train),
    }

    # Feature importances
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=False)

    metrics['Feature_Importances'] = importances

    return model, scaler, encoders, metrics, feature_names, X_df, importances


# ──────────────────────────────────────────────────────────────────────────────
# Single customer prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_customer(
    customer_id: str,
    df: pd.DataFrame,
    model,
    scaler,
    encoders: dict,
    feature_names: list,
) -> dict:
    """
    Predict purchase probability for a single customer.

    Returns dict with 'probability', 'prediction', 'label'.
    """
    if model is None or safe_empty(df):
        return {}

    from src.utils import build_rfm
    rfm = build_rfm(df)

    # Add preferred payment
    if 'Payment Method' in df.columns:
        pay = (df.groupby(['Customer ID', 'Payment Method'])
                 .size()
                 .reset_index(name='cnt')
                 .sort_values('cnt', ascending=False)
                 .drop_duplicates('Customer ID')
                 [['Customer ID', 'Payment Method']]
                 .rename(columns={'Payment Method': 'Preferred_Payment'}))
        rfm = rfm.merge(pay, on='Customer ID', how='left')

    row = rfm[rfm['Customer ID'] == customer_id]
    if row.empty:
        return {'error': f"Customer '{customer_id}' not found in dataset."}

    row = row.iloc[0]
    base_features = ['Age', 'Recency', 'Frequency', 'Monetary',
                     'Avg_Order_Value', 'Total_Quantity']
    X_row = {f: row.get(f, 0) for f in base_features}

    for col, le in encoders.items():
        val = str(row.get(col, ''))
        try:
            X_row[col + '_enc'] = le.transform([val])[0]
        except ValueError:
            X_row[col + '_enc'] = 0  # unseen label → 0

    X_vec = np.array([[X_row[f] for f in feature_names]])
    X_scaled = scaler.transform(X_vec)

    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(model.predict(X_scaled)[0])

    return {
        'customer_id': customer_id,
        'probability': round(prob * 100, 1),
        'prediction': pred,
        'label': 'Likely to Purchase Again' if pred == 1 else 'Less Likely to Purchase',
        'segment_label': 'High Activity' if pred == 1 else 'Low Activity',
        'frequency': int(row.get('Frequency', 0)),
        'total_spending': round(row.get('Monetary', 0), 2),
        'recency_days': int(row.get('Recency', 0)),
    }
