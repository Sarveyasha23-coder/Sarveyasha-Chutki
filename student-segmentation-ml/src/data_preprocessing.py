import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    
    # Ensure numeric types as requested in part (b) i
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'gradyear' in df.columns:
        df['gradyear'] = pd.to_numeric(df['gradyear'], errors='coerce')
    if 'numberoffriends' in df.columns:
        df['numberoffriends'] = pd.to_numeric(df['numberoffriends'], errors='coerce')
        
    return df

def get_skewness(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].skew().sort_values(ascending=False)

def preprocess_data(df):
    df = df.copy()
    
    # Handle missing values as per part (b) iv
    if 'gender' in df.columns:
        non_null_gender = df['gender'].dropna()
        if not non_null_gender.empty:
            df['gender'] = df['gender'].fillna(non_null_gender.mode()[0])
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].median())
        
    # Drop remaining NaNs if any exist in numeric features
    all_numeric = df.select_dtypes(include=['int64', 'float64']).columns
    # Exclude ID columns from features to prevent clustering on noise
    numeric_cols = [col for col in all_numeric if col.lower() not in ['userid', 'id', 'unnamed: 0', 'cluster']]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for preprocessing.")
    df = df.dropna(subset=numeric_cols).copy()

    # Treat Outliers using IQR Capping
    for col in numeric_cols:
        series = df[col]
        if series.nunique() <= 2: # Skip binary columns
            continue
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    X = df[numeric_cols]

    # Transform skewness
    pt = PowerTransformer(method='yeo-johnson')
    X_transformed = pt.fit_transform(X)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)

    return X_scaled, df, pt, scaler, numeric_cols
