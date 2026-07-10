
import pandas as pd
import numpy as np

def prepare_data(csv_path):
    df = pd.read_csv(r"C:\Users\mythr\OneDrive\factory_optimization_project\data\Nassau Candy Distributor (1).csv")

    # Handle missing values
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical variables
    X = df[["Product ID", "Division", "Region", "Ship Mode"]]
    X_encoded = pd.get_dummies(X)

    # Target: lead time in days
    y = (
        pd.to_datetime(df["Ship Date"], format="%d-%m-%Y")
        - pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    ).dt.days

    return df, X_encoded, y
