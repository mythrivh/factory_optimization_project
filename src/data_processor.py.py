import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Dataset
# Note: The dataset has no header in the provided snippet, so we define columns manually
columns = [
    'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 
    'Customer ID', 'Country/Region', 'City', 'State/Province', 'Postal Code', 
    'Division', 'Region', 'Product ID', 'Product Name', 'Sales', 'Units', 
    'Gross Profit', 'Cost'
]
df = pd.read_csv(r"C:\Users\mythr\OneDrive\factory_optimization_project\data\Nassau Candy Distributor (1).csv", header=None, names=columns)

# 2. Date Processing & Target Variable Creation
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')
df['Lead_Time_Days'] = (df['Ship Date'] - df['Order Date']).dt.days

# Clean invalid rows (negative lead times or missing dates)
df = df[df['Lead_Time_Days'] >= 0].dropna(subset=['Lead_Time_Days', 'Product Name'])

# 3. Merge with Factory Mapping (From your project requirements)
factory_mapping = pd.DataFrame({
    'Product Name': [
        'Wonka Bar - Nutty Crunch Surprise', 'Wonka Bar - Fudge Mallows', 
        'Wonka Bar -Scrumdiddlyumptious', 'Wonka Bar - Milk Chocolate', 
        'Wonka Bar - Triple Dazzle Caramel', 'Laffy Taffy', 'SweeTARTS', 
        'Nerds', 'Fun Dip', 'Fizzy Lifting Drinks', 'Everlasting Gobstopper', 
        'Hair Toffee', 'Lickable Wallpaper', 'Wonka Gum', 'Kazookles'
    ],
    'Factory': [
        "Lot's O' Nuts", "Lot's O' Nuts", "Lot's O' Nuts", "Wicked Choccy's", "Wicked Choccy's",
        'Sugar Shack', 'Sugar Shack', 'Sugar Shack', 'Sugar Shack', 'Sugar Shack',
        'Secret Factory', 'The Other Factory', 'Secret Factory', 'Secret Factory', 'The Other Factory'
    ]
})
df = df.merge(factory_mapping, on='Product Name', how='left')

# Fill any missing factories with 'Unknown'
df['Factory'] = df['Factory'].fillna('Unknown')

# 4. Feature Engineering for Modeling
features = ['Product Name', 'Factory', 'Region', 'Ship Mode', 'Division', 'Sales', 'Units', 'Cost']
X = df[features]
y = df['Lead_Time_Days']

# One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=['Product Name', 'Factory', 'Region', 'Ship Mode', 'Division'], drop_first=True)

# Align columns for future predictions
feature_columns = X_encoded.columns.tolist()

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Model Trained! MAE: {mean_absolute_error(y_test, preds):.2f} days | R2: {r2_score(y_test, preds):.4f}")

# 6. Save Artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Save a clean version of the data for the Streamlit app
df.to_csv('cleaned_data.csv', index=False)
print("Artifacts saved: model.pkl, feature_columns.pkl, cleaned_data.csv")