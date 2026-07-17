import pulp
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA & DEFINE SETS
# ==========================================
df = pd.read_csv('cleaned_data.csv') # From your previous data processing step

products = df['Product Name'].unique()
factories = ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory", "The Other Factory"]
regions = df['Region'].unique()

# Division Compatibility Rules (Risk Constraints)
division_factory_map = {
    'Chocolate': ["Lot's O' Nuts", "Wicked Choccy's"],
    'Sugar': ["Sugar Shack"],
    'Other': ["Secret Factory", "The Other Factory"]
}

# Map products to their divisions
product_divisions = df[['Product Name', 'Division']].drop_duplicates().set_index('Product Name')['Division'].to_dict()

# ==========================================
# 2. CALCULATE DEMAND VOLUME (Weights)
# ==========================================
# How many units of each product are shipped to each region?
demand_matrix = df.groupby(['Product Name', 'Region'])['Units'].sum().reset_index()
demand_matrix.columns = ['Product', 'Region', 'Volume']

# ==========================================
# 3. CALCULATE COST MATRIX (Using ML Predictions)
# ==========================================
# Note: Replace this with your actual `predict_lead_time` function from app.py
def mock_predict_lead_time(product, factory, region):
    # In production, this calls your trained Gradient Boosting model
    # For demonstration, we simulate a prediction based on historical averages
    subset = df[(df['Product Name'] == product) & (df['Region'] == region)]
    if len(subset) > 0:
        base_lead = subset['Lead_Time_Days'].mean()
        # Simulate factory efficiency variance
        factory_modifier = {"Lot's O' Nuts": 1.0, "Wicked Choccy's": 0.9, "Sugar Shack": 1.1, "Secret Factory": 1.05, "The Other Factory": 1.2}
        return base_lead * factory_modifier.get(factory, 1.0)
    return 10.0 # Default fallback

# Calculate C_p_f: Total weighted lead time for assigning product p to factory f
cost_dict = {}
for p in products:
    for f in factories:
        total_weighted_cost = 0
        for _, row in demand_matrix[demand_matrix['Product'] == p].iterrows():
            r = row['Region']
            vol = row['Volume']
            pred_lead = mock_predict_lead_time(p, f, r)
            total_weighted_cost += pred_lead * vol
        cost_dict[(p, f)] = total_weighted_cost

# ==========================================
# 4. BUILD THE OPTIMIZATION MODEL (PuLP)
# ==========================================
model = pulp.LpProblem("Nassau_Candy_Factory_Optimization", pulp.LpMinimize)

# Decision Variables: x[p][f] = 1 if product p is assigned to factory f
x = pulp.LpVariable.dicts("Assign", (products, factories), cat='Binary')

# Objective Function: Minimize total weighted lead time
model += pulp.lpSum([cost_dict[(p, f)] * x[p][f] for p in products for f in factories])

# ==========================================
# 5. ADD CONSTRAINTS
# ==========================================
# Constraint 1: Each product assigned to exactly one factory
for p in products:
    model += pulp.lpSum([x[p][f] for f in factories]) == 1

# Constraint 2: Division Compatibility (Risk Mitigation)
for p in products:
    div = product_divisions[p]
    allowed_factories = division_factory_map.get(div, [])
    
    for f in factories:
        if f not in allowed_factories:
            model += x[p][f] == 0  # Forbid this assignment

# ==========================================
# 6. SOLVE & EXTRACT RESULTS
# ==========================================
# Solve the model (CBC solver is included with PuLP)
status = model.solve()

print(f"Optimization Status: {pulp.LpStatus[status]}")
print(f"Minimum Total Weighted Lead Time: {pulp.value(model.objective):,.2f} Unit-Days\n")

# Extract the optimal assignments
optimal_assignments = []
for p in products:
    for f in factories:
        if pulp.value(x[p][f]) == 1:
            # Calculate improvement vs current assignment
            current_factory = df[df['Product Name'] == p]['Factory'].iloc[0]
            current_cost = cost_dict.get((p, current_factory), 0)
            new_cost = cost_dict[(p, f)]
            improvement = current_cost - new_cost
            
            optimal_assignments.append({
                "Product Name": p,
                "Current Factory": current_factory,
                "Optimal Factory": f,
                "Current Weighted Lead Time": round(current_cost, 1),
                "Optimal Weighted Lead Time": round(new_cost, 1),
                "Efficiency Gain (Unit-Days)": round(improvement, 1)
            })

results_df = pd.DataFrame(optimal_assignments)
results_df = results_df.sort_values(by="Efficiency Gain (Unit-Days)", ascending=False)

# Display the final recommendations
print("--- TOP FACTORY REALLOCATION RECOMMENDATIONS ---")
print(results_df.to_string(index=False))

# Save to CSV for the Streamlit Dashboard
results_df.to_csv('optimization_recommendations.csv', index=False)