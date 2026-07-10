# =========================
# dashboard.py
# =========================
import streamlit as st
import pandas as pd
import pulp
import io

from src.data_prep import prepare_data
from src.model import build_supply_chain_model
from test_visualization import sankey_from_allocations

st.title("Factory Reallocation & Shipping Optimization")

# Example input data (replace with your real CSV-driven costs/demand/capacity later)
costs = pd.DataFrame({
    "North": [2, 4],
    "South": [3, 1],
}, index=["FactoryA", "FactoryB"])

demand = {"North": 80, "South": 60}
capacity = {"FactoryA": 100, "FactoryB": 80}

# Run optimization
model, allocations = build_supply_chain_model(costs, demand, capacity)

# Show results
st.write("Status:", pulp.LpStatus[model.status])
st.write("Objective value (Total Cost):", pulp.value(model.objective))
st.write("Allocations:", allocations)

# Sankey diagram
fig = sankey_from_allocations(allocations)
st.plotly_chart(fig, use_container_width=True)

# Export allocations to Excel
output = io.BytesIO()
pd.DataFrame(allocations).to_excel(output, index=False)
excel_data = output.getvalue()

if st.download_button(
    label=" Download Allocations (Excel)",
    data=excel_data,
    file_name="allocations.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
):
    st.success(" Download ready! Your Excel file has been saved.")
