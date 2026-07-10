import streamlit as st
import pulp
import plotly.graph_objects as go
import pandas as pd
import io

# --- Example data setup ---
regions = ["Interior", "Atlantic", "Gulf", "Pacific"]

capacity = {
    "Lot's O' Nuts": 5000,
    "Wicked Choccy's": 4000,
    "Sugar Shack": 4500,
    "Secret Factory": 3000,
    "The Other Factory": 3500
}

demand = {
    "Interior": 8820,
    "Atlantic": 11159,
    "Gulf": 6209,
    "Pacific": 12466
}

shipping_costs = {
    ("Lot's O' Nuts", "Interior"): 4,
    ("Lot's O' Nuts", "Atlantic"): 6,
    ("Wicked Choccy's", "Atlantic"): 5,
    ("Sugar Shack", "Interior"): 7,
    ("Sugar Shack", "Gulf"): 6,
    ("Secret Factory", "Pacific"): 8,
    ("The Other Factory", "Gulf"): 5,
    ("The Other Factory", "Pacific"): 7
}

# --- Optimization model ---
model = pulp.LpProblem("Factory_Reallocation", pulp.LpMinimize)

x = pulp.LpVariable.dicts("ship",
                          [(f, r) for f in capacity for r in regions],
                          lowBound=0,
                          cat="Integer")

slack = pulp.LpVariable.dicts("unmet", regions, lowBound=0, cat="Integer")

penalty = 10 * max(shipping_costs.values())

model += (
    pulp.lpSum(
        shipping_costs[(f, r)] * x[(f, r)]
        for f in capacity for r in regions if (f, r) in shipping_costs
    )
    + pulp.lpSum(penalty * slack[r] for r in regions)
)

for r in regions:
    model += (
        pulp.lpSum(x[(f, r)] for f in capacity if (f, r) in shipping_costs)
        + slack[r] >= demand[r]
    )

for f in capacity:
    model += (
        pulp.lpSum(x[(f, r)] for r in regions if (f, r) in shipping_costs)
        <= capacity[f]
    )

model.solve()

# --- Collect allocations ---
allocations = []
for (f, r) in x:
    val = x[(f, r)].value()
    if val is not None and val > 0:
        allocations.append({"Factory": f, "Region": r, "Quantity": int(val)})

unmet = {r: int(slack[r].value()) if slack[r].value() else 0 for r in regions}

# --- Streamlit Dashboard ---
st.title(" Factory Optimization Dashboard")
st.write("Status:", pulp.LpStatus[model.status])

for a in allocations:
    st.write(f"Ship {a['Quantity']} units from **{a['Factory']}** to **{a['Region']}**")

for r in regions:
    if unmet[r] > 0:
        st.warning(f"Unmet demand in {r}: {unmet[r]} units")

# Show total minimized cost
st.metric("Total Minimized Cost (incl. penalties)", pulp.value(model.objective))

# --- Visualizations ---
st.subheader(" Factory → Region Flows")
factories = list({a["Factory"] for a in allocations})
nodes = factories + regions
node_indices = {name: i for i, name in enumerate(nodes)}

sources = [node_indices[a["Factory"]] for a in allocations]
targets = [node_indices[a["Region"]] for a in allocations]
values = [a["Quantity"] for a in allocations]

sankey_fig = go.Figure(data=[go.Sankey(
    node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=nodes),
    link=dict(source=sources, target=targets, value=values)
)])
sankey_fig.update_layout(title_text="Factory → Region Allocations", font_size=12)
st.plotly_chart(sankey_fig, use_container_width=True)

st.subheader(" Demand Fulfillment")
fulfilled = [demand[r] - unmet[r] for r in regions]
unmet_vals = [unmet[r] for r in regions]

bar_fig = go.Figure()
bar_fig.add_bar(x=regions, y=fulfilled, name="Fulfilled Demand", marker_color="green")
bar_fig.add_bar(x=regions, y=unmet_vals, name="Unmet Demand", marker_color="red")
bar_fig.update_layout(barmode="stack", title="Demand Fulfillment by Region")
st.plotly_chart(bar_fig, use_container_width=True)

st.subheader(" Allocation Table")
st.dataframe(pd.DataFrame(allocations))

# --- Export Allocations ---
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
