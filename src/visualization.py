# =========================
# visualization.py
# =========================
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

def sankey_from_allocations(allocations):
    """
    Build a Sankey diagram from factory-region allocations.
    allocations: list of dicts with keys ["Factory", "Region", "Quantity"]
    """

    factories = list({a["Factory"] for a in allocations})
    regions = list({a["Region"] for a in allocations})
    nodes = factories + regions
    node_indices = {name: i for i, name in enumerate(nodes)}

    sources = [node_indices[a["Factory"]] for a in allocations]
    targets = [node_indices[a["Region"]] for a in allocations]
    values = [a["Quantity"] for a in allocations]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(title_text="Factory → Region Allocations", font_size=12)
    return fig


def demand_bar_chart(demand, unmet):
    """
    Bar chart comparing demand vs unmet demand per region.
    """
    regions = list(demand.keys())
    fulfilled = [demand[r] - unmet.get(r, 0) for r in regions]
    unmet_vals = [unmet.get(r, 0) for r in regions]

    fig = go.Figure()
    fig.add_bar(x=regions, y=fulfilled, name="Fulfilled Demand", marker_color="green")
    fig.add_bar(x=regions, y=unmet_vals, name="Unmet Demand", marker_color="red")

    fig.update_layout(barmode="stack", title="Demand Fulfillment by Region")
    return fig


def show_visualizations(allocations, demand, unmet):
    """
    Streamlit wrapper to show Sankey + demand chart + allocation table.
    """
    st.subheader(" Factory → Region Flows")
    sankey_fig = sankey_from_allocations(allocations)
    st.plotly_chart(sankey_fig, use_container_width=True)

    st.subheader(" Demand Coverage")
    demand_fig = demand_bar_chart(demand, unmet)
    st.plotly_chart(demand_fig, use_container_width=True)

    st.subheader("Allocation Table")
    st.dataframe(pd.DataFrame(allocations))
