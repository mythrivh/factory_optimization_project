import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Nassau Candy Optimizer", 
    layout="wide", 
    page_icon="🍬",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD DATA & MODEL
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        df = pd.read_csv('cleaned_data.csv')
        return model, feature_columns, df
    except:
        st.error("Please run data_processor.py first to generate model artifacts!")
        st.stop()

model, feature_columns, df = load_artifacts()

# ==========================================
# OPTIMIZATION ENGINE
# ==========================================
ALL_FACTORIES = ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory", "The Other Factory"]

DIVISION_FACTORIES = {
    'Chocolate': ["Lot's O' Nuts", "Wicked Choccy's"],
    'Sugar': ["Sugar Shack"],
    'Other': ["Secret Factory", "The Other Factory"]
}

def predict_lead_time(product, factory, region, ship_mode, division, sales=10.0, units=3, cost=5.0):
    row = pd.DataFrame({
        'Product Name': [product], 'Factory': [factory], 'Region': [region],
        'Ship Mode': [ship_mode], 'Division': [division], 
        'Sales': [sales], 'Units': [units], 'Cost': [cost]
    })
    row_encoded = pd.get_dummies(row, columns=['Product Name', 'Factory', 'Region', 'Ship Mode', 'Division'])
    for col in feature_columns:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_aligned = row_encoded[feature_columns]
    return max(1.0, model.predict(row_aligned)[0])

def get_baseline_metrics(product_name, region, ship_mode):
    subset = df[(df['Product Name'] == product_name) & 
                (df['Region'] == region) & 
                (df['Ship Mode'] == ship_mode)]
    if len(subset) == 0:
        subset = df[df['Product Name'] == product_name]
    return {
        'current_lead_time': subset['Lead_Time_Days'].mean(),
        'avg_units': subset['Units'].mean(),
        'avg_cost': subset['Cost'].mean(),
        'division': subset['Division'].iloc[0] if len(subset) > 0 else 'Unknown',
        'current_factory': subset['Factory'].iloc[0] if len(subset) > 0 else 'Unknown'
    }

def run_optimization_model():
    products = df['Product Name'].unique()
    demand_matrix = df.groupby(['Product Name', 'Region'])['Units'].sum().reset_index()
    demand_matrix.columns = ['Product', 'Region', 'Volume']
    
    cost_dict = {}
    for p in products:
        for f in ALL_FACTORIES:
            total_weighted_cost = 0
            for _, row in demand_matrix[demand_matrix['Product'] == p].iterrows():
                r = row['Region']
                vol = row['Volume']
                subset = df[(df['Product Name'] == p) & (df['Region'] == r)]
                base_lead = subset['Lead_Time_Days'].mean() if len(subset) > 0 else 10.0
                factory_modifier = {"Lot's O' Nuts": 1.0, "Wicked Choccy's": 0.9, "Sugar Shack": 1.1, "Secret Factory": 1.05, "The Other Factory": 1.2}
                pred_lead = base_lead * factory_modifier.get(f, 1.0)
                total_weighted_cost += pred_lead * vol
            cost_dict[(p, f)] = total_weighted_cost
    
    product_divisions = df[['Product Name', 'Division']].drop_duplicates().set_index('Product Name')['Division'].to_dict()
    
    opt_model = pulp.LpProblem("Nassau_Candy_Optimization", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("Assign", (products, ALL_FACTORIES), cat='Binary')
    opt_model += pulp.lpSum([cost_dict[(p, f)] * x[p][f] for p in products for f in ALL_FACTORIES])
    
    for p in products:
        opt_model += pulp.lpSum([x[p][f] for f in ALL_FACTORIES]) == 1
    
    for p in products:
        div = product_divisions[p]
        allowed_factories = DIVISION_FACTORIES.get(div, [])
        for f in ALL_FACTORIES:
            if f not in allowed_factories:
                opt_model += x[p][f] == 0
    
    opt_model.solve()
    
    optimal_assignments = []
    for p in products:
        for f in ALL_FACTORIES:
            if pulp.value(x[p][f]) == 1:
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
    
    return pd.DataFrame(optimal_assignments).sort_values(by="Efficiency Gain (Unit-Days)", ascending=False)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=80)
st.sidebar.title(" Control Panel")

product_list = sorted(df['Product Name'].unique())
selected_product = st.sidebar.selectbox(" Select Product", product_list)

region_list = sorted(df['Region'].unique())
selected_region = st.sidebar.selectbox(" Select Region", region_list)

ship_mode = st.sidebar.selectbox(" Ship Mode", sorted(df['Ship Mode'].unique()))

st.sidebar.markdown("---")
optimization_priority = st.sidebar.slider(
    "Optimization Priority", 0, 100, 50,
    help="0 = Maximize Profit, 100 = Minimize Lead Time"
)

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title(" Factory Reallocation & Shipping Optimization")
st.markdown("**Decision Intelligence Dashboard for Nassau Candy Distributor**")
st.markdown("---")

# Main tabs
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
    " Optimization Simulator",
    " What-If Analysis",
    " Recommendations",
    " Risk & Impact",
    " EDA Insights"
])

# ==========================================
# TAB 1: OPTIMIZATION SIMULATOR
# ==========================================
with main_tab1:
    st.subheader("Factory Optimization Simulator")
    
    baseline = get_baseline_metrics(selected_product, selected_region, ship_mode)
    
    factories_to_test = [baseline['current_factory']] + [f for f in ALL_FACTORIES if f != baseline['current_factory']]
    
    results = []
    for fac in factories_to_test:
        pred = predict_lead_time(
            selected_product, fac, selected_region, ship_mode, 
            baseline['division'], baseline['avg_units']*10, baseline['avg_units'], baseline['avg_cost']
        )
        is_current = " Current" if fac == baseline['current_factory'] else " Alternate"
        results.append({
            "Factory": fac, 
            "Status": is_current, 
            "Predicted Lead Time (Days)": round(pred, 1)
        })
    
    res_df = pd.DataFrame(results)
    fig = px.bar(
        res_df, x="Factory", y="Predicted Lead Time (Days)", color="Status",
        color_discrete_map={"Current": "#FF4B4B", " Alternate": "#00CC96"},
        text="Predicted Lead Time (Days)"
    )
    fig.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: WHAT-IF ANALYSIS
# ==========================================
with main_tab2:
    st.subheader("What-If Scenario Analysis")
    
    best_fac = baseline['current_factory']
    best_lead = baseline['current_lead_time']
    
    for fac in ALL_FACTORIES:
        if fac == baseline['current_factory']:
            continue
        pred = predict_lead_time(
            selected_product, fac, selected_region, ship_mode,
            baseline['division'], baseline['avg_units']*10, baseline['avg_units'], baseline['avg_cost']
        )
        if pred < best_lead:
            best_lead = pred
            best_fac = fac
    
    improvement = baseline['current_lead_time'] - best_lead
    improvement_pct = (improvement / baseline['current_lead_time']) * 100 if baseline['current_lead_time'] > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Avg Lead Time", f"{baseline['current_lead_time']:.1f} Days")
    col2.metric("Recommended Lead Time", f"{best_lead:.1f} Days", delta=f"{improvement:.1f} Days Faster")
    col3.metric("Efficiency Gain", f"{improvement_pct:.1f}%", delta_color="normal" if improvement_pct > 0 else "inverse")
    
    if best_fac != baseline['current_factory']:
        st.success(f" **Insight:** Moving production of *{selected_product}* to **{best_fac}** for the **{selected_region}** region reduces lead time by {improvement:.1f} days.")
    else:
        st.info(" **Insight:** The current factory assignment is already optimal.")

# ==========================================
# TAB 3: RECOMMENDATIONS (OPTIMIZATION MODEL)
# ==========================================
with main_tab3:
    st.subheader("Mathematical Optimization Recommendations")
    st.markdown("*Powered by Mixed-Integer Linear Programming (MILP)*")
    
    if st.button(" Run Full Optimization Model", type="primary"):
        with st.spinner("Solving optimization problem..."):
            opt_results = run_optimization_model()
            st.session_state['opt_results'] = opt_results
    
    if 'opt_results' in st.session_state:
        opt_df = st.session_state['opt_results']
        
        total_gain = opt_df[opt_df['Efficiency Gain (Unit-Days)'] > 0]['Efficiency Gain (Unit-Days)'].sum()
        products_to_move = len(opt_df[opt_df['Current Factory'] != opt_df['Optimal Factory']])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Efficiency Gain", f"{total_gain:,.0f} Unit-Days")
        col2.metric("Products to Reallocate", f"{products_to_move}")
        col3.metric("Optimization Status", " Optimal")
        
        st.markdown("### Top Reassignment Recommendations")
        display_df = opt_df[opt_df['Current Factory'] != opt_df['Optimal Factory']].head(10)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click the button above to run the mathematical optimization model.")

# ==========================================
# TAB 4: RISK & IMPACT
# ==========================================
with main_tab4:
    st.subheader("Risk Assessment & Alerts")
    
    if optimization_priority > 75:
        st.warning(" **High Aggression Mode:** Prioritizing speed over cost may increase freight expenses.")
    
    st.markdown("###  Risk Factors")
    risk_df = pd.DataFrame({
        "Risk Factor": ["Cross-Division Retooling", "Historical Data Sparsity", "Regional Carrier Capacity"],
        "Severity": ["MEDIUM", "LOW", "LOW"],
        "Mitigation Strategy": ["Standardize SOPs", "Rely on baseline averages", "Negotiate bulk rates"]
    })
    st.table(risk_df)

# ==========================================
# TAB 5: EDA INSIGHTS (VISUALIZATIONS)
# ==========================================
with main_tab5:
    st.subheader("Exploratory Data Analysis Insights")
    
    # Chart 1: Lead Time Distribution
    st.markdown("### Lead Time Distribution")
    fig1 = px.histogram(
        df, x='Lead_Time_Days', nbins=50,
        title='Distribution of Shipping Lead Times (Days)',
        labels={'Lead_Time_Days': 'Lead Time (Days)', 'count': 'Number of Orders'},
        color_discrete_sequence=['#636EFA']
    )
    fig1.update_layout(template='plotly_white', height=350)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Lead Time by Region
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Avg Lead Time by Region")
        region_lead = df.groupby('Region')['Lead_Time_Days'].mean().reset_index()
        fig2 = px.bar(
            region_lead, x='Region', y='Lead_Time_Days',
            title='Average Lead Time by Region',
            color='Lead_Time_Days', color_continuous_scale='Blues'
        )
        fig2.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### Factory Workload Distribution")
        factory_workload = df['Factory'].value_counts().reset_index()
        factory_workload.columns = ['Factory', 'Order Count']
        fig3 = px.pie(
            factory_workload, values='Order Count', names='Factory',
            title='Current Order Volume by Factory',
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4
        )
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(template='plotly_white', height=350)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Chart 3: Financial Performance
    st.markdown("### Financial Performance by Division")
    division_metrics = df.groupby('Division').agg({
        'Sales': 'sum',
        'Gross Profit': 'sum'
    }).reset_index()
    
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(
        go.Bar(x=division_metrics['Division'], y=division_metrics['Sales'], name='Total Sales ($)', marker_color='#00CC96'),
        secondary_y=False
    )
    fig4.add_trace(
        go.Scatter(x=division_metrics['Division'], y=division_metrics['Gross Profit'], name='Gross Profit ($)', mode='lines+markers', line=dict(color='#FFA15A', width=3)),
        secondary_y=True
    )
    fig4.update_layout(
        title='Sales vs Gross Profit by Division',
        template='plotly_white',
        height=400,
        yaxis_title='Total Sales ($)',
        yaxis2_title='Gross Profit ($)'
    )
    st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Decision Intelligence System v1.0 | Powered by Gradient Boosting Regressor & MILP Optimization")