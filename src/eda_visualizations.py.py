import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading and preprocessing data...")

# Define columns based on your dataset structure
columns = [
    'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 
    'Customer ID', 'Country/Region', 'City', 'State/Province', 'Postal Code', 
    'Division', 'Region', 'Product ID', 'Product Name', 'Sales', 'Units', 
    'Gross Profit', 'Cost'
]

# Load data (replace with your actual filename)
df = pd.read_csv('Nassau Candy Distributor (1).csv', header=None, names=columns)

# Parse dates (DD-MM-YYYY format)
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')

# Calculate Lead Time in Days
df['Lead_Time_Days'] = (df['Ship Date'] - df['Order Date']).dt.days

# Drop rows with invalid dates or negative lead times
df = df[df['Lead_Time_Days'] >= 0].dropna(subset=['Lead_Time_Days', 'Product Name'])

# Merge with Factory Mapping (From project requirements)
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
df['Factory'] = df['Factory'].fillna('Unknown')

print(f"Data ready! Total valid rows: {len(df)}")

# ==========================================
# 2. VISUALIZATIONS FOR RESEARCH PAPER / DASHBOARD
# ==========================================

# --- Figure 1: Lead Time Distribution ---
fig1 = px.histogram(
    df, x='Lead_Time_Days', nbins=50, 
    title='Distribution of Shipping Lead Times (Days)',
    labels={'Lead_Time_Days': 'Lead Time (Days)', 'count': 'Number of Orders'},
    color_discrete_sequence=['#636EFA']
)
fig1.update_layout(template='plotly_white')
fig1.show()

# --- Figure 2: Average Lead Time by Region & Ship Mode ---
fig2 = px.bar(
    df.groupby(['Region', 'Ship Mode'])['Lead_Time_Days'].mean().reset_index(),
    x='Region', y='Lead_Time_Days', color='Ship Mode', barmode='group',
    title='Average Lead Time by Region and Shipping Mode',
    labels={'Lead_Time_Days': 'Avg Lead Time (Days)', 'Region': 'Customer Region'},
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig2.update_layout(template='plotly_white')
fig2.show()

# --- Figure 3: Sales vs. Gross Profit by Division ---
division_metrics = df.groupby('Division').agg({
    'Sales': 'sum',
    'Gross Profit': 'sum',
    'Units': 'sum'
}).reset_index()

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(
    go.Bar(x=division_metrics['Division'], y=division_metrics['Sales'], name='Total Sales ($)', marker_color='#00CC96'),
    secondary_y=False
)
fig3.add_trace(
    go.Scatter(x=division_metrics['Division'], y=division_metrics['Gross Profit'], name='Gross Profit ($)', mode='lines+markers', line=dict(color='#FFA15A', width=3)),
    secondary_y=True
)
fig3.update_layout(
    title='Financial Performance by Product Division',
    template='plotly_white',
    yaxis_title='Total Sales ($)',
    yaxis2_title='Gross Profit ($)'
)
fig3.show()

# --- Figure 4: Top 10 Products with Highest Average Lead Time ---
top_slow_products = df.groupby('Product Name')['Lead_Time_Days'].mean().reset_index()
top_slow_products = top_slow_products.sort_values('Lead_Time_Days', ascending=False).head(10)

fig4 = px.bar(
    top_slow_products, x='Lead_Time_Days', y='Product Name', orientation='h',
    title='Top 10 Products with Highest Average Lead Time',
    labels={'Lead_Time_Days': 'Avg Lead Time (Days)', 'Product Name': 'Product'},
    color='Lead_Time_Days', color_continuous_scale='Reds'
)
fig4.update_layout(template='plotly_white')
fig4.show()

# --- Figure 5: Current Factory Workload Distribution ---
factory_workload = df['Factory'].value_counts().reset_index()
factory_workload.columns = ['Factory', 'Order Count']

fig5 = px.pie(
    factory_workload, values='Order Count', names='Factory',
    title='Current Order Volume Distribution Across Factories',
    color_discrete_sequence=px.colors.sequential.Blues_r,
    hole=0.4
)
fig5.update_traces(textposition='inside', textinfo='percent+label')
fig5.update_layout(template='plotly_white')
fig5.show()

# ==========================================
# 3. SAVE FIGURES (Optional, for Research Paper)
# ==========================================
# fig1.write_image("fig1_lead_time_distribution.png")
# fig2.write_image("fig2_lead_time_by_region.png")
# fig3.write_image("fig3_financial_performance.png")
# fig4.write_image("fig4_slowest_products.png")
# fig5.write_image("fig5_factory_workload.png")
print("Visualizations generated successfully!")