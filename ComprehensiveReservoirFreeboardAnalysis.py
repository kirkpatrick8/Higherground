import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set page config
st.set_page_config(page_title="Reservoir Freeboard Analysis", layout="wide")

# Title and creator information
st.title("Comprehensive Reservoir Freeboard Analysis")
st.write("Created by Mark Kirkpatrick (mark.kirkpatrick@aecom.com)")

# Constants
CATEGORY_A_MIN_FREEBOARD = 0.6  # meters
WIND_SPEED = 15  # m/s
PUMP_FAILURE_INFLOW = 0.75e6  # 0.75Mm3 per hour
PUMP_FAILURE_DURATION = 1  # 1 hour
OUTAGE_DURATION = 24 * 7  # 1 week in hours

# Load data
@st.cache_data
def load_data():
    reservoir_data = pd.read_csv("Reservoir_data.csv")
    return_periods = pd.read_csv("extended_specific_return_periods.csv")
    return reservoir_data, return_periods

reservoir_data, return_periods = load_data()

# Display the first few rows of each dataset
st.subheader("Reservoir Data Preview")
st.write(reservoir_data.head())

st.subheader("Return Period Data Preview")
st.write(return_periods.head())

# Process Embankment Slopes
reservoir_data['Embankment_Slopes_Numeric'] = reservoir_data['Embankment Slopes'].apply(lambda x: float(x.split('h:')[0]))

# Functions
def calculate_wave_surcharge(fetch, wind_speed, dam_type, slope):
    g = 9.81  # acceleration due to gravity (m/s^2)
    Hs = 0.0016 * (wind_speed**2 / g) * np.sqrt(fetch / g)
    HD = Hs * 1.2  # Using the "Surfaced road" factor
    Rf = 2.2 if slope == 1/2 else 1.75 if slope == 1/3 else 2.0
    wave_surcharge = Rf * HD
    wave_surcharge = max(wave_surcharge, CATEGORY_A_MIN_FREEBOARD)
    safety_margin = 0.2  # meters
    total_freeboard = wave_surcharge + safety_margin
    return total_freeboard

def calculate_normal_operation(row, wind_speed, rainfall, year):
    top_level = row["Top Water Level (m)"]
    surface_area = row["Water Surface Area (m2)"]
    slope = 1 / row["Embankment_Slopes_Numeric"]
    fetch = np.sqrt(surface_area)
    wave_surcharge = calculate_wave_surcharge(fetch, wind_speed, "Earthfill", slope)
    volume_increase = rainfall * surface_area / 1000  # Convert mm to m
    depth_increase = volume_increase / surface_area
    optimal_crest = top_level + wave_surcharge
    return pd.Series({
        'year': year,
        'optimal_crest': optimal_crest,
        'depth_increase': depth_increase
    })

def calculate_pump_failure(row):
    top_level = row["Top Water Level (m)"]
    surface_area = row["Water Surface Area (m2)"]
    volume_increase = PUMP_FAILURE_INFLOW * PUMP_FAILURE_DURATION
    depth_increase = volume_increase / surface_area
    optimal_crest = top_level + depth_increase + CATEGORY_A_MIN_FREEBOARD
    return pd.Series({
        'optimal_crest': optimal_crest,
        'depth_increase': depth_increase
    })

def calculate_system_failure(row, rainfall_max):
    top_level = row["Top Water Level (m)"]
    surface_area = row["Water Surface Area (m2)"]
    volume_increase = rainfall_max * surface_area / 1000 * OUTAGE_DURATION  # Convert mm to m
    depth_increase = volume_increase / surface_area
    optimal_crest = top_level + depth_increase + CATEGORY_A_MIN_FREEBOARD
    return pd.Series({
        'optimal_crest': optimal_crest,
        'depth_increase': depth_increase
    })

# Main analysis function
@st.cache_data
def perform_analysis(reservoir_data, return_periods):
    results = []
    for _, row in reservoir_data.iterrows():
        row_results = {'Option': row['Option']}
        for _, rp_row in return_periods.iterrows():
            normal_op = calculate_normal_operation(row, WIND_SPEED, rp_row['Net Rainfall (mm)'], rp_row['Year'])
            row_results[f'Normal_Operation_{rp_row["Year"]}yr_optimal_crest'] = normal_op['optimal_crest']
            row_results[f'Normal_Operation_{rp_row["Year"]}yr_depth_increase'] = normal_op['depth_increase']
        
        pump_failure = calculate_pump_failure(row)
        row_results['Pump_Failure_optimal_crest'] = pump_failure['optimal_crest']
        row_results['Pump_Failure_depth_increase'] = pump_failure['depth_increase']
        
        system_failure = calculate_system_failure(row, return_periods['Net Rainfall (mm)'].max())
        row_results['System_Failure_optimal_crest'] = system_failure['optimal_crest']
        row_results['System_Failure_depth_increase'] = system_failure['depth_increase']
        
        results.append(row_results)
    
    return pd.DataFrame(results)

# Perform analysis
results = perform_analysis(reservoir_data, return_periods)

# Interactive Dashboard
st.header("Interactive Dashboard")

# Scenario selection
scenario = st.selectbox(
    "Choose scenario to visualize:",
    ["Normal Operation"] + [f"Normal Operation ({year}yr)" for year in return_periods['Year']] + 
    ["Pump Failure", "System Failure"]
)

# Option selection
selected_options = st.multiselect(
    "Select reservoir options to compare:",
    options=results['Option'].unique(),
    default=results['Option'].unique()[:5]  # Default to first 5 options
)

# Filter data based on selection
filtered_results = results[results['Option'].isin(selected_options)]

# Prepare data for visualization
if "Normal Operation" in scenario:
    if scenario == "Normal Operation":
        columns_to_plot = [col for col in filtered_results.columns if col.startswith('Normal_Operation_') and col.endswith('_optimal_crest')]
        melted_data = filtered_results.melt(
            id_vars=['Option'],
            value_vars=columns_to_plot,
            var_name='Return Period',
            value_name='Water Level'
        )
        melted_data['Return Period'] = melted_data['Return Period'].str.extract('(\d+)').astype(int)
        fig = px.line(melted_data, x='Return Period', y='Water Level', color='Option', markers=True)
        fig.update_layout(title="Normal Operation - Water Level vs Return Period", 
                          xaxis_title="Return Period (years)", 
                          yaxis_title="Water Level (m)")
    else:
        year = scenario.split('(')[1].split('yr')[0]
        column_to_plot = f'Normal_Operation_{year}yr_optimal_crest'
        fig = px.bar(filtered_results, x='Option', y=column_to_plot, color='Option')
        fig.update_layout(title=f"Normal Operation ({year}-year) - Water Level by Option", 
                          xaxis_title="Option", 
                          yaxis_title="Water Level (m)")
elif scenario == "Pump Failure":
    fig = px.bar(filtered_results, x='Option', y='Pump_Failure_optimal_crest', color='Option')
    fig.update_layout(title="Pump Failure - Water Level by Option", 
                      xaxis_title="Option", 
                      yaxis_title="Water Level (m)")
else:  # System Failure
    fig = px.bar(filtered_results, x='Option', y='System_Failure_optimal_crest', color='Option')
    fig.update_layout(title="System Failure - Water Level by Option", 
                      xaxis_title="Option", 
                      yaxis_title="Water Level (m)")

st.plotly_chart(fig)

# Cost Analysis
st.subheader("Cost Comparison")
cost_fig = px.scatter(filtered_results, x='Storage Volume (m3)', y='Cost', color='Option', 
                      size='Max Embankment Height (m)',
                      hover_data=['Option', 'Cost', 'Storage Volume (m3)', 'Max Embankment Height (m)'])
cost_fig.update_layout(title="Cost vs Storage Volume", 
                       xaxis_title="Storage Volume (m³)", 
                       yaxis_title="Cost")
st.plotly_chart(cost_fig)

# Freeboard Margin Comparison
st.subheader("Freeboard Margin Comparison")
filtered_results['Max_Water_Level'] = filtered_results[[col for col in filtered_results.columns if col.endswith('_optimal_crest')]].max(axis=1)
filtered_results['Freeboard_Margin'] = filtered_results['Max_Water_Level'] - filtered_results['Top Water Level (m)']

freeboard_fig = px.bar(filtered_results, x='Option', y='Freeboard_Margin', color='Option')
freeboard_fig.update_layout(title="Freeboard Margin by Option", 
                            xaxis_title="Option", 
                            yaxis_title="Freeboard Margin (m)")
st.plotly_chart(freeboard_fig)

# Key Metrics Table
st.subheader("Key Metrics Table")
metrics_cols = ['Option', 'Cost', 'Storage Volume (m3)', 'Max Embankment Height (m)', 'Freeboard_Margin']
st.dataframe(filtered_results[metrics_cols].set_index('Option'))

# Download results
st.subheader("Download Results")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(results)
st.download_button(
    label="Download full results as CSV",
    data=csv,
    file_name="reservoir_analysis_results.csv",
    mime="text/csv",
)

# Conclusion
st.header("Conclusion and Recommendations")
st.write("""
Based on the analysis performed, consider the following recommendations:

1. Prioritize options with larger freeboard margins for better safety against overtopping.
2. Balance cost and storage volume when comparing options.
3. Consider options that maintain adequate freeboard under various conditions.
4. Conduct more detailed studies for options that perform well in this analysis.
5. Ensure compliance with relevant local and national regulations for dam safety.
6. Consider ease of operation and maintenance for long-term costs and safety.

Remember that this analysis provides a high-level comparison. Final decision-making should involve a multidisciplinary team and consider additional factors not covered here.
""")

st.write("Analysis complete. Thank you for using the Comprehensive Reservoir Freeboard Analysis tool.")
