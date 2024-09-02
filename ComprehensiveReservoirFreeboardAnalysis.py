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

# Script information
st.info("""
This script performs a comprehensive analysis of reservoir freeboard for various scenarios and return periods.
It calculates wave characteristics, optimal crest levels, and freeboard margins for different operational conditions.

Data Requirements:
1. Reservoir Data CSV:
   - Columns: Option, Bottom Water Level (m), Top Water Level (m), Storage Volume (m3), Water Surface Area (m2), Embankment Slopes, Max Embankment Height (m)
2. Return Period Data CSV:
   - Columns: Year, Net Rainfall (mm)

The script provides interactive widgets to adjust scenario parameters and visualize results for different options and return periods.
""")

# File uploaders
uploaded_reservoir_file = st.file_uploader("Upload Reservoir Data CSV", type="csv")
uploaded_return_period_file = st.file_uploader("Upload Return Period Data CSV", type="csv")

# Load data
@st.cache_data
def load_data(reservoir_file, return_period_file):
    if reservoir_file is not None and return_period_file is not None:
        reservoir_data = pd.read_csv(reservoir_file)
        return_periods = pd.read_csv(return_period_file)
        return reservoir_data, return_periods
    return None, None

reservoir_data, return_periods = load_data(uploaded_reservoir_file, uploaded_return_period_file)

if reservoir_data is None or return_periods is None:
    st.warning("Please upload both CSV files to proceed with the analysis.")
    st.stop()

# Display the first few rows of each dataset
st.subheader("Reservoir Data Preview")
st.write(reservoir_data.head())

st.subheader("Return Period Data Preview")
st.write(return_periods.head())

# Process Embankment Slopes
reservoir_data['Embankment_Slopes_Numeric'] = reservoir_data['Embankment Slopes'].apply(lambda x: float(x.split('h:')[0]))

# Sidebar for scenario parameters
st.sidebar.header("Scenario Parameters")
CATEGORY_A_MIN_FREEBOARD = st.sidebar.number_input("Category A Min Freeboard (m)", value=0.6, step=0.1)
WIND_SPEED = st.sidebar.number_input("Wind Speed (m/s)", value=15.0, step=0.5)
PUMP_FAILURE_INFLOW = st.sidebar.number_input("Pump Failure Inflow (Mm³/hour)", value=0.75, step=0.05) * 1e6
PUMP_FAILURE_DURATION = st.sidebar.number_input("Pump Failure Duration (hours)", value=1, step=1)
OUTAGE_DURATION = st.sidebar.number_input("System Failure Outage Duration (hours)", value=24*7, step=24)

# Functions
def calculate_wave_characteristics(fetch, wind_speed, slope):
    g = 9.81  # acceleration due to gravity (m/s^2)
    Hs = 0.0016 * (wind_speed**2 / g) * np.sqrt(fetch / g)
    HD = Hs * 1.2  # Using the "Surfaced road" factor
    Rf = 2.2 if slope == 1/2 else 1.75 if slope == 1/3 else 2.0
    wave_surcharge = Rf * HD
    wave_surcharge = max(wave_surcharge, CATEGORY_A_MIN_FREEBOARD)
    safety_margin = 0.2  # meters
    total_freeboard = wave_surcharge + safety_margin
    return {
        'Hs': Hs,
        'HD': HD,
        'Rf': Rf,
        'wave_surcharge': wave_surcharge,
        'total_freeboard': total_freeboard
    }

def calculate_normal_operation(row, wind_speed, rainfall, year):
    top_level = row["Top Water Level (m)"]
    surface_area = row["Water Surface Area (m2)"]
    slope = 1 / row["Embankment_Slopes_Numeric"]
    fetch = np.sqrt(surface_area)
    wave_char = calculate_wave_characteristics(fetch, wind_speed, slope)
    volume_increase = rainfall * surface_area / 1000  # Convert mm to m
    depth_increase = volume_increase / surface_area
    optimal_crest = top_level + wave_char['total_freeboard']
    return pd.Series({
        'year': year,
        'optimal_crest': optimal_crest,
        'depth_increase': depth_increase,
        **wave_char
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
            for key, value in normal_op.items():
                row_results[f'Normal_Operation_{rp_row["Year"]}yr_{key}'] = value
        
        pump_failure = calculate_pump_failure(row)
        for key, value in pump_failure.items():
            row_results[f'Pump_Failure_{key}'] = value
        
        system_failure = calculate_system_failure(row, return_periods['Net Rainfall (mm)'].max())
        for key, value in system_failure.items():
            row_results[f'System_Failure_{key}'] = value
        
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
            value_name='Optimal Crest Level'
        )
        melted_data['Return Period'] = melted_data['Return Period'].str.extract(r'(\d+)').astype(int)
        fig = px.line(melted_data, x='Return Period', y='Optimal Crest Level', color='Option', markers=True)
        fig.update_layout(title="Normal Operation - Optimal Crest Level vs Return Period", 
                          xaxis_title="Return Period (years)", 
                          yaxis_title="Optimal Crest Level (m)")
    else:
        year = scenario.split('(')[1].split('yr')[0]
        column_to_plot = f'Normal_Operation_{year}yr_optimal_crest'
        fig = px.bar(filtered_results, x='Option', y=column_to_plot, color='Option')
        fig.update_layout(title=f"Normal Operation ({year}-year) - Optimal Crest Level by Option", 
                          xaxis_title="Option", 
                          yaxis_title="Optimal Crest Level (m)")
elif scenario == "Pump Failure":
    fig = px.bar(filtered_results, x='Option', y='Pump_Failure_optimal_crest', color='Option')
    fig.update_layout(title="Pump Failure - Optimal Crest Level by Option", 
                      xaxis_title="Option", 
                      yaxis_title="Optimal Crest Level (m)")
else:  # System Failure
    fig = px.bar(filtered_results, x='Option', y='System_Failure_optimal_crest', color='Option')
    fig.update_layout(title="System Failure - Optimal Crest Level by Option", 
                      xaxis_title="Option", 
                      yaxis_title="Optimal Crest Level (m)")

st.plotly_chart(fig)

# Wave Characteristics Dashboard
st.header("Wave Characteristics Dashboard")

wave_scenario = st.selectbox(
    "Choose return period for wave characteristics:",
    [f"{year}yr" for year in return_periods['Year']]
)

wave_characteristic = st.selectbox(
    "Choose wave characteristic to visualize:",
    ["Significant Wave Height (Hs)", "Design Wave Height (HD)", "Run-up Factor (Rf)", "Wave Surcharge", "Total Freeboard"]
)

wave_column = f"Normal_Operation_{wave_scenario}_{wave_characteristic.split('(')[0].strip().lower().replace(' ', '_')}"

wave_fig = px.bar(filtered_results, x='Option', y=wave_column, color='Option')
wave_fig.update_layout(title=f"{wave_characteristic} for {wave_scenario} Return Period", 
                       xaxis_title="Option", 
                       yaxis_title=wave_characteristic)
st.plotly_chart(wave_fig)

# Rainfall Frequency Curve
st.header("Rainfall Frequency Curve")

def perform_rainfall_frequency_analysis(rainfall_data):
    shape, loc, scale = stats.genextreme.fit(rainfall_data['Net Rainfall (mm)'])
    return_periods = np.logspace(0, 5, num=100)
    rainfall_depths = stats.genextreme.isf(1/return_periods, shape, loc, scale)
    return pd.DataFrame({'Return Period': return_periods, 'Rainfall': rainfall_depths})

rainfall_freq_data = perform_rainfall_frequency_analysis(return_periods)

rainfall_fig = px.line(rainfall_freq_data, x='Return Period', y='Rainfall', log_x=True)
rainfall_fig.update_layout(title="Rainfall Frequency Curve", 
                           xaxis_title="Return Period (years)", 
                           yaxis_title="Rainfall (mm)")
st.plotly_chart(rainfall_fig)

# Freeboard Margin Comparison
st.header("Freeboard Margin Comparison")
filtered_results['Max_Water_Level'] = filtered_results[[col for col in filtered_results.columns if col.endswith('_optimal_crest')]].max(axis=1)
filtered_results['Freeboard_Margin'] = filtered_results['Max_Water_Level'] - filtered_results['Top Water Level (m)']

freeboard_fig = px.bar(filtered_results, x='Option', y='Freeboard_Margin', color='Option')
freeboard_fig.update_layout(title="Freeboard Margin by Option", 
                            xaxis_title="Option", 
                            yaxis_title="Freeboard Margin (m)")
st.plotly_chart(freeboard_fig)

# Key Metrics Table
st.header("Key Metrics Table")
metrics_cols = ['Option', 'Storage Volume (m3)', 'Max Embankment Height (m)', 'Freeboard_Margin']
st.dataframe(filtered_results[metrics_cols].set_index('Option'))

# Download results
st.header("Download Results")

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
2. Consider options that maintain adequate freeboard under various conditions.
3. Evaluate the wave characteristics for each option to ensure they meet safety standards.
4. Conduct more detailed studies for options that perform well in this analysis.
5. Ensure compliance with relevant local and national regulations for dam safety.
6. Consider ease of operation and maintenance for long-term safety.

Remember that this analysis provides a high-level comparison. Final decision-making should involve a multidisciplinary team and consider additional factors not covered here.
""")

st.write("Analysis complete. Thank you for using the Comprehensive Reservoir Freeboard Analysis tool.")
