import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page config
st.set_page_config(page_title="Reservoir Freeboard Analysis", layout="wide")

# Title and creator information
st.title("Comprehensive Reservoir Freeboard Analysis")
st.write("Created by Mark Kirkpatrick (mark.kirkpatrick@aecom.com)")

# Data requirements information
st.info("""
This script requires two CSV files to be present in the same directory:
1. 'Reservoir_data.csv': Contains information about reservoir options.
2. 'extended_specific_return_periods.csv': Contains return period data.

Please ensure these files are available before running the script.
""")

# Sidebar for constants and inputs
st.sidebar.header("Constants and Inputs")

# Constants
st.sidebar.subheader("General Parameters")
CATEGORY_A_MIN_FREEBOARD = st.sidebar.number_input("Category A Min Freeboard (m)", value=0.6, step=0.1)
WIND_SPEED = st.sidebar.number_input("Wind Speed (m/s)", value=15.0, step=0.5)

st.sidebar.subheader("Pump Failure Scenario")
PUMP_FAILURE_INFLOW = st.sidebar.number_input("Pump Failure Inflow (Mm³/hour)", value=0.75, step=0.05) * 1e6
PUMP_FAILURE_DURATION = st.sidebar.number_input("Pump Failure Duration (hours)", value=1, step=1)

st.sidebar.subheader("System Failure Scenario")
OUTAGE_DURATION = st.sidebar.number_input("Outage Duration (hours)", value=24*7, step=24)

# Load data
@st.cache_data
def load_data():
    try:
        reservoir_data = pd.read_csv("Reservoir_data.csv")
        return_periods = pd.read_csv("extended_specific_return_periods.csv")
        return reservoir_data, return_periods
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure the required CSV files are in the same directory as the script.")
        st.stop()

reservoir_data, return_periods = load_data()

# Display the first few rows of each dataset
st.subheader("Reservoir Data Preview")
st.write(reservoir_data.head())

st.subheader("Return Period Data Preview")
st.write(return_periods.head())

# Process Embankment Slopes
reservoir_data['Embankment_Slopes_Numeric'] = reservoir_data['Embankment Slopes'].apply(lambda x: float(x.split('h:')[0]) if isinstance(x, str) else x)

# Functions
def calculate_wave_surcharge(fetch, wind_speed, dam_type, slope):
    g = 9.81  # acceleration due to gravity (m/s^2)
    
    # Calculate significant wave height (Hs)
    Hs = 0.0016 * (wind_speed**2 / g) * np.sqrt(fetch / g)
    
    # Get design wave height (HD)
    if dam_type == "Earthfill with random grass downstream face":
        HD = Hs * 1.2  # Using the "Surfaced road" factor
    else:
        HD = Hs  # Default case
    
    # Get run-up factor (Rf)
    if slope == 1/2:
        Rf = 2.2
    elif slope == 1/3:
        Rf = 1.75
    else:
        Rf = 2.0  # Default case
    
    # Calculate wave surcharge
    wave_surcharge = Rf * HD
    
    # Apply minimum allowance
    wave_surcharge = max(wave_surcharge, CATEGORY_A_MIN_FREEBOARD)
    
    # Add safety margin
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
    bottom_level = row["Bottom Water Level (m)"]
    top_level = row["Top Water Level (m)"]
    surface_area = row["Water Surface Area (m2)"]
    slope = 1 / row["Embankment_Slopes_Numeric"]
    
    fetch = np.sqrt(surface_area)
    
    wave_calc = calculate_wave_surcharge(
        fetch=fetch,
        wind_speed=wind_speed,
        dam_type="Earthfill with random grass downstream face",
        slope=slope
    )
    
    volume_increase = rainfall * surface_area / 1000  # Convert mm to m
    depth_increase = volume_increase / surface_area
    
    optimal_crest = top_level + wave_calc['total_freeboard']
    
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
    reservoir_results = reservoir_data.copy()
    
    # Normal Operation
    normal_operation_results = []
    for _, row in reservoir_data.iterrows():
        for _, rp_row in return_periods.iterrows():
            result = calculate_normal_operation(row, WIND_SPEED, rp_row['Net Rainfall (mm)'], rp_row['Year'])
            result['Option'] = row['Option']
            normal_operation_results.append(result)
    
    normal_operation_df = pd.DataFrame(normal_operation_results)
    normal_operation_pivot = normal_operation_df.pivot(index='Option', columns='year', values=['optimal_crest', 'depth_increase'])
    normal_operation_pivot.columns = [f'Normal_Operation_{col[1]}yr_{col[0]}' for col in normal_operation_pivot.columns]
    
    # Pump Failure
    pump_failure_results = reservoir_data.apply(calculate_pump_failure, axis=1)
    pump_failure_results.columns = ['Pump_Failure_' + col for col in pump_failure_results.columns]
    
    # System Failure
    system_failure_results = reservoir_data.apply(lambda row: calculate_system_failure(row, return_periods['Net Rainfall (mm)'].max()), axis=1)
    system_failure_results.columns = ['System_Failure_' + col for col in system_failure_results.columns]
    
    # Combine results
    final_results = pd.concat([
        reservoir_results,
        normal_operation_pivot,
        pump_failure_results,
        system_failure_results
    ], axis=1)
    
    return final_results

# Perform analysis
results = perform_analysis(reservoir_data, return_periods)

# Display results
st.header("Analysis Results")
st.dataframe(results)

# Plotting functions
def plot_normal_operation_surcharge(data):
    long_data = data.melt(
        id_vars=['Option'],
        value_vars=[col for col in data.columns if col.startswith('Normal_Operation_') and col.endswith('_optimal_crest')],
        var_name='Scenario',
        value_name='Water_Level'
    )
    long_data['Scenario'] = long_data['Scenario'].str.extract('(\d+)yr')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Option', y='Water_Level', hue='Scenario', data=long_data, ax=ax)
    ax.set_title("Normal Operation Surcharge Results")
    ax.set_xlabel("Option")
    ax.set_ylabel("Top Water Level (m)")
    plt.xticks(rotation=90)
    plt.legend(title="Return Period (years)", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

def plot_scenario_comparison(data):
    scenarios = ['Normal_Operation_100yr_optimal_crest', 'Pump_Failure_optimal_crest', 'System_Failure_optimal_crest']
    long_data = data.melt(
        id_vars=['Option'],
        value_vars=scenarios,
        var_name='Scenario',
        value_name='Water_Level'
    )
    long_data['Scenario'] = long_data['Scenario'].str.replace('_optimal_crest', '').str.replace('Normal_Operation_100yr', 'Normal')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Option', y='Water_Level', hue='Scenario', data=long_data, ax=ax)
    ax.set_title("Comparison of Surcharge Scenarios")
    ax.set_xlabel("Option")
    ax.set_ylabel("Top Water Level (m)")
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# Display plots
st.header("Visualizations")
plot_normal_operation_surcharge(results)
plot_scenario_comparison(results)

# Rainfall Frequency Analysis
st.header("Rainfall Frequency Analysis")

def perform_rainfall_frequency_analysis(rainfall_data):
    # Fit a GEV distribution to the data
    shape, loc, scale = stats.genextreme.fit(rainfall_data['Net Rainfall (mm)'])
    
    # Generate points for the frequency curve
    return_periods = [2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    rainfall_depths = stats.genextreme.isf(1/np.array(return_periods), shape, loc, scale)
    
    return pd.DataFrame({'return_period': return_periods, 'rainfall': rainfall_depths})

rainfall_frequency_data = perform_rainfall_frequency_analysis(return_periods)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rainfall_frequency_data['return_period'], rainfall_frequency_data['rainfall'])
ax.set_xscale('log')
ax.set_xlabel("Return Period (years)")
ax.set_ylabel("Net Rainfall (mm)")
ax.set_title("Rainfall Frequency Curve")
st.pyplot(fig)

# SIL Analysis
st.header("SIL Analysis")

def perform_sil_analysis(failure_rate, test_interval):
    pfd = 1 - np.exp(-failure_rate * test_interval)
    
    if pfd > 1e-1:
        sil_level = "SIL 0"
    elif 1e-2 < pfd <= 1e-1:
        sil_level = "SIL 1"
    elif 1e-3 < pfd <= 1e-2:
        sil_level = "SIL 2"
    elif 1e-4 < pfd <= 1e-3:
        sil_level = "SIL 3"
    else:
        sil_level = "SIL 4"
    
    return pfd, sil_level

failure_rate = st.number_input("Failure Rate", value=1e-4, format="%.2e", step=1e-5)
test_interval = st.number_input("Test Interval (hours)", value=8760, step=24)

pfd, sil_level = perform_sil_analysis(failure_rate, test_interval)

st.write(f"Probability of Failure on Demand (PFD): {pfd:.2e}")
st.write(f"SIL Level: {sil_level}")

# Cost Analysis
st.header("Cost Analysis")

# Create a scatter plot of Cost vs Storage Volume
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(reservoir_data['Storage Volume (m3)'], reservoir_data['Cost'], 
                     c=reservoir_data['Max Embankment Height (m)'], cmap='viridis')
ax.set_xlabel("Storage Volume (m³)")
ax.set_ylabel("Cost")
ax.set_title("Cost vs Storage Volume")
plt.colorbar(scatter, label='Max Embankment Height (m)')
st.pyplot(fig)

# Create a bar plot of costs for each option
fig, ax = plt.subplots(figsize=(12, 6))
reservoir_data.sort_values('Cost').plot(x='Option', y='Cost', kind='bar', ax=ax)
ax.set_xlabel("Option")
ax.set_ylabel("Cost")
ax.set_title("Cost Comparison of Reservoir Options")
plt.xticks(rotation=90)
st.pyplot(fig)

# Download results
st.header("Download Results")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(results)
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name="reservoir_analysis_results.csv",
    mime="text/csv",
)
