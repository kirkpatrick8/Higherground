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
This script requires two CSV files to be uploaded:
1. Reservoir Data CSV:
   Required columns: Option, Bottom Water Level (m), Top Water Level (m), Storage Volume (m3), Water Surface Area (m2), Embankment Slopes, Cost, Max Embankment Height (m)

2. Return Period Data CSV:
   Required columns: Year, Net Rainfall (mm)

Please ensure your CSV files contain these columns before uploading.
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
    else:
        return None, None

reservoir_data, return_periods = load_data(uploaded_reservoir_file, uploaded_return_period_file)

if reservoir_data is None or return_periods is None:
    st.warning("Please upload both CSV files to proceed with the analysis.")
    st.stop()

# Check for required columns
required_reservoir_columns = ["Option", "Bottom Water Level (m)", "Top Water Level (m)", "Storage Volume (m3)", "Water Surface Area (m2)", "Embankment Slopes", "Cost", "Max Embankment Height (m)"]
required_return_period_columns = ["Year", "Net Rainfall (mm)"]

missing_reservoir_columns = [col for col in required_reservoir_columns if col not in reservoir_data.columns]
missing_return_period_columns = [col for col in required_return_period_columns if col not in return_periods.columns]

if missing_reservoir_columns or missing_return_period_columns:
    st.error(f"Missing columns in Reservoir Data: {', '.join(missing_reservoir_columns)}")
    st.error(f"Missing columns in Return Period Data: {', '.join(missing_return_period_columns)}")
    st.stop()

# Display the first few rows of each dataset
st.subheader("Reservoir Data Preview")
st.write(reservoir_data.head())

st.subheader("Return Period Data Preview")
st.write(return_periods.head())

# Process Embankment Slopes
reservoir_data['Embankment_Slopes_Numeric'] = reservoir_data['Embankment Slopes'].apply(lambda x: float(x.split('h:')[0]) if isinstance(x, str) else x)

# The rest of your script remains the same...
# (Include all the functions and analysis code here)

# Perform analysis
results = perform_analysis(reservoir_data, return_periods)

# Display results
st.header("Analysis Results")
st.dataframe(results)

# Display plots
st.header("Visualizations")
plot_normal_operation_surcharge(results)
plot_scenario_comparison(results)

# Rainfall Frequency Analysis
st.header("Rainfall Frequency Analysis")
rainfall_frequency_data = perform_rainfall_frequency_analysis(return_periods)
plot_rainfall_frequency_curve(rainfall_frequency_data)

# SIL Analysis
st.header("SIL Analysis")
failure_rate = st.number_input("Failure Rate", value=1e-4, format="%.2e", step=1e-5)
test_interval = st.number_input("Test Interval (hours)", value=8760, step=24)
pfd, sil_level = perform_sil_analysis(failure_rate, test_interval)
st.write(f"Probability of Failure on Demand (PFD): {pfd:.2e}")
st.write(f"SIL Level: {sil_level}")

# Cost Analysis
st.header("Cost Analysis")
plot_cost_vs_storage(reservoir_data)
plot_cost_comparison(reservoir_data)

# Download results
st.header("Download Results")
csv = convert_df_to_csv(results)
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name="reservoir_analysis_results.csv",
    mime="text/csv",
)
