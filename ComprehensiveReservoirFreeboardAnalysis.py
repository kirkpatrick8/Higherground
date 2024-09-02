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

# Additional Analysis: Freeboard Margin
st.header("Freeboard Margin Analysis")

def calculate_freeboard_margin(row):
    normal_op_cols = [col for col in row.index if col.startswith('Normal_Operation_') and col.endswith('_optimal_crest')]
    max_normal_op = row[normal_op_cols].max()
    max_scenario = max(max_normal_op, row['Pump_Failure_optimal_crest'], row['System_Failure_optimal_crest'])
    return max_scenario - row['Top Water Level (m)']

results['Freeboard_Margin'] = results.apply(calculate_freeboard_margin, axis=1)

# Plot Freeboard Margin
st.subheader("Freeboard Margin for Each Option")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Option', y='Freeboard_Margin', data=results, ax=ax)
ax.set_xlabel("Option")
ax.set_ylabel("Freeboard Margin (m)")
ax.set_title("Freeboard Margin Comparison")
plt.xticks(rotation=90)
st.pyplot(fig)

# Sensitivity Analysis
st.header("Sensitivity Analysis")

st.write("Adjust the parameters below to see how they affect the freeboard margin:")

wind_speed_factor = st.slider("Wind Speed Factor", 0.5, 2.0, 1.0, 0.1)
rainfall_factor = st.slider("Rainfall Factor", 0.5, 2.0, 1.0, 0.1)

@st.cache_data
def perform_sensitivity_analysis(reservoir_data, return_periods, wind_speed_factor, rainfall_factor):
    # Modify input data for sensitivity analysis
    modified_reservoir_data = reservoir_data.copy()
    modified_return_periods = return_periods.copy()
    
    # Adjust wind speed and rainfall
    global WIND_SPEED
    WIND_SPEED *= wind_speed_factor
    modified_return_periods['Net Rainfall (mm)'] *= rainfall_factor
    
    # Perform analysis with modified data
    sensitivity_results = perform_analysis(modified_reservoir_data, modified_return_periods)
    
    # Calculate new freeboard margin
    sensitivity_results['Freeboard_Margin'] = sensitivity_results.apply(calculate_freeboard_margin, axis=1)
    
    return sensitivity_results

sensitivity_results = perform_sensitivity_analysis(reservoir_data, return_periods, wind_speed_factor, rainfall_factor)

# Plot sensitivity results
st.subheader("Sensitivity Analysis: Freeboard Margin")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Option', y='Freeboard_Margin', data=sensitivity_results, ax=ax)
ax.set_xlabel("Option")
ax.set_ylabel("Freeboard Margin (m)")
ax.set_title(f"Freeboard Margin (Wind Speed Factor: {wind_speed_factor}, Rainfall Factor: {rainfall_factor})")
plt.xticks(rotation=90)
st.pyplot(fig)

# Comparison of original and sensitivity results
st.subheader("Comparison: Original vs Sensitivity Analysis")
comparison_data = pd.DataFrame({
    'Option': results['Option'],
    'Original_Freeboard_Margin': results['Freeboard_Margin'],
    'Sensitivity_Freeboard_Margin': sensitivity_results['Freeboard_Margin']
})

fig, ax = plt.subplots(figsize=(12, 6))
comparison_data.plot(x='Option', y=['Original_Freeboard_Margin', 'Sensitivity_Freeboard_Margin'], kind='bar', ax=ax)
ax.set_xlabel("Option")
ax.set_ylabel("Freeboard Margin (m)")
ax.set_title("Comparison of Freeboard Margins: Original vs Sensitivity Analysis")
plt.xticks(rotation=90)
plt.legend(["Original", "Sensitivity"])
st.pyplot(fig)

# Conclusion
st.header("Conclusion and Recommendations")
st.write("""
Based on the analysis performed, here are some key observations and recommendations:

1. Freeboard Margin: Options with higher freeboard margins generally provide better safety against overtopping. Consider prioritizing options with larger freeboard margins.

2. Cost-Effectiveness: When comparing options, consider the balance between cost and storage volume. Options that provide more storage for a lower cost may be more economical, provided they meet safety requirements.

3. Sensitivity to Environmental Factors: The sensitivity analysis shows how changes in wind speed and rainfall affect the freeboard margin. Options that maintain adequate freeboard under various conditions may be more resilient.

4. SIL Analysis: Ensure that the chosen option meets the required Safety Integrity Level (SIL) for the project.

5. Further Investigation: For options that perform well in this analysis, consider conducting more detailed studies, including geotechnical investigations, environmental impact assessments, and detailed cost estimations.

6. Regulatory Compliance: Ensure that the selected option complies with all relevant local and national regulations for dam safety and reservoir management.

7. Operational Considerations: Consider the ease of operation and maintenance for each option, as these factors can impact long-term costs and safety.

Remember that this analysis provides a high-level comparison of options. Final decision-making should involve a multidisciplinary team and consider additional factors not covered in this analysis.
""")

# End of script
st.write("Analysis complete. Thank you for using the Comprehensive Reservoir Freeboard Analysis tool.")
