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
   - Columns: Option, Bottom Water Level (m), Top Water Level (m), Storage Volume (m3), Water Surface Area (m2), Embankment Slopes, Cost, Max Embankment Height (m)
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
    for idx, row in reservoir_data.iterrows():
        try:
            row_results = row.to_dict()  # Preserve all original columns
            fetch = np.sqrt(row["Water Surface Area (m2)"])
            slope = 1 / row["Embankment_Slopes_Numeric"]
            
            wave_char = calculate_wave_characteristics(fetch, WIND_SPEED, slope)
            row_results.update({f'Wave_{k}': v for k, v in wave_char.items()})
            
            for _, rp_row in return_periods.iterrows():
                normal_op = calculate_normal_operation(row, WIND_SPEED, rp_row['Net Rainfall (mm)'], rp_row['Year'])
                for key, value in normal_op.items():
                    row_results[f'Normal_Operation_{rp_row["Year"]:.1f}yr_{key}'] = value
            
            pump_failure = calculate_pump_failure(row)
            for key, value in pump_failure.items():
                row_results[f'Pump_Failure_{key}'] = value
            
            system_failure = calculate_system_failure(row, return_periods['Net Rainfall (mm)'].max())
            for key, value in system_failure.items():
                row_results[f'System_Failure_{key}'] = value
            
            results.append(row_results)
        except Exception as e:
            st.error(f"Error processing row {idx} (Option: {row['Option']}): {str(e)}")
            st.write("Row data:", row)
            raise e

    final_results = pd.DataFrame(results)
    return final_results

# Perform analysis
results = perform_analysis(reservoir_data, return_periods)

# Debug information
st.subheader("Debug Information")
st.write("Original reservoir_data columns:", reservoir_data.columns.tolist())
st.write("Results columns:", results.columns.tolist())

# Interactive Dashboard
st.header("Interactive Dashboard")

# Scenario selection
scenario_options = ["Normal Operation"] + [f"Normal Operation ({year:.1f}yr)" for year in return_periods['Year']] + ["Pump Failure", "System Failure"]
scenario = st.selectbox("Choose scenario to visualize:", scenario_options)

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
        melted_data['Return Period'] = melted_data['Return Period'].str.extract(r'(\d+\.?\d*)').astype(float)
        fig = px.line(melted_data, x='Return Period', y='Optimal Crest Level', color='Option', markers=True)
        fig.update_layout(title="Normal Operation - Optimal Crest Level vs Return Period", 
                          xaxis_title="Return Period (years)", 
                          yaxis_title="Optimal Crest Level (m)")
    else:
        year = float(scenario.split('(')[1].split('yr')[0])
        column_to_plot = f'Normal_Operation_{year:.1f}yr_optimal_crest'
        fig = px.bar(filtered_results, x='Option', y=column_to_plot, color='Option')
        fig.update_layout(title=f"Normal Operation ({year:.1f}-year) - Optimal Crest Level by Option", 
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

wave_characteristics = ['Hs', 'HD', 'Rf', 'wave_surcharge', 'total_freeboard']
selected_characteristic = st.selectbox("Select wave characteristic to display:", wave_characteristics)

wave_column = f'Wave_{selected_characteristic}'
if wave_column in filtered_results.columns:
    wave_fig = px.bar(filtered_results, x='Option', y=wave_column, color='Option')
    wave_fig.update_layout(title=f"{selected_characteristic} by Option", 
                           xaxis_title="Option", 
                           yaxis_title=selected_characteristic)
    st.plotly_chart(wave_fig)
else:
    st.error(f"Wave characteristic '{selected_characteristic}' not found in results.")

# Rainfall Frequency Analysis
st.header("Rainfall Frequency Analysis")

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

# Check if necessary columns exist
if 'Top Water Level (m)' not in results.columns:
    st.error("'Top Water Level (m)' column is missing from the results. Please check the analysis function.")
    st.stop()

# Calculate Max Water Level
optimal_crest_columns = [col for col in results.columns if col.endswith('_optimal_crest')]
if optimal_crest_columns:
    results['Max_Water_Level'] = results[optimal_crest_columns].max(axis=1)
    st.write("Max Water Level calculated from columns:", optimal_crest_columns)
else:
    st.error("No '_optimal_crest' columns found. Unable to calculate Max Water Level.")
    st.stop()

# Calculate Freeboard Margin
try:
    results['Freeboard_Margin'] = results['Max_Water_Level'] - results['Top Water Level (m)']
    
    freeboard_fig = px.bar(filtered_results, x='Option', y='Freeboard_Margin', color='Option')
    freeboard_fig.update_layout(title="Freeboard Margin by Option", 
                                xaxis_title="Option", 
                                yaxis_title="Freeboard Margin (m)")
    st.plotly_chart(freeboard_fig)
except Exception as e:
    st.error(f"Error calculating Freeboard Margin: {str(e)}")
    st.write("Results head:")
    st.write(results.head())

# Key Metrics Table
st.header("Key Metrics Table")
metrics_cols = ['Option', 'Storage Volume (m3)', 'Max Embankment Height (m)', 'Freeboard_Margin']
available_cols = [col for col in metrics_cols if col in filtered_results.columns]
if available_cols:
    st.dataframe(filtered_results[available_cols].set_index('Option'))
else:
    st.error("No matching columns found for Key Metrics Table.")

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
4. Conduct more detailed studies for options that perform well in this analysis.
5. Ensure compliance with relevant local and national regulations for dam safety.
6. Consider ease of operation and maintenance for long-term safety.
7. Evaluate the cost-effectiveness of each option, balancing safety considerations with economic factors.
8. Assess the environmental impact of each option, including effects on local ecosystems and water resources.
9. Consider the potential impacts of climate change on future rainfall patterns and adjust designs accordingly.
10. Involve stakeholders in the decision-making process, including local communities and regulatory bodies.

Remember that this analysis provides a high-level comparison. Final decision-making should involve a multidisciplinary team and consider additional factors not covered here.
""")

# Final Notes
st.header("Final Notes")
st.write("""
- This tool provides a comprehensive analysis of reservoir freeboard for various scenarios, but it should be used in conjunction with other engineering analyses and expert judgment.
- The results are based on the input data and parameters provided. Always verify the accuracy and relevance of input data.
- Regularly update the analysis with new data and revised parameters to ensure its continued relevance.
- Consider performing sensitivity analyses by adjusting key parameters to understand the robustness of different design options.
- Consult with geotechnical, hydrological, and structural engineering experts for a complete assessment of each reservoir option.
- Document all assumptions and limitations of this analysis for future reference and transparency in decision-making.
""")

# User Feedback
st.header("User Feedback")
st.write("""
We value your input! If you have any suggestions for improving this tool or encounter any issues, please let us know.
You can provide feedback by:
- Emailing the creator: mark.kirkpatrick@aecom.com
- Submitting an issue on our project repository (if applicable)
- Using the feedback form below
""")

feedback = st.text_area("Enter your feedback here:")
if st.button("Submit Feedback"):
    # Here you would typically send this feedback to a database or email
    # For now, we'll just acknowledge it
    st.success("Thank you for your feedback! We appreciate your input.")

# Disclaimer
st.header("Disclaimer")
st.write("""
This tool is provided for informational and educational purposes only. While every effort has been made to ensure the accuracy and reliability of the analysis, the results should not be used as the sole basis for any decision-making in real-world reservoir design or management.

Users should always consult with qualified professionals and comply with all applicable laws, regulations, and industry standards when designing, constructing, or managing reservoir systems.

The creators and maintainers of this tool assume no liability for any consequences resulting from the use of this tool or the information it provides.
""")

# Closing
st.write("Analysis complete. Thank you for using the Comprehensive Reservoir Freeboard Analysis tool.")

# Optional: Add a timestamp for when the analysis was run
from datetime import datetime
st.write(f"Analysis run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
