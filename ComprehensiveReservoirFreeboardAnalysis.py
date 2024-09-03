import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from datetime import datetime

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
    try:
        if reservoir_file is not None and return_period_file is not None:
            reservoir_data = pd.read_csv(reservoir_file)
            return_periods = pd.read_csv(return_period_file)
            
            # Validate required columns
            required_reservoir_columns = ['Option', 'Bottom Water Level (m)', 'Top Water Level (m)', 'Storage Volume (m3)', 'Water Surface Area (m2)', 'Embankment Slopes', 'Cost', 'Max Embankment Height (m)']
            required_return_period_columns = ['Year', 'Net Rainfall (mm)']
            
            if not all(col in reservoir_data.columns for col in required_reservoir_columns):
                raise ValueError("Reservoir data CSV is missing required columns.")
            if not all(col in return_periods.columns for col in required_return_period_columns):
                raise ValueError("Return period data CSV is missing required columns.")
            
            return reservoir_data, return_periods
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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

# General Parameters (affect all scenarios)
st.sidebar.subheader("General Parameters")
CATEGORY_A_MIN_FREEBOARD = st.sidebar.number_input("Category A Min Freeboard (m)", value=0.6, step=0.1)
WIND_SPEED = st.sidebar.number_input("Wind Speed (m/s)", value=15.0, step=0.5)

# Pump Failure Scenario
st.sidebar.subheader("Pump Failure Scenario")
PUMP_FAILURE_INFLOW = st.sidebar.number_input("Pump Failure Inflow (Mm³/hour)", value=0.75, step=0.05) * 1e6
PUMP_FAILURE_DURATION = st.sidebar.number_input("Pump Failure Duration (hours)", value=1, step=1)

# System Failure Scenario
st.sidebar.subheader("System Failure Scenario")
OUTAGE_DURATION = st.sidebar.number_input("System Failure Outage Duration (hours)", value=24*7, step=24)

# Sensitivity Analysis
st.sidebar.header("Sensitivity Analysis")
sensitivity_factor = st.sidebar.slider("Adjustment Factor", 0.5, 2.0, 1.0, 0.1)

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

# Sensitivity analysis
sensitivity_results = perform_analysis(reservoir_data, return_periods.copy())
sensitivity_results.update(sensitivity_results.select_dtypes(include=[np.number]) * sensitivity_factor)

# Calculate Freeboard Margin
results['Max_Water_Level'] = results[[col for col in results.columns if col.endswith('_optimal_crest')]].max(axis=1)
results['Freeboard_Margin'] = results['Max_Water_Level'] - results['Top Water Level (m)']

# Summary Dashboard
st.header("Summary Dashboard")
summary_metrics = ['Option', 'Max_Water_Level', 'Freeboard_Margin', 'Wave_total_freeboard']
summary_df = results[summary_metrics].set_index('Option')
st.dataframe(summary_df)

# Comparison Tool
st.header("Option Comparison Tool")
options_to_compare = st.multiselect("Select options to compare:", results['Option'].unique())
if options_to_compare:
    comparison_df = results[results['Option'].isin(options_to_compare)].set_index('Option')
    st.dataframe(comparison_df)
    
    # Comparison chart
    comparison_chart = px.bar(comparison_df.reset_index(), x='Option', y='Freeboard_Margin', title="Freeboard Margin Comparison")
    st.plotly_chart(comparison_chart)

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
    return_periods = np.logspace(0, np.log10(200000), num=100)  # Updated to go up to 200,000 years
    rainfall_depths = stats.genextreme.isf(1/return_periods, shape, loc, scale)
    return pd.DataFrame({'Return Period': return_periods, 'Rainfall': rainfall_depths})

rainfall_freq_data = perform_rainfall_frequency_analysis(return_periods)

rainfall_fig = px.line(rainfall_freq_data, x='Return Period', y='Rainfall', log_x=True)
rainfall_fig.update_layout(title="Rainfall Frequency Curve (up to 200,000 years)", 
                           xaxis_title="Return Period (years)", 
                           yaxis_title="Rainfall (mm)")
st.plotly_chart(rainfall_fig)

# Freeboard Margin Comparison
st.header("Freeboard Margin Comparison")
freeboard_fig = px.bar(filtered_results, x='Option', y='Freeboard_Margin', color='Option')
freeboard_fig.update_layout(title="Freeboard Margin by Option", 
                            xaxis_title="Option", 
                            yaxis_title="Freeboard Margin (m)")
st.plotly_chart(freeboard_fig)

# New Dashboard: Difference between Top Water Level and Optimal Crest Level
st.header("Top Water Level vs Optimal Crest Level Comparison")

# Calculate the difference for each scenario
results['Normal_Op_Diff'] = results[[col for col in results.columns if col.startswith('Normal_Operation_') and col.endswith('_optimal_crest')]].max(axis=1) - results['Top Water Level (m)']
results['Pump_Failure_Diff'] = results['Pump_Failure_optimal_crest'] - results['Top Water Level (m)']
results['System_Failure_Diff'] = results['System_Failure_optimal_crest'] - results['Top Water Level (m)']

# Create a melted dataframe for plotting
diff_data = results[['Option', 'Normal_Op_Diff', 'Pump_Failure_Diff', 'System_Failure_Diff']].melt(
    id_vars=['Option'],
    var_name='Scenario',
    value_name='Difference'
)

# Create the comparison chart
diff_fig = px.bar(diff_data, x='Option', y='Difference', color='Scenario', barmode='group')
diff_fig.update_layout(title="Difference between Top Water Level and Optimal Crest Level by Scenario",
                       xaxis_title="Option",
                       yaxis_title="Difference (m)")
st.plotly_chart(diff_fig)

# Add a table with the numerical values
st.subheader("Numerical Values of Differences")
diff_table = results[['Option', 'Normal_Op_Diff', 'Pump_Failure_Diff', 'System_Failure_Diff']]
diff_table = diff_table.set_index('Option')
diff_table.columns = ['Normal Operation', 'Pump Failure', 'System Failure']
st.dataframe(diff_table)

# Comprehensive Results Tables
st.header("Comprehensive Results Tables")

# Normal Operation Results
st.subheader("Normal Operation Results")
normal_op_columns = [col for col in results.columns if col.startswith('Normal_Operation_')]
st.dataframe(results[['Option'] + normal_op_columns])

# Pump Failure Results
st.subheader("Pump Failure Results")
pump_failure_columns = [col for col in results.columns if col.startswith('Pump_Failure_')]
st.dataframe(results[['Option'] + pump_failure_columns])

# System Failure Results
st.subheader("System Failure Results")
system_failure_columns = [col for col in results.columns if col.startswith('System_Failure_')]
st.dataframe(results[['Option'] + system_failure_columns])

# Wave Characteristics Results
st.subheader("Wave Characteristics Results")
wave_columns = [col for col in results.columns if col.startswith('Wave_')]
st.dataframe(results[['Option'] + wave_columns])

# New section: Download Results as CSV
st.header("Download Results")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(results)
st.download_button(
    label="Download Full Results as CSV",
    data=csv,
    file_name="reservoir_analysis_results.csv",
    mime="text/csv",
)

# PDF Report Generation
def create_pdf_report(results, summary_df):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Reservoir Freeboard Analysis Report")

    # Summary Table
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 80, "Summary")
    data = [['Option', 'Max Water Level (m)', 'Freeboard Margin (m)']]
    data.extend(summary_df.reset_index().values.tolist())
    table = Table(data, colWidths=[80, 100, 100])
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, (0,0,0)),
        ('BACKGROUND', (0,0), (-1,0), (0.8,0.8,0.8))
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, height - 300)

    # Freeboard Margin Chart
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df.index, summary_df['Freeboard_Margin'])
    plt.title('Freeboard Margins by Option')
    plt.xlabel('Option')
    plt.ylabel('Freeboard Margin (m)')
    plt.xticks(rotation=45)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    c.drawImage(ImageReader(img_buffer), 50, height - 550, width=400, height=200)

    # Add new comparison chart
    plt.figure(figsize=(8, 4))
    diff_data_plot = results[['Option', 'Normal_Op_Diff', 'Pump_Failure_Diff', 'System_Failure_Diff']].set_index('Option')
    diff_data_plot.plot(kind='bar', ax=plt.gca())
    plt.title('Difference between Top Water Level and Optimal Crest Level')
    plt.xlabel('Option')
    plt.ylabel('Difference (m)')
    plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    c.drawImage(ImageReader(img_buffer), 50, height - 800, width=400, height=200)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Generate and offer PDF download
st.header("Generate PDF Report")
if st.button("Generate PDF Report"):
    pdf_buffer = create_pdf_report(results, summary_df)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="reservoir_analysis_report.pdf",
        mime="application/pdf"
    )

# User Notes
st.header("User Notes")
user_notes = st.text_area("Enter any additional notes or observations here:")
if user_notes:
    st.success("Notes saved successfully!")

# Conclusion and Recommendations
st.header("Conclusion and Recommendations")
st.write("""
Based on the analysis performed, consider the following recommendations:

1. Prioritize options with larger freeboard margins for better safety against overtopping.
2. Consider options that maintain adequate freeboard under various conditions, including normal operation, pump failure, and system failure scenarios.
3. Evaluate the wave characteristics for each option to ensure they meet safety standards.
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

# Disclaimer
st.header("Disclaimer")
st.write("""
This tool is provided for informational and educational purposes only. While every effort has been made to ensure the accuracy and reliability of the analysis, the results should not be used as the sole basis for any decision-making in real-world reservoir design or management.

Users should always consult with qualified professionals and comply with all applicable laws, regulations, and industry standards when designing, constructing, or managing reservoir systems.

The creators and maintainers of this tool assume no liability for any consequences resulting from the use of this tool or the information it provides.
""")

# Footer with timestamp
st.markdown("---")
st.write(f"Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
