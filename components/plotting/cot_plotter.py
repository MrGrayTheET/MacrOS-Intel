# Example usage:
"""
# Load your COT data
df = pd.read_csv('your_cot_data.csv')

# Create plotter instance
cot_plotter = COTPlotter(df)

# Plot COT report
fig = cot_plotter.plot_cot_report(
    show_net_positions=True,
    show_disaggregated=True,
    title="Crude Oil COT Report"
)
fig.show()

# Show concentration analysis
fig2 = cot_plotter.plot_concentration_analysis()
fig2.show()

# Get latest positions
latest = cot_plotter.get_latest_positions()
print(f"Latest positions as of {latest['date']}")
"""