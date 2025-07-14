import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple

class COTPlotter:
    """
    A class for plotting Commitment of Traders (COT) reports using Plotly.
    Creates interactive charts matching professional COT report visualizations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the COT plotter with data."""
        self.df = df.copy()
        self.prepare_data()
        
        # Define color scheme
        self.colors = {
            'commercials': '#FF6B6B',
            'non_commercials': '#4ECDC4',
            'small_speculators': '#45B7D1',
            'swap_dealers': '#96CEB4',
            'money_managers': '#FFEAA7',
            'other_reportables': '#DDA0DD'
        }
        
    def prepare_data(self):
        """Prepare and clean the data for plotting."""
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        # Calculate net positions
        self.df['producer_merchant_net'] = (
            self.df['producer_merchant_processor_user_longs'] - 
            self.df['producer_merchant_processor_user_shorts']
        )
        
        self.df['swap_dealer_net'] = (
            self.df['swap_dealer_longs'] - 
            self.df['swap_dealer_shorts']
        )
        
        self.df['money_manager_net'] = (
            self.df['money_manager_longs'] - 
            self.df['money_manager_shorts']
        )
        
        self.df['other_reportable_net'] = (
            self.df['other_reportable_longs'] - 
            self.df['other_reportable_shorts']
        )
        
        # Calculate total non-commercial positions
        self.df['non_commercial_longs'] = (
            self.df['money_manager_longs'] + self.df['swap_dealer_longs']
        )
        
        self.df['non_commercial_shorts'] = (
            self.df['money_manager_shorts'] + self.df['swap_dealer_shorts']
        )
        
        self.df['non_commercial_net'] = (
            self.df['non_commercial_longs'] - self.df['non_commercial_shorts']
        )
        
    def plot_cot_report(self, 
                       show_net_positions: bool = True,
                       show_disaggregated: bool = True,
                       date_range: Optional[Tuple[str, str]] = None,
                       title: str = "Commitment of Traders Report",
                       height: int = 800) -> go.Figure:
        """Create comprehensive COT report visualization."""
        
        # Filter data by date range if specified
        plot_data = self.df.copy()
        if date_range:
            start_date, end_date = date_range
            plot_data = plot_data[
                (plot_data['date'] >= start_date) & 
                (plot_data['date'] <= end_date)
            ]
        
        # Create subplots
        if show_disaggregated:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Traditional COT View", "Disaggregated COT View"),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
        else:
            fig = go.Figure()
        
        # Plot traditional COT view
        self._add_traditional_cot(fig, plot_data, show_net_positions, row=1 if show_disaggregated else None)
        
        # Plot disaggregated view if requested
        if show_disaggregated:
            self._add_disaggregated_cot(fig, plot_data, show_net_positions, row=2)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Date",
            yaxis_title="Contracts"
        )
        
        if show_disaggregated:
            fig.update_yaxes(title_text="Contracts", row=1, col=1)
            fig.update_yaxes(title_text="Contracts", row=2, col=1)
        
        return fig
        
    def _add_traditional_cot(self, fig, data, show_net_positions, row=None):
        """Add traditional COT traces to figure."""
        dates = data['date']
        
        if show_net_positions:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_net'], 
                          name='Commercials (Net)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['non_commercial_net'], 
                          name='Non-Commercials (Net)', line=dict(color=self.colors['non_commercials'], width=2)),
                go.Scatter(x=dates, y=data['non_reportable_longs'] - data['non_reportable_shorts'], 
                          name='Small Speculators (Net)', line=dict(color=self.colors['small_speculators'], width=2))
            ]
        else:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_processor_user_longs'], 
                          name='Commercials (Long)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['producer_merchant_processor_user_shorts'], 
                          name='Commercials (Short)', line=dict(color=self.colors['commercials'], width=2, dash='dash')),
                go.Scatter(x=dates, y=data['non_commercial_longs'], 
                          name='Non-Commercials (Long)', line=dict(color=self.colors['non_commercials'], width=2)),
                go.Scatter(x=dates, y=data['non_commercial_shorts'], 
                          name='Non-Commercials (Short)', line=dict(color=self.colors['non_commercials'], width=2, dash='dash'))
            ]
        
        for trace in traces:
            fig.add_trace(trace, row=row, col=1)
        
        # Add zero line for net positions
        if show_net_positions:
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=row, col=1)
            
    def _add_disaggregated_cot(self, fig, data, show_net_positions, row):
        """Add disaggregated COT traces to figure."""
        dates = data['date']
        
        if show_net_positions:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_net'], 
                          name='Producer/Merchant (Net)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['swap_dealer_net'], 
                          name='Swap Dealers (Net)', line=dict(color=self.colors['swap_dealers'], width=2)),
                go.Scatter(x=dates, y=data['money_manager_net'], 
                          name='Money Managers (Net)', line=dict(color=self.colors['money_managers'], width=2)),
                go.Scatter(x=dates, y=data['other_reportable_net'], 
                          name='Other Reportables (Net)', line=dict(color=self.colors['other_reportables'], width=2))
            ]
        else:
            traces = [
                go.Scatter(x=dates, y=data['producer_merchant_processor_user_longs'], 
                          name='Producer/Merchant (Long)', line=dict(color=self.colors['commercials'], width=2)),
                go.Scatter(x=dates, y=data['swap_dealer_longs'], 
                          name='Swap Dealers (Long)', line=dict(color=self.colors['swap_dealers'], width=2)),
                go.Scatter(x=dates, y=data['money_manager_longs'], 
                          name='Money Managers (Long)', line=dict(color=self.colors['money_managers'], width=2)),
                go.Scatter(x=dates, y=data['other_reportable_longs'], 
                          name='Other Reportables (Long)', line=dict(color=self.colors['other_reportables'], width=2))
            ]
        
        for trace in traces:
            fig.add_trace(trace, row=row, col=1)
        
        # Add zero line for net positions
        if show_net_positions:
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3, row=row, col=1)
            
    def plot_concentration_analysis(self, 
                                  date_range: Optional[Tuple[str, str]] = None,
                                  height: int = 600) -> go.Figure:
        """Plot concentration analysis as stacked area chart."""
        
        # Filter data by date range if specified
        plot_data = self.df.copy()
        if date_range:
            start_date, end_date = date_range
            plot_data = plot_data[
                (plot_data['date'] >= start_date) & 
                (plot_data['date'] <= end_date)
            ]
        
        # Calculate percentages
        total_longs = plot_data['total_reportable_longs'] + plot_data['non_reportable_longs']
        
        fig = go.Figure()
        
        # Add stacked area traces
        categories = [
            ('Commercials', plot_data['producer_merchant_processor_user_longs'] / total_longs * 100, self.colors['commercials']),
            ('Swap Dealers', plot_data['swap_dealer_longs'] / total_longs * 100, self.colors['swap_dealers']),
            ('Money Managers', plot_data['money_manager_longs'] / total_longs * 100, self.colors['money_managers']),
            ('Other Reportables', plot_data['other_reportable_longs'] / total_longs * 100, self.colors['other_reportables']),
            ('Non-Reportables', plot_data['non_reportable_longs'] / total_longs * 100, self.colors['small_speculators'])
        ]
        
        for name, values, color in categories:
            fig.add_trace(go.Scatter(
                x=plot_data['date'],
                y=values,
                mode='lines',
                stackgroup='one',
                name=name,
                line=dict(width=0.5, color=color),
                fill='tonexty'
            ))
        
        fig.update_layout(
            title='COT Position Concentration Analysis',
            xaxis_title='Date',
            yaxis_title='Percentage of Total Open Interest (%)',
            height=height,
            hovermode='x unified'
        )
        
        return fig
        
    def get_latest_positions(self) -> Dict:
        """Get the latest position data for all categories."""
        latest_data = self.df.iloc[-1]
        
        return {
            'date': latest_data['date'],
            'commercials': {
                'longs': latest_data['producer_merchant_processor_user_longs'],
                'shorts': latest_data['producer_merchant_processor_user_shorts'],
                'net': latest_data['producer_merchant_net']
            },
            'swap_dealers': {
                'longs': latest_data['swap_dealer_longs'],
                'shorts': latest_data['swap_dealer_shorts'],
                'net': latest_data['swap_dealer_net']
            },
            'money_managers': {
                'longs': latest_data['money_manager_longs'],
                'shorts': latest_data['money_manager_shorts'],
                'net': latest_data['money_manager_net']
            }
        }

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