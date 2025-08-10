from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from dash import html, dcc, callback, Output, Input, callback_context, no_update, State
from plotly import graph_objects as go, express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from data.data_tables import EIATable
from components.frames import FundamentalFrame, FlexibleMenu
from callbacks.callback_registry import CallbackRegistry
from utils.data_tools import key_to_name, fetch_contract_keys, calc_contract_spreads


class LinearRegressionApp:
    """
    General linear regression application framework with variable selection
    """
    
    def __init__(self, 
                 table_client,
                 app_id: str = 'linear-regression',
                 title: str = 'Linear Regression Analysis'):
        self.table_client = table_client
        self.app_id = app_id
        self.title = title
        self.registry = CallbackRegistry()
        self.data = pd.DataFrame()
        self.filtered_data = pd.DataFrame()
        self.menu = None
        self.frame = None
        
    def setup_menu(self, additional_components: List[Dict] = None) -> FlexibleMenu:
        """Setup the flexible menu with standard regression controls"""
        menu = FlexibleMenu(
            menu_id=f'{self.app_id}-menu',
            title='Regression Controls',
            width='350px'
        )
        
        # Date range picker
        menu.add_date_range_picker(
            f'{self.app_id}-date-range',
            'Date Range',
            start_date=self.data.index.min() if not self.data.empty else datetime.now() - timedelta(days=365),
            end_date=self.data.index.max() if not self.data.empty else datetime.now()
        )
        
        # Variable selection dropdowns
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        options = [{'label': col, 'value': col} for col in numeric_columns]
        
        menu.add_dropdown(
            f'{self.app_id}-x-variable',
            'Independent Variable (X)',
            options=options,
            value=numeric_columns[0] if numeric_columns else None
        )
        
        menu.add_dropdown(
            f'{self.app_id}-y-variable', 
            'Dependent Variable (Y)',
            options=options,
            value=numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0] if numeric_columns else None
        )
        
        # Filter controls
        menu.add_range_slider(
            f'{self.app_id}-month-filter',
            'Month Filter',
            min_val=1, max_val=12,
            value=[1, 12]
        )
        
        # Action buttons
        menu.add_button(f'{self.app_id}-apply-filters', 'Apply Filters', style={'margin': '10px'})
        menu.add_button(f'{self.app_id}-reset-filters', 'Reset Filters', style={'margin': '10px'})
        
        # Add any additional components
        if additional_components:
            for component in additional_components:
                comp = component.copy()
                comp_type = comp.pop('type')
                
                if comp_type == 'dropdown':
                    menu.add_dropdown(**comp)
                elif comp_type == 'range_slider':
                    menu.add_range_slider(**comp)
                elif comp_type == 'button':
                    menu.add_button(**comp)
        
        return menu
    
    def setup_frame(self) -> FundamentalFrame:
        """Setup the fundamental frame with regression charts"""
        chart_configs = [
            {
                'title': 'Scatter Plot with Regression Line',
                'chart_type': 'scatter',
                'starting_key': '',
                'width': '100%',
                'height': 400
            },
            {
                'title': 'Time Series Comparison',
                'chart_type': 'line',
                'starting_key': '',
                'width': '100%', 
                'height': 300
            },
            {
                'title': 'Residuals Analysis',
                'chart_type': 'scatter',
                'starting_key': '',
                'width': '100%',
                'height': 300
            }
        ]
        
        frame = FundamentalFrame(
            table_client=self.table_client,
            chart_configs=chart_configs,
            div_prefix=f'{self.app_id}-frame'
        )
        
        return frame
    
    def apply_filters(self, data: pd.DataFrame, filter_params: Dict) -> pd.DataFrame:
        """Apply filters to the data"""
        filtered = data.copy()
        
        # Date range filter
        if 'date_range' in filter_params and filter_params['date_range']:
            start_date, end_date = filter_params['date_range']
            if start_date and end_date:
                filtered = filtered[
                    (filtered.index >= pd.to_datetime(start_date)) &
                    (filtered.index <= pd.to_datetime(end_date))
                ]
        
        # Month filter
        if 'month_range' in filter_params and filter_params['month_range']:
            month_min, month_max = filter_params['month_range']
            filtered = filtered[
                (filtered.index.month >= month_min) & 
                (filtered.index.month <= month_max)
            ]
        
        return filtered
    
    def perform_regression(self, data: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """Perform linear regression and return results"""
        if x_col not in data.columns or y_col not in data.columns:
            return {}
        
        # Remove NaN values
        clean_data = data[[x_col, y_col]].dropna()
        
        if len(clean_data) < 2:
            return {}
        
        X = clean_data[x_col].values.reshape(-1, 1)
        y = clean_data[y_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return {
            'model': model,
            'r2': r2,
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'predictions': y_pred,
            'residuals': y - y_pred,
            'x_values': clean_data[x_col].values,
            'y_values': y,
            'data': clean_data
        }
    
    def create_scatter_figure(self, regression_results: Dict, x_col: str, y_col: str) -> go.Figure:
        """Create scatter plot with regression line"""
        if not regression_results:
            return go.Figure().add_annotation(
                text="No valid data for regression",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Scatter points
        fig.add_trace(go.Scatter(
            x=regression_results['x_values'],
            y=regression_results['y_values'],
            mode='markers',
            name='Data Points',
            marker=dict(color='blue', size=6)
        ))
        
        # Regression line
        fig.add_trace(go.Scatter(
            x=regression_results['x_values'],
            y=regression_results['predictions'],
            mode='lines',
            name=f'Regression Line (R² = {regression_results["r2"]:.3f})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{y_col} vs {x_col}',
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='closest'
        )
        
        return fig
    
    def create_timeseries_figure(self, data: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
        """Create time series comparison figure"""
        fig = go.Figure()
        
        if x_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[x_col],
                name=x_col,
                line=dict(color='blue'),
                yaxis='y'
            ))
        
        if y_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[y_col],
                name=y_col,
                line=dict(color='red'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title='Time Series Comparison',
            xaxis_title='Date',
            yaxis=dict(title=x_col, side='left'),
            yaxis2=dict(title=y_col, side='right', overlaying='y')
        )
        
        return fig
    
    def create_residuals_figure(self, regression_results: Dict) -> go.Figure:
        """Create residuals analysis figure"""
        if not regression_results:
            return go.Figure().add_annotation(
                text="No residuals to display",
                xref="paper", yref="paper", 
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=regression_results['predictions'],
            y=regression_results['residuals'],
            mode='markers',
            name='Residuals',
            marker=dict(color='green', size=6)
        ))
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Residuals vs Fitted Values',
            xaxis_title='Fitted Values',
            yaxis_title='Residuals'
        )
        
        return fig
    
    def generate_layout(self) -> html.Div:
        """Generate the complete application layout"""
        self.menu = self.setup_menu()
        self.frame = self.setup_frame()
        
        return html.Div([
            html.H1(self.title, style={'textAlign': 'center'}),
            html.Div([
                self.menu.generate_layout(),
                html.Div([
                    # Main charts area
                    html.Div(id=f'{self.app_id}-scatter-chart'),
                    html.Div(id=f'{self.app_id}-timeseries-chart'),
                    html.Div(id=f'{self.app_id}-residuals-chart'),
                    
                    # Metrics display
                    html.Div(id=f'{self.app_id}-metrics', 
                            style={'margin': '20px', 'padding': '20px', 
                                   'border': '1px solid #ddd', 'borderRadius': '5px'})
                ], style={'flex': '1', 'padding': '20px'})
            ], style={'display': 'flex', 'height': '100vh'})
        ])


class NaturalGasStorageApp(LinearRegressionApp):
    """
    Specialized Natural Gas Storage Analysis Application
    """
    
    def __init__(self, 
                 storage_key='storage/total_lower_48',
                 price_key='prices/NG_1', 
                 withdrawal_key='consumption/net_withdrawals',
                 app_id='natgas-storage'):
        
        table_client = EIATable('NG')
        super().__init__(table_client, app_id, 'Natural Gas Storage Analysis')
        
        self.storage_key = storage_key
        self.price_key = price_key
        self.withdrawal_key = withdrawal_key
        
        # Get column names
        self.storage_col = key_to_name(storage_key)
        self.price_col = key_to_name(price_key)
        self.withdrawal_col = key_to_name(withdrawal_key)
        
        # Load and process data
        self._load_data()
        
    def _load_data(self):
        """Load and process natural gas data"""
        raw_data = self.table_client.get_keys([self.storage_key, self.price_key, self.withdrawal_key])
        raw_data = raw_data.ffill()
        
        # Resample to weekly
        self.data = raw_data.resample('1W').agg({
            self.storage_col: 'last',
            self.price_col: 'last', 
            self.withdrawal_col: 'last'
        })
        
        # Calculate seasonal storage metrics
        self.data = self._calculate_seasonal_storage_metrics(self.data)
        
    def _calculate_seasonal_storage_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonal storage metrics including (data - 5ymin)/(5ymax - 5ymin)
        """
        df = data.copy()
        df['date'] = df.index
        df = df.sort_values('date')
        
        # Add temporal columns
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Initialize columns
        df['hist_max'] = np.nan
        df['hist_min'] = np.nan
        df['hist_mean'] = np.nan
        
        # Calculate 5-year historical metrics for each week
        for idx, row in df.iterrows():
            current_week = row['week_of_year']
            current_year = row['year']
            
            # Get historical data for same week over past 5 years
            historical_mask = (
                (df['week_of_year'] == current_week) &
                (df['year'] <= current_year) &
                (df['year'] > current_year - 5) &
                (df.index <= idx)
            )
            historical_data = df.loc[historical_mask, self.storage_col]
            
            if len(historical_data) > 0:
                df.loc[idx, 'hist_max'] = historical_data.max()
                df.loc[idx, 'hist_min'] = historical_data.min()
                df.loc[idx, 'hist_mean'] = historical_data.mean()
        
        # Calculate the key metric: (data - 5ymin)/(5ymax - 5ymin)
        df['storage_percentile'] = (
            (df[self.storage_col] - df['hist_min']) / 
            (df['hist_max'] - df['hist_min'])
        ) * 100
        
        # Additional metrics
        df['dev_from_mean'] = (df[self.storage_col] - df['hist_mean']) / df['hist_mean']
        
        # Forward-looking price changes
        df['price_1mo'] = df[self.price_col].shift(-4)  # 4 weeks ahead
        df['price_3mo'] = df[self.price_col].shift(-12)  # 12 weeks ahead
        df['return_1mo'] = (df['price_1mo'] / df[self.price_col]) - 1
        df['return_3mo'] = (df['price_3mo'] / df[self.price_col]) - 1
        
        # Signal strength deciles
        df['signal_strength_decile'] = pd.qcut(df['dev_from_mean'].dropna(), 10, labels=False, duplicates='drop')
        
        return df
    
    def setup_menu(self, additional_components: List[Dict] = None) -> FlexibleMenu:
        """Setup menu with natural gas specific controls"""
        
        # Natural gas specific components
        natgas_components = [
            {
                'type': 'dropdown',
                'component_id': 'storage-series',
                'label': 'Storage Series',
                'options': [
                    {'label': 'Total Lower 48', 'value': 'storage/total_lower_48'},
                    {'label': 'East Region', 'value': 'storage/east'},
                    {'label': 'West Region', 'value': 'storage/west'},
                    {'label': 'Producing Region', 'value': 'storage/producing'}
                ],
                'value': self.storage_key
            },
            {
                'type': 'dropdown', 
                'component_id': 'price-series',
                'label': 'Price Series',
                'options': [
                    {'label': 'Natural Gas Futures (Front Month)', 'value': 'prices/NG_1'},
                    {'label': 'Natural Gas Futures (2nd Month)', 'value': 'prices/NG_2'},
                    {'label': 'Henry Hub Spot', 'value': 'prices/henry_hub_spot'}
                ],
                'value': self.price_key
            },
            {
                'type': 'range_slider',
                'component_id': 'signal-strength-filter',
                'label': 'Signal Strength Decile',
                'min_val': 0, 'max_val': 9,
                'value': [0, 9]
            }
        ]
        
        # Combine with additional components if provided
        all_components = natgas_components + (additional_components or [])
        
        return super().setup_menu(all_components)
    
    def register_callbacks(self, app):
        """Register callbacks for the natural gas storage app"""
        
        @self.registry.register(
            name=f'{self.app_id}-main-callback',
            outputs=[
                (f'{self.app_id}-scatter-chart', 'children'),
                (f'{self.app_id}-timeseries-chart', 'children'),
                (f'{self.app_id}-residuals-chart', 'children'),
                (f'{self.app_id}-metrics', 'children')
            ],
            inputs=[
                (f'{self.app_id}-menu_x-variable', 'value'),
                (f'{self.app_id}-menu_y-variable', 'value'),
                (f'{self.app_id}-menu_date-range', 'start_date'),
                (f'{self.app_id}-menu_date-range', 'end_date'),
                (f'{self.app_id}-menu_month-filter', 'value'),
                (f'{self.app_id}-menu_storage-series', 'value'),
                (f'{self.app_id}-menu_price-series', 'value'),
                (f'{self.app_id}-menu_apply-filters', 'n_clicks'),
                (f'{self.app_id}-menu_reset-filters', 'n_clicks')
            ],
            states=[
                (f'{self.app_id}-menu_signal-strength-filter', 'value')
            ]
        )
        def update_analysis(x_var, y_var, start_date, end_date, month_range, 
                           storage_series, price_series, apply_clicks, reset_clicks,
                           signal_strength_range):
            
            # Handle data updates if series changed
            if storage_series != self.storage_key or price_series != self.price_key:
                self.storage_key = storage_series
                self.price_key = price_series
                self.storage_col = key_to_name(storage_series)
                self.price_col = key_to_name(price_series)
                self._load_data()
            
            # Apply filters
            filter_params = {
                'date_range': [start_date, end_date],
                'month_range': month_range
            }
            
            ctx = callback_context
            if ctx.triggered and f'{self.app_id}-menu_reset-filters' in ctx.triggered[0]['prop_id']:
                # Reset filters
                filtered_data = self.data.copy()
            else:
                filtered_data = self.apply_filters(self.data, filter_params)
            
            # Additional filtering by signal strength
            if signal_strength_range:
                min_strength, max_strength = signal_strength_range
                filtered_data = filtered_data[
                    (filtered_data['signal_strength_decile'] >= min_strength) &
                    (filtered_data['signal_strength_decile'] <= max_strength)
                ]
            
            # Perform regression
            regression_results = self.perform_regression(filtered_data, x_var, y_var)
            
            # Create figures
            scatter_fig = self.create_scatter_figure(regression_results, x_var, y_var)
            timeseries_fig = self.create_timeseries_figure(filtered_data, x_var, y_var)
            residuals_fig = self.create_residuals_figure(regression_results)
            
            # Create metrics display
            if regression_results:
                metrics_div = html.Div([
                    html.H3("Regression Statistics"),
                    html.P(f"R-squared: {regression_results['r2']:.4f}"),
                    html.P(f"Slope: {regression_results['slope']:.4f}"),
                    html.P(f"Intercept: {regression_results['intercept']:.4f}"),
                    html.P(f"Sample Size: {len(regression_results['data'])}"),
                    html.H4("Interpretation"),
                    html.P(f"Storage Percentile Range: {filtered_data['storage_percentile'].min():.1f}% - {filtered_data['storage_percentile'].max():.1f}%"),
                    html.P(f"Mean Storage Percentile: {filtered_data['storage_percentile'].mean():.1f}%")
                ])
            else:
                metrics_div = html.P("No valid data for analysis")
            
            return [
                dcc.Graph(figure=scatter_fig),
                dcc.Graph(figure=timeseries_fig), 
                dcc.Graph(figure=residuals_fig),
                metrics_div
            ]
        
        # Register the callbacks with the app
        self.registry.register_callbacks(app)