import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html
from datetime import datetime, timedelta
import numpy as np
from plotly.subplots import make_subplots
class MarketChart:
    """
    A reusable market chart component that can display line charts or candlestick charts.
    
    Parameters:
    - chart_id: Unique identifier for the chart
    - title: Chart title
    - market_data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] or ['Date', 'Price']
    - chart_type: 'line' or 'candlestick'
    - width: Width as percentage string (e.g., '49%') or pixels (e.g., '400px')
    - height: Height in pixels (default: 300)
    - float_position: 'left', 'right', or None for CSS float positioning
    - margin: Margin string (e.g., '5px 1%')
    """
    
    def __init__(self, chart_id, title="Market Chart", market_data=None, chart_type='line', 
                 width='49%', height=300, float_position='left', margin='5px 1%'):
        self.chart_id = chart_id
        self.title = title
        self.market_data = market_data
        self.chart_type = chart_type
        self.width = width
        self.height = height
        self.float_position = float_position
        self.margin = margin
        
    def _create_line_chart(self):
        """Create a line chart from market data"""
        fig = go.Figure()
        
        if self.market_data is not None and not self.market_data.empty:
            # Check if we have OHLC data or just price data
            if 'Close' in self.market_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.market_data['Date'],
                    y=self.market_data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#1f77b4', width=2)
                ))
            elif 'Price' in self.market_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.market_data['Date'],
                    y=self.market_data['Price'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#1f77b4', width=2)
                ))
            else:
                # Fallback to first numeric column
                numeric_cols = self.market_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig.add_trace(go.Scatter(
                        x=self.market_data['Date'],
                        y=self.market_data[numeric_cols[0]],
                        mode='lines',
                        name=numeric_cols[0],
                        line=dict(color='#1f77b4', width=2)
                    ))
        else:
            # Empty placeholder
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))
            
        fig.update_layout(
            title=self.title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=self.height,
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_candlestick_chart(self):
        """Create a candlestick chart from market data"""
        fig = go.Figure()
        
        if (self.market_data is not None and not self.market_data.empty and 
            all(col in self.market_data.columns for col in ['Open', 'High', 'Low', 'Close'])):
            
            fig.add_trace(go.Candlestick(
                x=self.market_data['Date'],
                open=self.market_data['Open'],
                high=self.market_data['High'],
                low=self.market_data['Low'],
                close=self.market_data['Close'],
                name='OHLC'
            ))
        else:
            # Fallback to empty chart with message
            fig.add_annotation(
                text="Candlestick chart requires Open, High, Low, Close data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="gray")
            )
            
        fig.update_layout(
            title=self.title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=self.height,
            xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def get_chart_figure(self):
        """Get the appropriate chart figure based on chart_type"""
        if self.chart_type == 'candlestick':
            return self._create_candlestick_chart()
        else:
            return self._create_line_chart()
    
    def get_chart_component(self):
        """Get the complete Dash component with styling"""
        # Determine CSS float style
        float_style = {}
        if self.float_position:
            float_style['float'] = self.float_position
        
        style = {
            'width': self.width,
            'border': '2px solid #34495e',
            'padding': '10px',
            'margin': self.margin,
            **float_style
        }
        
        return html.Div([
            html.H4(self.title, style={'text-align': 'center', 'margin': '10px 0'}),
            dcc.Graph(
                id=self.chart_id,
                figure=self.get_chart_figure()
            )
        ], style=style)
    
    def update_data(self, new_data):
        """Update the chart with new market data"""
        self.market_data = new_data
        return self.get_chart_figure()
    
    def change_chart_type(self, new_type):
        """Change between line and candlestick chart types"""
        self.chart_type = new_type
        return self.get_chart_figure()


class FundamentalChart:
    """
    A reusable supply and demand chart component that reads data from HDF5 files.
    
    Parameters:
    - chart_id: Unique identifier for the chart
    - title: Chart title
    - hdf_file_path: Path to the HDF5 file
    - hdf_key: Key to access the dataframe in the HDF5 file
    - y_column: Column name for y-axis data (default: 'Value')
    - x_column: Column name for x-axis data (default: 'Date')
    - chart_type: 'bar', 'line', or 'area'
    - width: Width as percentage string (e.g., '49%') or pixels (e.g., '400px')
    - height: Height in pixels (default: 300)
    - float_position: 'left', 'right', or None for CSS float positioning
    - margin: Margin string (e.g., '5px 1%')
    """
    
    def __init__(self, chart_id, TableClient, starting_key=None, title="Supply/Demand Chart", x_column='Date', chart_type='bar',
                 width='49%',line_color='blue', height=300, float_position='left', margin='5px 1%'):
        self.chart_id = chart_id
        self.line_color = line_color
        self.title = title
        self.table_client = TableClient
        self.hdf_file_path = TableClient.table_db
        self.hdf_key = starting_key

        self.chart_type = chart_type
        self.width = width
        self.height = height
        self.float_position = float_position
        self.margin = margin
        self.data = None

        # Load data if file path and key are provided
        if self.hdf_file_path and self.hdf_key:
            self._load_data()
        self.y_column = self.data.columns[0]
        self.x_column = x_column


    def _load_data(self):
        """Load data from HDF5 file using TableClient"""
        try:
            self.data = self.table_client.get_key(self.hdf_key, use_prefix=True)
            print(f"Successfully loaded data for key: {self.hdf_key}")
            if len(self.data.columns) == 1:
                self.y_column =self.data.columns[0]
                self.title = self.y_column

        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None
    
    def _create_bar_chart(self):
        """Create a bar chart from the data"""
        fig = go.Figure()
        
        if self.data is not None and not self.data.empty:
            if self.y_column in self.data.columns:
                x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index
                
                fig.add_trace(go.Bar(
                    x=x_data,
                    y=self.data[self.y_column],
                    name=self.y_column,
                    marker_color=f'{self.line_color}'
                ))
            else:
                # Show available columns if y_column not found
                available_cols = list(self.data.columns)
                fig.add_annotation(
                    text=f"Column '{self.y_column}' not found. Available: {available_cols}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=10, color="red")
                )
        else:
            # Placeholder data
            fig.add_trace(go.Bar(x=[], y=[], name='No Data'))
            
        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title=self.y_column,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_line_chart(self):
        """Create a line chart from the data"""
        fig = go.Figure()
        
        if self.data is not None and not self.data.empty:
            if self.y_column in self.data.columns:
                x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=self.data[self.y_column],
                    mode='lines+markers',
                    name=self.y_column,
                    line=dict(color=f'{self.line_color}', width=2)
                ))
            else:
                available_cols = list(self.data.columns)
                fig.add_annotation(
                    text=f"Column '{self.y_column}' not found. Available: {available_cols}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=10, color="red")
                )
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))
            
        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title=self.y_column,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_area_chart(self):
        """Create an area chart from the data"""
        fig = go.Figure()
        
        if self.data is not None and not self.data.empty:
            if self.y_column in self.data.columns:
                x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=self.data[self.y_column],
                    mode='lines',
                    name=self.y_column,
                    fill='tozeroy',
                    fillcolor='rgba(46, 134, 171, 0.3)',
                    line=dict(color=f'{self.line_color}', width=2)
                ))
            else:
                available_cols = list(self.data.columns)
                fig.add_annotation(
                    text=f"Column '{self.y_column}' not found. Available: {available_cols}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=10, color="red")
                )
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))
            
        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title=self.y_column,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def get_chart_figure(self):
        """Get the appropriate chart figure based on chart_type"""
        if self.chart_type == 'line':
            return self._create_line_chart()
        elif self.chart_type == 'area':
            return self._create_area_chart()
        else:
            return self._create_bar_chart()
    
    def get_chart_component(self):
        """Get the complete Dash component with styling"""
        # Determine CSS float style
        float_style = {}
        if self.float_position:
            float_style['float'] = self.float_position
        
        style = {
            'width': self.width,
            'border': '2px solid #34495e',
            'padding': '10px',
            'margin': self.margin,
            **float_style
        }
        
        return html.Div([
            html.H4(self.title, style={'text-align': 'center', 'margin': '10px 0'}),
            dcc.Graph(
                id=self.chart_id,
                figure=self.get_chart_figure()
            )
        ], style=style)

    def update_data_source(self, new_hdf_file_path, new_hdf_key):
        """Update the data source and reload data"""
        self.hdf_file_path = new_hdf_file_path
        self.hdf_key = new_hdf_key
        self._load_data()
        return self.get_chart_figure()

    def change_y_column(self, new_y_column):
        """Change the y-axis column"""
        self.y_column = new_y_column
        return self.get_chart_figure()
    
    def change_chart_type(self, new_chart_type):
        """Change the chart type"""
        self.chart_type = new_chart_type
        return self.get_chart_figure()
    
    def get_available_columns(self):
        """Get list of available columns in the loaded data"""
        if self.data is not None:
            return list(self.data.columns)
        return []
    
    def get_data_info(self):
        """Get basic information about the loaded data"""
        if self.data is not None:
            return {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'sample': self.data.head().to_dict()
            }
        return None


class MultiChart(FundamentalChart):
    """
    A multi-series chart component that can plot multiple data columns with different scales.

    Parameters:
    - chart_id: Unique identifier for the chart
    - TableClient: Client for accessing table data
    - keys: List of keys to fetch data from
    - title: Chart title
    - x_column: Column name for x-axis data (default: 'Date')
    - chart_type: 'bar', 'line', or 'area'
    - width: Width as percentage string or pixels
    - line_colors: List of colors for each series (optional)
    - height: Height in pixels (default: 300)
    - float_position: 'left', 'right', or None for CSS float positioning
    - margin: Margin string
    - dual_y: Whether to use dual y-axes (default: False)
    - secondary_y_columns: List of columns to plot on secondary y-axis
    """

    def __init__(self, chart_id, TableClient, keys, title="Multi-Series Chart", x_column='Date',
                 chart_type='line', width='49%', line_colors=None, height=300,
                 float_position='left', margin='5px 1%', dual_y=False, secondary_y_columns=None):

        # Initialize parent class without starting_key
        super().__init__(chart_id, TableClient, title=title, x_column=x_column,
                         chart_type=chart_type, width=width, height=height,
                         float_position=float_position, margin=margin)

        self.keys = keys
        self.line_colors = line_colors or ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        self.dual_y = dual_y
        self.secondary_y_columns = secondary_y_columns or []
        self.selected_columns = []  # Columns to plot
        self.data = None

        # Load data from multiple keys
        self._load_multi_data()

        # Set default selected columns (all numeric columns)
        if self.data is not None:
            self.selected_columns = [col for col in self.data.columns
                                     if col != self.x_column and pd.api.types.is_numeric_dtype(self.data[col])]

    def _load_multi_data(self):
        """Load data from multiple keys using TableClient"""
        try:
            self.data = self.table_client.get_keys(self.keys)
            print(f"Successfully loaded data for keys: {self.keys}")
            print(f"Data shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
        except Exception as e:
            print(f"Error loading multi-key data: {e}")
            self.data = None

    def set_selected_columns(self, columns):
        """Set which columns to plot"""
        if self.data is not None:
            available_cols = [col for col in self.data.columns if col != self.x_column]
            self.selected_columns = [col for col in columns if col in available_cols]
        return self

    def set_secondary_y_columns(self, columns):
        """Set which columns should be plotted on secondary y-axis"""
        self.secondary_y_columns = columns
        return self

    def enable_dual_y(self, enable=True):
        """Enable or disable dual y-axis"""
        self.dual_y = enable
        return self

    def _create_multi_line(self):
        """Create a multi-line chart with optional dual y-axis"""
        if self.dual_y and self.secondary_y_columns:
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()

        if self.data is not None and not self.data.empty and self.selected_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.selected_columns):
                if col in self.data.columns:
                    color = self.line_colors[i % len(self.line_colors)]

                    trace = go.Scatter(
                        x=x_data,
                        y=self.data[col],
                        mode='lines+markers',
                        name=col,
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    )

                    # Add to secondary y-axis if specified
                    if self.dual_y and col in self.secondary_y_columns:
                        fig.add_trace(trace, secondary_y=True)
                    else:
                        if self.dual_y:
                            fig.add_trace(trace, secondary_y=False)
                        else:
                            fig.add_trace(trace)
        else:
            # No data or columns selected
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))

        # Update layout
        if self.dual_y and self.secondary_y_columns:
            # Set y-axes titles
            primary_cols = [col for col in self.selected_columns if col not in self.secondary_y_columns]
            secondary_cols = [col for col in self.selected_columns if col in self.secondary_y_columns]

            fig.update_yaxes(title_text=f"Primary: {', '.join(primary_cols)}", secondary_y=False)
            fig.update_yaxes(title_text=f"Secondary: {', '.join(secondary_cols)}", secondary_y=True)
            fig.update_xaxes(title_text=self.x_column)
        else:
            fig.update_layout(
                xaxis_title=self.x_column,
                yaxis_title="Value"
            )

        fig.update_layout(
            title=self.title,
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='x unified'
        )

        return fig

    def _create_multi_bar(self):
        """Create a multi-series bar chart"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty and self.selected_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.selected_columns):
                if col in self.data.columns:
                    color = self.line_colors[i % len(self.line_colors)]

                    fig.add_trace(go.Bar(
                        x=x_data,
                        y=self.data[col],
                        name=col,
                        marker_color=color,
                        opacity=0.8
                    ))
        else:
            fig.add_trace(go.Bar(x=[], y=[], name='No Data'))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title="Value",
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            barmode='group'  # Grouped bars
        )

        return fig

    def _create_multi_area(self):
        """Create a multi-series area chart (stacked)"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty and self.selected_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.selected_columns):
                if col in self.data.columns:
                    color = self.line_colors[i % len(self.line_colors)]

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=self.data[col],
                        mode='lines',
                        name=col,
                        fill='tonexty' if i > 0 else 'tozeroy',
                        fillcolor=f'rgba({self._hex_to_rgb(color)}, 0.3)',
                        line=dict(color=color, width=2)
                    ))
        else:
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='No Data'))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.x_column,
            yaxis_title="Value",
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    def _create_normalized_chart(self):
        """Create a chart with normalized data (0-100 scale)"""
        fig = go.Figure()

        if self.data is not None and not self.data.empty and self.selected_columns:
            x_data = self.data[self.x_column] if self.x_column in self.data.columns else self.data.index

            for i, col in enumerate(self.selected_columns):
                if col in self.data.columns:
                    # Normalize data to 0-100 scale
                    col_data = self.data[col]
                    min_val = col_data.min()
                    max_val = col_data.max()
                    normalized_data = ((col_data - min_val) / (max_val - min_val)) * 100

                    color = self.line_colors[i % len(self.line_colors)]

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=normalized_data,
                        mode='lines+markers',
                        name=f"{col} (Normalized)",
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ))

        fig.update_layout(
            title=f"{self.title} - Normalized (0-100)",
            xaxis_title=self.x_column,
            yaxis_title="Normalized Value (0-100)",
            height=self.height,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='x unified'
        )

        return fig

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB values"""
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"

    def get_chart_figure(self, normalized=False):
        """Get the appropriate chart figure based on chart_type"""
        if normalized:
            return self._create_normalized_chart()

        if self.chart_type == 'line':
            return self._create_multi_line()
        elif self.chart_type == 'area':
            return self._create_multi_area()
        elif self.chart_type == 'bar':
            return self._create_multi_bar()
        else:
            return self._create_multi_line()

    def get_chart_component(self, normalized=False):
        """Get the complete Dash component with styling"""
        float_style = {}
        if self.float_position:
            float_style['float'] = self.float_position

        style = {
            'width': self.width,
            'border': '2px solid #34495e',
            'padding': '10px',
            'margin': self.margin,
            **float_style
        }

        # Create column selector dropdown
        column_options = []
        if self.data is not None:
            column_options = [{'label': col, 'value': col}
                              for col in self.data.columns
                              if col != self.x_column and pd.api.types.is_numeric_dtype(self.data[col])]

        return html.Div([
            html.H4(self.title, style={'text-align': 'center', 'margin': '10px 0'}),

            # Column selector
            html.Div([
                html.Label("Select Columns to Plot:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id=f'{self.chart_id}_column_selector',
                    options=column_options,
                    value=self.selected_columns,
                    multi=True,
                    style={'margin-bottom': '10px'}
                )
            ]),

            # Chart type selector
            html.Div([
                html.Label("Chart Type:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                dcc.RadioItems(
                    id=f'{self.chart_id}_chart_type',
                    options=[
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Bar', 'value': 'bar'},
                        {'label': 'Area', 'value': 'area'}
                    ],
                    value=self.chart_type,
                    inline=True,
                    style={'margin-bottom': '10px'}
                )
            ]),

            # Options
            html.Div([
                dcc.Checklist(
                    id=f'{self.chart_id}_options',
                    options=[
                        {'label': 'Dual Y-Axis', 'value': 'dual_y'},
                        {'label': 'Normalized View', 'value': 'normalized'}
                    ],
                    value=['dual_y'] if self.dual_y else [],
                    inline=True,
                    style={'margin-bottom': '10px'}
                )
            ]),

            # Chart
            dcc.Graph(
                id=self.chart_id,
                figure=self.get_chart_figure(normalized=normalized)
            )
        ], style=style)

    def update_selected_columns(self, selected_columns):
        """Update selected columns and return new figure"""
        self.selected_columns = selected_columns
        return self.get_chart_figure()

    def update_chart_type(self, chart_type):
        """Update chart type and return new figure"""
        self.chart_type = chart_type
        return self.get_chart_figure()

    def update_options(self, options):
        """Update chart options and return new figure"""
        self.dual_y = 'dual_y' in options
        normalized = 'normalized' in options
        return self.get_chart_figure(normalized=normalized)

    def get_data_summary(self):
        """Get summary statistics for all numeric columns"""
        if self.data is not None:
            numeric_cols = [col for col in self.data.columns
                            if pd.api.types.is_numeric_dtype(self.data[col])]
            return self.data[numeric_cols].describe()
        return None

    def get_correlation_matrix(self):
        """Get correlation matrix for selected columns"""
        if self.data is not None and self.selected_columns:
            return self.data[self.selected_columns].corr()
        return None


# Example usage and testing functions
def create_sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Create OHLC data
    base_price = 75
    prices = []
    current_price = base_price
    
    for _ in range(100):
        daily_change = np.random.normal(0, 2)
        current_price += daily_change
        
        # Generate OHLC from current price
        open_price = current_price + np.random.normal(0, 0.5)
        high_price = max(open_price, current_price) + abs(np.random.normal(0, 1))
        low_price = min(open_price, current_price) - abs(np.random.normal(0, 1))
        close_price = current_price + np.random.normal(0, 0.5)
        
        prices.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Price': close_price  # Also include simple price column
        })
    
    df = pd.DataFrame(prices)
    df['Date'] = dates
    df['Volume'] = np.random.randint(10000, 100000, size=100)
    
    return df

def create_sample_supply_demand_data():
    """Create sample supply/demand data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    
    data = {
        'Date': dates,
        'Value': np.random.randint(80, 120, 12),
        'Supply': np.random.randint(75, 115, 12),
        'Demand': np.random.randint(85, 125, 12),
        'Production': np.random.randint(70, 110, 12)
    }
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Test MarketChart
    sample_market_data = create_sample_market_data()
    
    market_chart = MarketChart(
        chart_id='test-market-chart',
        title='Test Market Chart',
        market_data=sample_market_data,
        chart_type='line',
        width='600px',
        height=400
    )
    
    print("Market Chart created successfully")
    print(f"Chart has {len(sample_market_data)} data points")
    
    # Test FundamentalChart (without actual HDF5 file)
    supply_chart = FundamentalChart(
        chart_id='test-supply-chart',
        title='Test Supply/Demand Chart',
        y_column='Value',
        chart_type='bar',
        width='600px',
        height=400
    )
    
    # Manually set data for testing
    supply_chart.data = create_sample_supply_demand_data()
    
    print("Supply/Demand Chart created successfully")
    print(f"Available columns: {supply_chart.get_available_columns()}")
