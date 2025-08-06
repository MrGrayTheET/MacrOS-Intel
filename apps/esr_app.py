#!/usr/bin/env python3
"""
ESR (Export Sales Reporting) Analysis Application
A comprehensive dashboard for analyzing U.S. agricultural export sales data.

This application provides multiple views for ESR data analysis:
1. Sales Trends Analysis
2. Country Performance Analysis
3. Commitment Analysis
4. Comparative Analysis
"""


# Import your framework components
from data.data_tables import ESRTableClient
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
# Import your framework components
from components.frames import FundamentalFrame
# Import the new enhanced menu components
from components.frames import FlexibleMenu, EnhancedFrameGrid
from callbacks.agricultural import create_esr_chart_update_functions


class ESRAnalysisApp:
    """
    Main ESR Analysis Application class.
    Creates and manages the multi-page dashboard.
    """

    def __init__(self):
        """Initialize the ESR Analysis App."""
        # Initialize table client
        self.table_client = ESRTableClient()


        # App configuration
        self.app_config = {
            'title': 'ESR Analysis Dashboard',
            'theme': 'dark',
            'update_interval': 300000,  # 5 minutes
        }

        # Store data locally for efficient filtering
        self.current_data = {}  # Will store data by page

        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.title = self.app_config['title']

        # Create pages
        self.pages = self._create_pages()

        # Setup layout and callbacks
        self._setup_layout()
        self._register_callbacks()

    def _create_mock_table_client(self):
        """Create a mock table client with sample ESR data for demonstration."""

        # This would be replaced with actual table client initialization
        class MockESRTableClient:
            def __init__(self):
                # Load the sample data
                sample_data = pd.read_csv('esr_weekly_example.csv')
                sample_data['weekEndingDate'] = pd.to_datetime(sample_data['weekEndingDate'])
                self.sample_data = sample_data

            def get_key(self, key):
                # Return sample data for any key request
                return self.sample_data.copy()

            def available_keys(self):
                return ['/cattle/exports/2024', '/corn/exports/2024', '/wheat/exports/2024']

            def get_esr_data(self, commodity, year=None, country=None, start_date=None, end_date=None):
                data = self.sample_data.copy()

                if country:
                    data = data[data['country'].str.contains(country, case=False, na=False)]
                if start_date:
                    data = data[data['weekEndingDate'] >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data['weekEndingDate'] <= pd.to_datetime(end_date)]

                return data

            def get_available_commodities(self):
                return ['cattle', 'corn', 'wheat', 'soybeans']

            def get_available_years(self, commodity):
                return [2023, 2024]

            def __getitem__(self, key):
                return self.get_key(key)

        return MockESRTableClient()

    def _create_pages(self):
        """Create different analysis pages."""
        pages = {}

        # Page 1: Sales Trends Analysis
        pages['sales_trends'] = self._create_sales_trends_page()

        # Page 2: Country Performance Analysis
        pages['country_analysis'] = self._create_country_analysis_page()

        # Page 3: Commitment Analysis
        pages['commitment_analysis'] = self._create_commitment_analysis_page()

        # Page 4: Comparative Analysis
        pages['comparative_analysis'] = self._create_comparative_analysis_page()

        return pages

    def _create_sales_trends_page(self):
        """Create the sales trends analysis page with country-colored plots."""

        # Chart configurations for sales trends
        chart_configs = [
            {
                'title': 'Weekly Export Trends',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'weeklyExports',
                'x_column': 'weekEndingDate',
                'width': '100%',
                'height': 400,
                'line_color': '#1f77b4'
            },
            {
                'title': 'Outstanding Sales Analysis',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'outstandingSales',
                'x_column': 'weekEndingDate',
                'width': '100%',
                'height': 400,
                'line_color': '#ff7f0e'
            },
            {
                'title': 'Gross New Sales Trends',
                'chart_type': 'bar',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'grossNewSales',
                'x_column': 'weekEndingDate',
                'width': '100%',
                'height': 400,
                'line_color': '#2ca02c'
            }
        ]

        # Create fundamental frame for sales trends
        sales_frame = FundamentalFrame(
            table_client=self.table_client,
            chart_configs=chart_configs,
            layout="horizontal",
            div_prefix="sales_trends",
            width="100%",
            height="1200px"
        )

        # Create Sales Trends specific menu
        sales_menu = FlexibleMenu('sales_trends_menu', position='right', width='300px', title='Sales Trends Controls')

        # Commodity selector
        sales_menu.add_dropdown('commodity', 'Commodity', [
            {'label': 'Cattle', 'value': 'cattle'},
            {'label': 'Corn', 'value': 'corn'},
            {'label': 'Wheat', 'value': 'wheat'},
            {'label': 'Soybeans', 'value': 'soybeans'}
        ], value='cattle')

        # Country filter checklist
        sales_menu.add_checklist('countries', 'Countries to Display', [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Mexico', 'value': 'Mexico'},
            {'label': 'Canada', 'value': 'Canada'},
            {'label': 'Taiwan', 'value': 'Taiwan'}
        ], value=['Korea, South', 'Japan', 'China'])

        # Apply button
        sales_menu.add_button('apply', 'Apply Changes')

        # Create enhanced grid with sales-specific menu
        page_grid = EnhancedFrameGrid(
            frames=[sales_frame],
            flexible_menu=sales_menu
        )

        return page_grid

    def _create_country_analysis_page(self):
        """Create the country performance analysis page with multi-year data and single country selection."""

        # Chart configurations for country analysis (multi-year, single country)
        chart_configs = [
            {
                'title': 'Country Export Performance (5-Year)',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_columns': 'weeklyExports',
                'x_column': 'weekEndingDate',
                'width': '100%',
                'height': 450,
                'dual_y': False
            },
            {
                'title': 'Outstanding Sales Trend (5-Year)',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_columns': 'outstandingSales',
                'x_column': 'weekEndingDate',
                'width': '100%',
                'height': 450,
                'dual_y': False
            }
        ]

        country_frame = FundamentalFrame(
            table_client=self.table_client,
            chart_configs=chart_configs,
            layout="horizontal",
            div_prefix="country_analysis",
            width="100%",
            height="900px"
        )

        # Create Country Analysis specific menu
        country_menu = FlexibleMenu('country_analysis_menu', position='right', width='300px',
                                    title='Country Analysis Controls')

        # Commodity selector
        country_menu.add_dropdown('commodity', 'Commodity', [
            {'label': 'Cattle', 'value': 'cattle'},
            {'label': 'Corn', 'value': 'corn'},
            {'label': 'Wheat', 'value': 'wheat'},
            {'label': 'Soybeans', 'value': 'soybeans'}
        ], value='cattle')

        # Single country selector
        country_menu.add_dropdown('country', 'Select Country', [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Mexico', 'value': 'Mexico'},
            {'label': 'Canada', 'value': 'Canada'},
            {'label': 'Taiwan', 'value': 'Taiwan'}
        ], value='Korea, South')

        # Apply button
        country_menu.add_button('apply', 'Apply Changes')

        # Create enhanced grid with country-specific menu
        page_grid = EnhancedFrameGrid(
            frames=[country_frame],
            flexible_menu=country_menu
        )

        return page_grid

    def _create_commitment_analysis_page(self):
        """Create the commitment analysis page with commodity, year, and country filtering."""

        chart_configs = [
            {
                'title': 'Current MY Total Commitment',
                'chart_type': 'area',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'currentMYTotalCommitment',
                'x_column': 'weekEndingDate',
                'width': '49%',
                'height': 400,
                'line_color': '#9467bd',
                'float_position': 'left'
            },
            {
                'title': 'Current MY Net Sales',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'currentMYNetSales',
                'x_column': 'weekEndingDate',
                'width': '49%',
                'height': 400,
                'line_color': '#8c564b',
                'float_position': 'right'
            },
            {
                'title': 'Outstanding Sales',
                'chart_type': 'bar',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'outstandingSales',
                'x_column': 'weekEndingDate',
                'width': '49%',
                'height': 400,
                'line_color': '#e377c2',
                'float_position': 'left'
            },
            {
                'title': 'Next MY Net Sales',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'nextMYNetSales',
                'x_column': 'weekEndingDate',
                'width': '49%',
                'height': 400,
                'line_color': '#7f7f7f',
                'float_position': 'right'
            }
        ]

        commitment_frame = FundamentalFrame(
            table_client=self.table_client,
            chart_configs=chart_configs,
            layout="vertical",
            div_prefix="commitment_analysis",
            width="100%",
            height="850px"
        )

        # Create Commitment Analysis specific menu
        commitment_menu = FlexibleMenu('commitment_analysis_menu', position='right', width='300px',
                                       title='Commitment Analysis Controls')

        # Commodity selector
        commitment_menu.add_dropdown('commodity', 'Commodity', [
            {'label': 'Cattle', 'value': 'cattle'},
            {'label': 'Corn', 'value': 'corn'},
            {'label': 'Wheat', 'value': 'wheat'},
            {'label': 'Soybeans', 'value': 'soybeans'}
        ], value='cattle')

        # Year selector
        current_year = pd.Timestamp.now().year
        commitment_menu.add_dropdown('year', 'Marketing Year', [
            {'label': str(year), 'value': year}
            for year in range(current_year - 2, current_year + 1)
        ], value=current_year)

        # Country filter checklist
        commitment_menu.add_checklist('countries', 'Countries to Display', [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Mexico', 'value': 'Mexico'},
            {'label': 'Canada', 'value': 'Canada'},
            {'label': 'Taiwan', 'value': 'Taiwan'}
        ], value=['Korea, South', 'Japan', 'China'])

        # Apply button
        commitment_menu.add_button('apply', 'Apply Changes')

        # Create enhanced grid with commitment-specific menu
        page_grid = EnhancedFrameGrid(
            frames=[commitment_frame],
            flexible_menu=commitment_menu
        )

        return page_grid

    def _create_comparative_analysis_page(self):
        """Create the comparative analysis page with separate commodity selection."""

        # Create two frames for comparison
        frame1_configs = [
            {
                'title': 'Commodity A - Export Metrics',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'weeklyExports',
                'x_column': 'weekEndingDate',
                'width': '100%',
                'height': 350,
                'line_color': '#1f77b4'
            }
        ]

        frame2_configs = [
            {
                'title': 'Commodity B - Export Metrics',
                'chart_type': 'line',
                'starting_key': 'cattle/exports/2024',
                'y_column': 'weeklyExports',
                'x_column': 'weekEndingDate',
                'width': '100%',
                'height': 350,
                'line_color': '#ff7f0e'
            }
        ]

        comparison_frame1 = FundamentalFrame(
            table_client=self.table_client,
            chart_configs=frame1_configs,
            div_prefix="comparison_frame1"
        )

        comparison_frame2 = FundamentalFrame(
            table_client=self.table_client,
            chart_configs=frame2_configs,
            div_prefix="comparison_frame2"
        )

        # Create Comparative Analysis specific menu with separate commodity controls
        comparative_menu = FlexibleMenu('comparative_analysis_menu', position='right', width='320px',
                                        title='Comparative Analysis Controls')

        # Commodity A selector
        comparative_menu.add_dropdown('commodity_a', 'Commodity A', [
            {'label': 'Cattle', 'value': 'cattle'},
            {'label': 'Corn', 'value': 'corn'},
            {'label': 'Wheat', 'value': 'wheat'},
            {'label': 'Soybeans', 'value': 'soybeans'}
        ], value='cattle')

        # Commodity B selector
        comparative_menu.add_dropdown('commodity_b', 'Commodity B', [
            {'label': 'Cattle', 'value': 'cattle'},
            {'label': 'Corn', 'value': 'corn'},
            {'label': 'Wheat', 'value': 'wheat'},
            {'label': 'Soybeans', 'value': 'soybeans'}
        ], value='corn')

        # Year selector (applies to both commodities)
        current_year = pd.Timestamp.now().year
        comparative_menu.add_dropdown('year', 'Marketing Year', [
            {'label': str(year), 'value': year}
            for year in range(current_year - 2, current_year + 1)
        ], value=current_year)

        # Metric selector (applies to both commodities)
        comparative_menu.add_dropdown('metric', 'Metric to Compare', [
            {'label': 'Weekly Exports', 'value': 'weeklyExports'},
            {'label': 'Outstanding Sales', 'value': 'outstandingSales'},
            {'label': 'Gross New Sales', 'value': 'grossNewSales'},
            {'label': 'Current MY Net Sales', 'value': 'currentMYNetSales'}
        ], value='weeklyExports')

        # Country filter checklist (applies to both commodities)
        comparative_menu.add_checklist('countries', 'Countries to Display', [
            {'label': 'Korea, South', 'value': 'Korea, South'},
            {'label': 'Japan', 'value': 'Japan'},
            {'label': 'China', 'value': 'China'},
            {'label': 'Mexico', 'value': 'Mexico'},
            {'label': 'Canada', 'value': 'Canada'},
            {'label': 'Taiwan', 'value': 'Taiwan'}
        ], value=['Korea, South', 'Japan', 'China'])

        # Apply button
        comparative_menu.add_button('apply', 'Apply Changes')

        # Create enhanced grid with comparative-specific menu
        page_grid = EnhancedFrameGrid(
            frames=[comparison_frame1, comparison_frame2],
            flexible_menu=comparative_menu
        )

        return page_grid

    def _setup_layout(self):
        """Setup the main application layout with navigation."""

        # Main navigation
        nav_style = {
            'backgroundColor': '#222222',
            'padding': '10px 20px',
            'marginBottom': '20px',
            'borderRadius': '5px'
        }

        button_style = {
            'backgroundColor': '#333333',
            'color': '#e8e8e8',
            'border': '1px solid #444444',
            'padding': '10px 20px',
            'margin': '0 10px',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'fontSize': '14px'
        }

        active_button_style = {
            **button_style,
            'backgroundColor': '#4CAF50',
            'borderColor': '#4CAF50'
        }

        navigation = html.Div([
            html.H1("ESR Analysis Dashboard",
                    style={'color': '#e8e8e8', 'textAlign': 'center', 'marginBottom': '20px'}),

            html.Div([
                html.Button("Sales Trends", id="nav-sales-trends", n_clicks=0, style=active_button_style),
                html.Button("Country Analysis", id="nav-country-analysis", n_clicks=0, style=button_style),
                html.Button("Commitment Analysis", id="nav-commitment-analysis", n_clicks=0, style=button_style),
                html.Button("Comparative Analysis", id="nav-comparative-analysis", n_clicks=0, style=button_style),
            ], style={'textAlign': 'center'})
        ], style=nav_style)

        # Page content container
        page_content = html.Div(id="page-content")

        # Main layout
        self.app.layout = html.Div([
            # Store for current page
            dcc.Store(id="current-page", data="sales_trends"),

            # Navigation
            navigation,

            # Page content
            page_content,

            # Global styles
            html.Div(id="global-styles")
        ], style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'})

    def _register_callbacks(self):
        """Register all application callbacks."""

        # Navigation callback
        @self.app.callback(
            [Output("page-content", "children"),
             Output("current-page", "data"),
             Output("nav-sales-trends", "style"),
             Output("nav-country-analysis", "style"),
             Output("nav-commitment-analysis", "style"),
             Output("nav-comparative-analysis", "style")],
            [Input("nav-sales-trends", "n_clicks"),
             Input("nav-country-analysis", "n_clicks"),
             Input("nav-commitment-analysis", "n_clicks"),
             Input("nav-comparative-analysis", "n_clicks")],
            [State("current-page", "data")]
        )
        def navigate_pages(sales_clicks, country_clicks, commitment_clicks, comparative_clicks, current_page):
            """Handle page navigation."""
            ctx = callback_context

            # Default styles
            default_style = {
                'backgroundColor': '#333333',
                'color': '#e8e8e8',
                'border': '1px solid #444444',
                'padding': '10px 20px',
                'margin': '0 10px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': '14px'
            }

            active_style = {
                **default_style,
                'backgroundColor': '#4CAF50',
                'borderColor': '#4CAF50'
            }

            # Default to sales trends
            page = "sales_trends"
            styles = [active_style, default_style, default_style, default_style]

            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                if button_id == "nav-country-analysis":
                    page = "country_analysis"
                    styles = [default_style, active_style, default_style, default_style]
                elif button_id == "nav-commitment-analysis":
                    page = "commitment_analysis"
                    styles = [default_style, default_style, active_style, default_style]
                elif button_id == "nav-comparative-analysis":
                    page = "comparative_analysis"
                    styles = [default_style, default_style, default_style, active_style]

            # Get page content using enhanced grid layout
            page_layout = self.pages[page].generate_layout_with_menu(title="")

            return page_layout, page, styles[0], styles[1], styles[2], styles[3]

        # Register callbacks for each page's enhanced grid
        update_functions = create_esr_chart_update_functions(self)
        for page_name, page_grid in self.pages.items():
            # Use the enhanced grid's menu callbacks with simple ESR chart update function
            page_grid.create_menu_callbacks(self.app, update_functions[page_name])

    def create_additional_callbacks(self):
        """Create additional custom callbacks for ESR-specific functionality."""

        # Example: Dynamic country filtering callback
        # This is now handled by the enhanced grid's menu callbacks
        # using the simple_esr_chart_update function
        pass

    def run(self, debug=True, host='127.0.0.1', port=8050):
        """Run the ESR Analysis application."""
        print(f"Starting ESR Analysis Dashboard at http://{host}:{port}")
        print("Available pages:")
        print("  - Sales Trends Analysis")
        print("  - Country Performance Analysis")
        print("  - Commitment Analysis")
        print("  - Comparative Analysis")

        self.app.run(debug=debug, host=host, port=port)


# Additional ESR-specific analysis functions
class ESRAnalytics:
    """
    Utility class for ESR-specific calculations and analytics.
    """

    @staticmethod
    def calculate_sales_velocity(data, window=4):
        """
        Calculate sales velocity (rate of change in weekly exports).

        Args:
            data: ESR DataFrame
            window: Rolling window for velocity calculation

        Returns:
            pd.Series: Sales velocity
        """
        data = data.sort_values('weekEndingDate')
        velocity = data['weeklyExports'].rolling(window=window).mean().diff()
        return velocity

    @staticmethod
    def calculate_commitment_ratio(data):
        """
        Calculate commitment ratio (outstanding sales / total commitment).

        Args:
            data: ESR DataFrame

        Returns:
            pd.Series: Commitment ratio
        """
        ratio = data['outstandingSales'] / data['currentMYTotalCommitment']
        return ratio.fillna(0)

    @staticmethod
    def identify_sales_peaks(data, prominence=0.1):
        """
        Identify peak sales periods using signal processing.

        Args:
            data: ESR DataFrame
            prominence: Minimum prominence for peak detection

        Returns:
            list: Indices of peak sales periods
        """
        from scipy.signal import find_peaks

        sales_data = data['weeklyExports'].values
        peaks, _ = find_peaks(sales_data, prominence=prominence * np.max(sales_data))
        return peaks

    @staticmethod
    def calculate_market_share(data, total_exports_column='weeklyExports'):
        """
        Calculate market share by country for a given time period.

        Args:
            data: ESR DataFrame
            total_exports_column: Column to use for market share calculation

        Returns:
            pd.DataFrame: Market share by country
        """
        country_totals = data.groupby('country')[total_exports_column].sum()
        total_exports = country_totals.sum()
        market_share = (country_totals / total_exports * 100).round(2)

        return market_share.reset_index().rename(columns={total_exports_column: 'market_share_pct'})

    @staticmethod
    def generate_sales_forecast(data, periods=8, method='linear'):
        """
        Generate simple sales forecast based on historical trends.

        Args:
            data: ESR DataFrame
            periods: Number of periods to forecast
            method: Forecasting method ('linear', 'exponential')

        Returns:
            pd.DataFrame: Forecast data
        """
        from sklearn.linear_model import LinearRegression

        # Prepare data
        data = data.sort_values('weekEndingDate').reset_index(drop=True)
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['weeklyExports'].values

        # Fit model
        if method == 'linear':
            model = LinearRegression()
            model.fit(X, y)

            # Generate forecast
            future_X = np.arange(len(data), len(data) + periods).reshape(-1, 1)
            forecast = model.predict(future_X)
        else:
            # Simple exponential smoothing
            alpha = 0.3
            forecast_values = []
            last_value = y[-1]

            for _ in range(periods):
                last_value = alpha * last_value + (1 - alpha) * np.mean(y[-4:])
                forecast_values.append(last_value)

            forecast = np.array(forecast_values)

        # Create forecast dataframe
        last_date = pd.to_datetime(data['weekEndingDate'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=periods, freq='W')

        forecast_df = pd.DataFrame({
            'weekEndingDate': future_dates,
            'weeklyExports_forecast': forecast,
            'forecast_type': method
        })

        return forecast_df


# Configuration factory for different commodity types
class ESRConfigFactory:
    """
    Factory class for creating ESR configurations for different commodities.
    """

    @staticmethod
    def get_commodity_config(commodity_type):
        """
        Get commodity-specific configuration.

        Args:
            commodity_type: Type of commodity ('grains', 'livestock', 'oilseeds')

        Returns:
            dict: Commodity-specific configuration
        """

        base_config = {
            'key_metrics': ['outstandingSales', 'grossNewSales', 'currentMYNetSales',
                            'currentMYTotalCommitment', 'nextMYOutstandingSales', 'nextMYNetSales'],
            'chart_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'date_column': 'weekEndingDate',
            'export_column': 'weeklyExports'
        }

        if commodity_type == 'grains':
            return {
                **base_config,
                'commodities': ['wheat', 'corn', 'rice', 'barley'],
                'seasonal_patterns': True,
                'harvest_months': [9, 10, 11, 12],  # Sep-Dec
                'units': 'metric tons',
                'typical_buyers': ['China', 'Japan', 'Mexico', 'South Korea']
            }

        elif commodity_type == 'livestock':
            return {
                **base_config,
                'commodities': ['cattle', 'pork', 'poultry'],
                'seasonal_patterns': False,
                'harvest_months': None,
                'units': 'metric tons',
                'typical_buyers': ['Japan', 'South Korea', 'Mexico', 'Canada']
            }

        elif commodity_type == 'oilseeds':
            return {
                **base_config,
                'commodities': ['soybeans', 'soybean_meal', 'soybean_oil', 'canola'],
                'seasonal_patterns': True,
                'harvest_months': [9, 10, 11],  # Sep-Nov
                'units': 'metric tons',
                'typical_buyers': ['China', 'EU', 'Mexico', 'Japan']
            }

        return base_config


# Main execution function
def create_esr_app():
    """
    Factory function to create and configure ESR application.

    Args:
        table_client_path: Path to table client data
        commodity_type: Type of commodity analysis ('grains', 'livestock', 'oilseeds')

    Returns:
        ESRAnalysisApp: Configured ESR application
    """
    # Create application with enhanced grid system
    app = ESRAnalysisApp()

    return app


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the ESR Analysis Application.
    """

    # Create livestock-focused ESR application
    esr_app = create_esr_app()

    # Add custom callbacks for advanced functionality
    # Run the application
    print("=" * 60)
    print("ESR (Export Sales Reporting) Analysis Dashboard")
    print("=" * 60)
    print("\nThis application provides comprehensive analysis of U.S. agricultural export sales data.")
    print("\nFeatures:")
    print("✓ Sales Trends Analysis - Track weekly export patterns")
    print("✓ Country Performance - Analyze exports by destination")
    print("✓ Commitment Analysis - Monitor sales commitments and outstanding orders")
    print("✓ Comparative Analysis - Compare different commodities/time periods")
    print("\nKey Metrics Analyzed:")
    print("• Outstanding Sales")
    print("• Gross New Sales")
    print("• Current MY Net Sales")
    print("• Current MY Total Commitment")
    print("• Next MY Outstanding Sales")
    print("• Next MY Net Sales")

    # Uncomment to run the app
    esr_app.run(debug=True, port=8050)


# Additional utility functions for ESR data processing
def load_esr_sample_data(file_path='esr_weekly_example.csv'):
    """
    Load and preprocess ESR sample data.

    Args:
        file_path: Path to ESR CSV file

    Returns:
        pd.DataFrame: Processed ESR data
    """
    try:
        data = pd.read_csv(file_path)

        # Data preprocessing
        data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])

        # Clean country names
        data['country'] = data['country'].str.strip()

        # Ensure numeric columns are properly typed
        numeric_columns = ['weeklyExports', 'accumulatedExports', 'outstandingSales',
                           'grossNewSales', 'currentMYNetSales', 'currentMYTotalCommitment',
                           'nextMYOutstandingSales', 'nextMYNetSales']

        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

        # Add derived columns
        data['sales_velocity'] = ESRAnalytics.calculate_sales_velocity(data)
        data['commitment_ratio'] = ESRAnalytics.calculate_commitment_ratio(data)

        # Add time-based features
        data['week_of_year'] = data['weekEndingDate'].dt.isocalendar().week
        data['month'] = data['weekEndingDate'].dt.month
        data['quarter'] = data['weekEndingDate'].dt.quarter

        print(f"Loaded ESR data: {len(data)} records")
        print(f"Date range: {data['weekEndingDate'].min()} to {data['weekEndingDate'].max()}")
        print(f"Countries: {data['country'].nunique()} unique")
        print(f"Commodities: {data['commodity'].nunique()} unique")

        return data

    except Exception as e:
        print(f"Error loading ESR data: {e}")
        return pd.DataFrame()


def generate_esr_summary_report(data, commodity=None, country=None):
    """
    Generate a comprehensive summary report for ESR data.

    Args:
        data: ESR DataFrame
        commodity: Optional commodity filter
        country: Optional country filter

    Returns:
        dict: Summary report
    """
    # Apply filters
    filtered_data = data.copy()

    if commodity:
        filtered_data = filtered_data[filtered_data['commodity'] == commodity]

    if country:
        filtered_data = filtered_data[filtered_data['country'] == country]

    if filtered_data.empty:
        return {'error': 'No data available for specified filters'}

    # Calculate summary statistics
    report = {
        'data_summary': {
            'total_records': len(filtered_data),
            'date_range': {
                'start': filtered_data['weekEndingDate'].min().strftime('%Y-%m-%d'),
                'end': filtered_data['weekEndingDate'].max().strftime('%Y-%m-%d')
            },
            'countries': filtered_data['country'].nunique(),
            'commodities': filtered_data['commodity'].nunique()
        },

        'export_metrics': {
            'total_weekly_exports': filtered_data['weeklyExports'].sum(),
            'average_weekly_exports': filtered_data['weeklyExports'].mean(),
            'peak_weekly_exports': filtered_data['weeklyExports'].max(),
            'total_accumulated_exports': filtered_data['accumulatedExports'].max()
        },

        'sales_metrics': {
            'total_gross_new_sales': filtered_data['grossNewSales'].sum(),
            'average_gross_new_sales': filtered_data['grossNewSales'].mean(),
            'current_outstanding_sales': filtered_data['outstandingSales'].iloc[-1],
            'current_total_commitment': filtered_data['currentMYTotalCommitment'].iloc[-1]
        },

        'market_analysis': {
            'top_countries_by_exports': filtered_data.groupby('country')['weeklyExports'].sum().nlargest(5).to_dict(),
            'market_concentration': ESRAnalytics.calculate_market_share(filtered_data).head(5).to_dict('records')
        },

        'trends': {
            'sales_velocity_trend': 'increasing' if filtered_data['sales_velocity'].iloc[
                                                    -5:].mean() > 0 else 'decreasing',
            'commitment_ratio_current': filtered_data['commitment_ratio'].iloc[-1],
            'seasonal_pattern': 'detected' if filtered_data['month'].nunique() > 6 else 'limited_data'
        }
    }

    return report


# Example configuration for different dashboard layouts
DASHBOARD_CONFIGURATIONS = {
    'executive_summary': {
        'grid_config': {
            'layout_type': 'manual',
            'rows': 2,
            'cols': 2,
            'gap': '15px'
        },
        'charts': ['weekly_exports', 'outstanding_sales', 'top_countries', 'commitment_trends']
    },

    'detailed_analysis': {
        'grid_config': {
            'layout_type': 'auto',
            'responsive': True
        },
        'charts': ['weekly_exports', 'outstanding_sales', 'gross_new_sales',
                   'commitment_analysis', 'country_breakdown', 'forecast']
    },

    'country_focus': {
        'grid_config': {
            'layout_type': 'manual',
            'rows': 1,
            'cols': 3
        },
        'charts': ['country_exports', 'country_commitments', 'country_trends']
    }
}