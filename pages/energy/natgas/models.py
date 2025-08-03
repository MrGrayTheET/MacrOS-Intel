from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dash import html, dcc, callback, Output, Input, callback_context, no_update
from plotly import graph_objects as go, express as px
from data.data_tables import EIATable
from model_apps import ModelApp
from utils.data_tools import key_to_name, fetch_contract_keys, calc_contract_spreads


class StorageApp(ModelApp):
    """
    Enhanced StorageApp with flexible filtering capabilities
    """

    def __init__(self, storage_key='storage/total_lower_48', price_key='prices/NG_1', withdrawal_key ='consumption/net_withdrawals', app_id='price-store-reg'):
        super().__init__(table_client=EIATable('NG'), keys=[storage_key, price_key, withdrawal_key])
        self.storage_key = storage_key
        self.prices_key = price_key
        self.withdrawal_key = withdrawal_key
        self.prices_col = key_to_name(price_key)
        self.storage_col = key_to_name(storage_key)
        self.withdrawal_col = key_to_name(withdrawal_key)
        self.data.ffill(inplace=True)
        self.data = self.data.resample('1W').apply({self.storage_col: 'last', self.prices_col: 'last', self.withdrawal_col:
                                                    'last'})
        # Calculate storage metrics
        self.app_func = self.seasonal_storage_data
        self.data = self.seasonal_storage_data()
        # Configure filters for this specific application
        self.configure_filters({
            'month': {
                'key_type': 'range',
                'label': 'Month',
                'min_val': 1,
                'max_val': 12,

            },
            'signal_strength_decile': {
                'key_type': 'range',
                'label': 'Signal Strength Decile',
                'min_val':0,
                'max_val':9
            },
        })

    def update_keys(self, storage_key=None, price_key=None):
        keys_to_add = []
        cols_to_drop = []
        if isinstance(self.data.index, pd.Index):
            self.data.index = pd.to_datetime(self.data.index)

        if storage_key:
            if storage_key != self.storage_key:
                cols_to_drop.append(self.storage_col)
                self.storage_key = storage_key
                self.storage_col = key_to_name(storage_key)
                keys_to_add.append(storage_key)

        if price_key:
            if self.prices_key != price_key:
                cols_to_drop.append(self.prices_col)
                self.prices_key = price_key
                self.prices_col = key_to_name(price_key)
                keys_to_add.append(price_key)

        # Preserve the index range before adding new data
        min_idx, max_idx = self.data.index.min(), self.data.index.max()

        if keys_to_add:
            self.__add__(keys_to_add)
            self.data = self.data.ffill().drop(columns=cols_to_drop)


            # Filter Non-datetimes from index
            self.data = self.data[[isinstance(idx, (pd.Timestamp, datetime)) for idx in self.data.index]]
            self.data.index = pd.to_datetime(self.data.index) if isinstance(self.data.index,
                                                                            pd.Index) else self.data.index

            # Resample to weekly and recalculate metrics
            self.data = self.data.resample('1W').apply({
                self.storage_col: 'last',
                self.prices_col: 'last',
                self.withdrawal_col: 'last'
            })

            self.data = self.seasonal_storage_data()

        return self.data

    def seasonal_storage_data(self):
        df = self.data.copy()
        df['date'] = df.index
        df = df.sort_values('date')

        # Add week-of-year, month, and year columns
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Calculate seasonal historical min/max (same week, past 5 years)
        df['hist_max'] = np.nan
        df['hist_min'] = np.nan
        df['hist_mean'] = np.nan

        for idx, row in df.iterrows():
            current_week = row['week_of_year']
            current_year = row['year']

            # Get historical data (same week, past 5 years including current year)
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

        # Calculate percentage position within historical range
        df['pct_above_min'] = ((df[self.storage_col] - df['hist_min']) /
                               (df['hist_max'] - df['hist_min'])) * 100

        df['dev_from_mean'] = ((df[self.storage_col] - df['hist_mean']) / df[self.storage_col].std())

        # Calculate future price change (4 weeks ahead)
        df['price_1mo'] = df[self.prices_col].shift(-4)
        df['price_3mo'] = df[self.prices_col].shift(-12)
        df['return_1mo'] = np.log(df.price_1mo) - np.log(df[self.prices_col])
        df['return_3mo'] = np.log(df.price_3mo) - np.log(df[self.prices_col])
        df['signal_strength_decile'] = pd.qcut(df['dev_from_mean'],10, labels=False)

        return df

    def generate_layout_div(self):
        """Generate the enhanced Dash layout with flexible filtering"""
        # Filter dates to exclude last 4 weeks to avoid incomplete forward-looking data
        valid_dates = self.data[self.data['date'] <= self.data['date'].max() - timedelta(weeks=4)]

        if 'date' in valid_dates.columns:
            min_date = valid_dates['date'].min()
            max_date = valid_dates['date'].max()
            default_date = max_date
        else:
            min_date = valid_dates.index.min()
            max_date = valid_dates.index.max()
            default_date = max_date

        # Generate menus using the enhanced base class functionality
        menu_div = self.generate_menu(categories=['prices', 'storage'])

        # Generate the flexible filter menu
        filter_menu = self.generate_filter_menu(
            filter_columns=['month', 'signal_strength_decile'],  # Only these 2
            filter_configs={  # Only include configs for these 2
                'month': self.filter_configs['month'],
                'signal_strength_decile': self.filter_configs['signal_strength_decile']
            },
            include_series_selector=False
        )

        layout = html.Div([
            html.H1("Natural Gas Storage-Price Analysis", style={'textAlign': 'center'}),
            html.Div(children=[
                menu_div,
                filter_menu,
                html.Div(id=f'{self.app_id}-scatter-plot-div', children=[
                    dcc.Graph(id=f'{self.app_id}-scatter-plot')],
                         style={
                             'height': '20%',
                             'width': '50%',
                             'marginTop': '5px'
                         })],
                style={'display': 'flex', 'height': '20%', 'width': '100%'}),

            html.Div(id=f'{self.app_id}-line-plot-div', children=[

                dcc.Tabs(children=[
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-storage-plot')
                    ]),
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-withdrawal-plot')
                    ])
                ]),
            ],
                     style={
                         'height': '60%',
                         'width': '100%',
                         'marginTop': '10px'
                     }),

            html.Div(id=f'{self.app_id}-metrics-container', className='metrics-grid', style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '15px',
                'margin': '20px'
            }),

            html.Div([
                html.H3("Analysis Summary"),
                html.Div(id=f'{self.app_id}-summary')
            ], style={'margin': '20px'})
        ])

        return layout

    def register_callbacks(self, app):
        @callback(
            [Output(f'{self.app_id}-storage-plot', 'figure'),
             Output(f'{self.app_id}-withdrawal-plot', 'figure'),
             Output(f'{self.app_id}-metrics-container', 'children'),
             Output(f'{self.app_id}-scatter-plot', 'figure'),
             Output(f'{self.app_id}-summary', 'children'),
             Output(f'{self.app_id}-month-range-slider', 'value'),
             Output(f'{self.app_id}-signal_strength_decile-range-slider', 'value')],
            [Input(f'{self.app_id}-date-picker', 'date'),
             Input(f'{self.app_id}-apply-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-reset-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-prices-dropdown', 'value'),
             Input(f'{self.app_id}-storage-dropdown', 'value'),
             Input(f'{self.app_id}-month-range-slider', 'value'),
             Input(f'{self.app_id}-signal_strength_decile-range-slider', 'value')],
            prevent_initial_call=False
        )
        def update_analysis(selected_date, apply_clicks, reset_clicks,
                            price_key, storage_key, month_range, ssq_values):
            # Get callback context to identify trigger
            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            selected_date = pd.to_datetime(selected_date)

            # Handle key updates first (affects both original and filtered data)
            if price_key or storage_key:
                self.update_keys(price_key=price_key, storage_key=storage_key)
                self.original_data = self.data.copy()  # Update original data
                self.filtered_data = pd.DataFrame()  # Reset filtered data

            # Use original data as base for processing
            df = self.original_data.copy().dropna(subset=[self.withdrawal_col, self.prices_col, self.storage_col])

            # Handle reset action
            if trigger_id == f'{self.app_id}-reset-filters-btn':
                self.filtered_data = pd.DataFrame()  # Clear filtered data
                # Get default values for filters
                month_options = self.get_filter_options('month')
                ssq_options = self.get_filter_options('signal_strength_decile')
                reset_month_range = [month_options['min'], month_options['max']]
                reset_ssq_values = [ssq_options['min'], ssq_options['max']]
                summary_text = "Filters reset successfully"

            # Handle apply action
            elif trigger_id == f'{self.app_id}-apply-filters-btn':
                filter_values = {
                    'month_range': month_range,
                    'signal_strength_decile_values': ssq_values
                }
                self.filtered_data = self.apply_filters(df, filter_values)
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = f"Applied filters: {len(self.filtered_data)}/{len(df)} records"

            # No filter action - use last state
            else:
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = "Using current data selection"

            # Determine which dataset to use for visualizations
            display_df = self.filtered_data if not self.filtered_data.empty else df
            # Get data for selected date (same logic as before)
            if 'date' in df.columns:
                selected_data = df[df['date'] == selected_date]
            else:
                selected_data = df[df.index == selected_date]

            if selected_data.empty:
                if 'date' in df.columns:
                    closest_date = df.loc[(df['date'] - selected_date).abs().idxmin(), 'date']
                    selected_data = df[df['date'] == closest_date]
                else:
                    delta_series = pd.Series(df.index - selected_date)
                    closest_idx = delta_series.abs().idxmin()
                    selected_data = df.iloc[[closest_idx]]

            selected_row = selected_data.iloc[0]

            # Create visualizations (same logic as before but with filtered data)
            storage_fig = go.Figure()

            # Add storage level
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df[self.storage_col],
                name='Current Storage',
                line=dict(color='blue', width=2)
            ))

            # Add 5-year max/min bands
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_max'],
                name='5-Year Max',
                line=dict(color='red', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_min'],
                name='5-Year Min',
                line=dict(color='green', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_mean'],
                name='5-Year Mean',
                line=dict(color='orange', dash='dot')
            ))

            storage_fig.update_layout(
                title="Natural Gas Storage Levels vs Historical Range",
                xaxis_title="Date",
                yaxis_title="Storage Volume (BCF)",
                hovermode='x unified'
            )

            withdrawal_fig = go.Figure(go.Scatter(
                x=df['date'],
                y=df[self.withdrawal_col],
                line=dict(color='blue')
                ),
                layout=dict(xaxis_title = "Date",
                yaxis_title= "Net Withdrawals (BCF)"))

            # Prepare metrics (same as before)
            metrics = [
                ("Current Storage", f"{selected_row[self.storage_col]:,.0f} BCF"),
                ("5-Year Max", f"{selected_row['hist_max']:,.0f} BCF"),
                ("5-Year Min", f"{selected_row['hist_min']:,.0f} BCF"),
                ("% Above 5Y Min", f"{selected_row['pct_above_min']:.1f}%"),
                ("Current Price", f"${selected_row[self.prices_col]:,.2f}"),
                ("Price 1Mo Later",
                 f"${selected_row['price_1mo']:,.2f}" if pd.notna(selected_row['price_1mo']) else "N/A"),
                ("1Mo Price Change",
                 f"{selected_row['return_1mo']:+.1f}%" if pd.notna(selected_row['return_1mo']) else "N/A"),
                ("3Mo Price Change",
                 f"{selected_row['return_3mo']:+.1f}%" if pd.notna(selected_row['return_3mo']) else "N/A")
            ]

            # Create metrics HTML
            metrics_html = [
                html.Div([
                    html.Div(metric[0], className='metric-label', style={'fontWeight': 'bold'}),
                    html.Div(metric[1], className='metric-value', style={'fontSize': '1.2em'})
                ], className='metric-item', style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'backgroundColor': '#f9f9f9'
                }) for metric in metrics
            ]

            # Create scatter plot showing historical relationship with filtered data
            valid_data = display_df.dropna(subset=['dev_from_mean', 'return_1mo'])

            scatter_fig = px.scatter(
                display_df,
                x='dev_from_mean',
                y='return_1mo',
                color='signal_strength_decile' if 'signal_strength_decile' in valid_data.columns else None,
                trendline='ols',
                title=f"Storage Position vs Future Price Change (1 Month) - {len(valid_data)} observations",
                labels={
                    'dev_from_mean': 'Deviation from 5-Year Mean',
                    'return_1mo': 'Price Change Next Month (%)',
                    'signal_strength_decile': 'Signal Strength'
                }
            )

            # Add current selection
            if pd.notna(selected_row['return_1mo']):
                scatter_fig.add_trace(go.Scatter(
                    x=[selected_row['dev_from_mean']],
                    y=[selected_row['return_1mo']],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Selected Date'
                ))

            # Analysis summary with filter information
            storage_position = "high" if selected_row['pct_above_min'] > 50 else "low"
            expected_direction = "decrease" if selected_row['pct_above_min'] > 50 else "increase"

            filter_info = ""
            if apply_clicks and not reset_clicks:
                filter_info = f"\n• Analysis based on {len(valid_data)} filtered observations"

            summary_text = f"""
            Based on the analysis for {selected_date.strftime('%Y-%m-%d')}:

            • Storage is at {selected_row['pct_above_min']:.1f}% above the 5-year minimum, indicating a {storage_position} storage position
            • Historical patterns suggest prices typically {expected_direction} when storage is at this level
            • Actual price change was {selected_row['return_1mo']:+.1f}% over the following month{filter_info}
            """

            summary_html = html.Pre(summary_text, style={'whiteSpace': 'pre-wrap'})

            return  storage_fig,withdrawal_fig,  metrics_html,  scatter_fig, summary_html,  reset_month_range,  reset_ssq_values


class StorageAppWSpreads(StorageApp):
    """
    Enhanced StorageApp with flexible filtering capabilities
    """

    def __init__(self, storage_key='storage/total_lower_48', price_key='prices/NG_1', withdrawal_key ='consumption/net_withdrawals'):
        super().__init__()
        self.data.drop(columns=[self.prices_col], inplace=True)
        self.futures_keys = fetch_contract_keys(self.table_client)
        self.__add__(self.futures_keys)
        self.futures_cols = [key_to_name(k) for k in self.futures_keys]
        resample_params = {self.storage_col: 'last', self.prices_col: 'last', self.withdrawal_col:
                                                    'last'}
        resample_params.update({self.futures_cols[i]:'last' for i in range(0, len(self.futures_cols))})
        self.data.ffill(inplace=True)
        self.data = self.data.resample('1W').last()
        # Calculate storage metrics
        self.app_func = self.seasonal_storage_data
        self.data = self.seasonal_storage_data()
        # Configure filters for this specific application
        self.configure_filters({
            'month': {
                'key_type': 'range',
                'label': 'Month',
                'min_val': 1,
                'max_val': 12
            },
            'signal_strength_decile': {
                'key_type': 'range',
                'label': 'Signal Strength Decile',
                'min_val':0,
                'max_val':9
            },
            'signal_series': {
                'key_type': 'single_select',
                'label': 'Signal Series',
                'source': 'columns',
                'match': {'contains': '1mo'}
            }
        })

    def update_keys(self, storage_key=None, price_key=None):
        keys_to_add = []
        cols_to_drop = []
        if isinstance(self.data.index, pd.Index):
            self.data.index = pd.to_datetime(self.data.index)

        if storage_key:
            if storage_key != self.storage_key:
                cols_to_drop.append(self.storage_col)
                self.storage_key = storage_key
                self.storage_col = key_to_name(storage_key)
                keys_to_add.append(storage_key)

        if price_key:
            if self.prices_key != price_key:
                cols_to_drop.append(self.prices_col)
                self.prices_key = price_key
                self.prices_col = key_to_name(price_key)
                keys_to_add.append(price_key)

        # Preserve the index range before adding new data
        min_idx, max_idx = self.data.index.min(), self.data.index.max()

        if keys_to_add:
            self.__add__(keys_to_add)
            self.data = self.data.ffill().drop(columns=cols_to_drop)


            # Filter Non-datetimes from index
            self.data = self.data[[isinstance(idx, (pd.Timestamp, datetime)) for idx in self.data.index]]
            self.data.index = pd.to_datetime(self.data.index) if isinstance(self.data.index,
                                                                            pd.Index) else self.data.index

            # Resample to weekly and recalculate metrics
            self.data = self.data.resample('1W').apply({
                self.storage_col: 'last',
                self.prices_col: 'last',
                self.withdrawal_col: 'last'
            })

            self.data = self.seasonal_storage_data()

        return self.data

    def seasonal_storage_data(self):
        df = self.data.copy()
        df['date'] = df.index
        df = df.sort_values('date')

        # Add week-of-year, month, and year columns
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Calculate seasonal historical min/max (same week, past 5 years)
        df['hist_max'] = np.nan
        df['hist_min'] = np.nan
        df['hist_mean'] = np.nan

        for idx, row in df.iterrows():
            current_week = row['week_of_year']
            current_year = row['year']

            # Get historical data (same week, past 5 years including current year)
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

        # Calculate percentage position within historical range
        df['pct_above_min'] = ((df[self.storage_col] - df['hist_min']) /
                               (df['hist_max'] - df['hist_min'])) * 100

        df['dev_from_mean'] = ((df[self.storage_col] - df['hist_mean']) / df['hist_mean'])

        # Calculate future price change (4 weeks ahead)
        df['price_1mo'] = df[self.prices_col].shift(-4)
        df['price_3mo'] = df[self.prices_col].shift(-12)
        df['return_1mo'] = np.log(df.price_1mo) - np.log(df[self.prices_col])
        df['return_3mo'] = np.log(df.price_3mo) - np.log(df[self.prices_col])
        df['signal_strength_decile'] = pd.qcut(df['dev_from_mean'],10, labels=False)

        df = calc_contract_spreads(df, second_month=False)
        for i in df.columns:
            if 'spread' in i:
                df[f'{i}_return_1mo'] = df[i].shift(-4) - df[i]
                df[f'{i}_return_1wk'] = df[i].shift(-1) - df[i]

        return df

    def generate_layout_div(self):
        """Generate the enhanced Dash layout with flexible filtering"""
        # Filter dates to exclude last 4 weeks to avoid incomplete forward-looking data
        valid_dates = self.data[self.data['date'] <= self.data['date'].max() - timedelta(weeks=4)]

        if 'date' in valid_dates.columns:
            min_date = valid_dates['date'].min()
            max_date = valid_dates['date'].max()
            default_date = max_date
        else:
            min_date = valid_dates.index.min()
            max_date = valid_dates.index.max()
            default_date = max_date

        # Generate menus using the enhanced base class functionality
        menu_div = self.generate_menu(categories=['prices', 'storage'])

        # Generate the flexible filter menu
        filter_menu = self.generate_filter_menu(
            filter_columns=['month', 'signal_strength_decile'],  # Only these 2
            filter_configs={  # Only include configs for these 2
                'month': self.filter_configs['month'],
                'signal_strength_decile': self.filter_configs['signal_strength_decile']
            },
            include_series_selector=True
        )

        layout = html.Div([
            html.H1("Natural Gas Storage-Price Analysis", style={'textAlign': 'center'}),
            html.Div(children=[
                menu_div,
                filter_menu,
                html.Div(id=f'{self.app_id}-scatter-plot-div', children=[
                    dcc.Graph(id=f'{self.app_id}-scatter-plot')],
                         style={
                             'height': '20%',
                             'width': '50%',
                             'marginTop': '5px'
                         })],
                style={'display': 'flex', 'height': '20%', 'width': '100%'}),

            html.Div(id=f'{self.app_id}-line-plot-div', children=[

                dcc.Tabs(children=[
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-storage-plot')
                    ]),
                    dcc.Tab(children=[
                        dcc.Graph(id=f'{self.app_id}-withdrawal-plot')
                    ])
                ]),
            ],
                     style={
                         'height': '60%',
                         'width': '100%',
                         'marginTop': '10px'
                     }),

            html.Div(id=f'{self.app_id}-metrics-container', className='metrics-grid', style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '15px',
                'margin': '20px'
            }),

            html.Div([
                html.H3("Analysis Summary"),
                html.Div(id=f'{self.app_id}-summary')
            ], style={'margin': '20px'})
        ])

        return layout

    def register_callbacks(self, app):
        @callback(
            [Output(f'{self.app_id}-storage-plot', 'figure'),
             Output(f'{self.app_id}-withdrawal-plot', 'figure'),
             Output(f'{self.app_id}-metrics-container', 'children'),
             Output(f'{self.app_id}-scatter-plot', 'figure'),
             Output(f'{self.app_id}-summary', 'children'),
             Output(f'{self.app_id}-month-range-slider', 'value'),
             Output(f'{self.app_id}-signal_strength_decile-range-slider', 'value')],
            [Input(f'{self.app_id}-date-picker', 'date'),
             Input(f'{self.app_id}-apply-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-reset-filters-btn', 'n_clicks'),
             Input(f'{self.app_id}-prices-dropdown', 'value'),
             Input(f'{self.app_id}-storage-dropdown', 'value'),
             Input(f'{self.app_id}-month-range-slider', 'value'),
             Input(f'{self.app_id}-signal_strength_decile-range-slider', 'value'),
             Input(f'{self.app_id}-x-dd', 'value'),  # New X dropdown input
             Input(f'{self.app_id}-y-dd', 'value')],  # New Y dropdown input
            prevent_initial_call=False
        )
        def update_analysis(selected_date, apply_clicks, reset_clicks,
                            price_key, storage_key, month_range, ssq_values,
                            x_variable, y_variable):  # New parameters
            # Get callback context to identify trigger
            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            selected_date = pd.to_datetime(selected_date)

            # Handle key updates first (affects both original and filtered data)
            if price_key or storage_key:
                self.update_keys(price_key=price_key, storage_key=storage_key)
                self.original_data = self.data.copy()  # Update original data
                self.filtered_data = pd.DataFrame()  # Reset filtered data

            # Use original data as base for processing
            df = self.original_data.copy().dropna(subset=[self.withdrawal_col, self.prices_col, self.storage_col])

            # Handle reset action
            if trigger_id == f'{self.app_id}-reset-filters-btn':
                self.filtered_data = pd.DataFrame()  # Clear filtered data
                # Get default values for filters
                month_options = self.get_filter_options('month')
                ssq_options = self.get_filter_options('signal_strength_decile')
                reset_month_range = [month_options['min'], month_options['max']]
                reset_ssq_values = [ssq_options['min'], ssq_options['max']]
                summary_text = "Filters reset successfully"

            # Handle apply action
            elif trigger_id == f'{self.app_id}-apply-filters-btn':
                filter_values = {
                    'month_range': month_range,
                    'signal_strength_decile_range': ssq_values  # Fixed key name
                }
                self.filtered_data = self.apply_filters(df, filter_values)
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = f"Applied filters: {len(self.filtered_data)}/{len(df)} records"

            # No filter action - use last state
            else:
                reset_month_range = no_update
                reset_ssq_values = no_update
                summary_text = "Using current data selection"

            # Determine which dataset to use for visualizations
            display_df = self.filtered_data if not self.filtered_data.empty else df

            # Get data for selected date
            if 'date' in df.columns:
                selected_data = df[df['date'] == selected_date]
            else:
                selected_data = df[df.index == selected_date]

            if selected_data.empty:
                if 'date' in df.columns:
                    closest_date = df.loc[(df['date'] - selected_date).abs().idxmin(), 'date']
                    selected_data = df[df['date'] == closest_date]
                else:
                    delta_series = pd.Series(df.index - selected_date)
                    closest_idx = delta_series.abs().idxmin()
                    selected_data = df.iloc[[closest_idx]]

            selected_row = selected_data.iloc[0]

            # Create storage visualization
            storage_fig = go.Figure()

            # Add storage level
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df[self.storage_col],
                name='Current Storage',
                line=dict(color='blue', width=2)
            ))

            # Add 5-year max/min bands
            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_max'],
                name='5-Year Max',
                line=dict(color='red', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_min'],
                name='5-Year Min',
                line=dict(color='green', dash='dash')
            ))

            storage_fig.add_trace(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df['hist_mean'],
                name='5-Year Mean',
                line=dict(color='orange', dash='dot')
            ))

            storage_fig.update_layout(
                title="Natural Gas Storage Levels vs Historical Range",
                xaxis_title="Date",
                yaxis_title="Storage Volume (BCF)",
                hovermode='x unified'
            )

            # Create withdrawal figure
            withdrawal_fig = go.Figure(go.Scatter(
                x=df['date'] if 'date' in df.columns else df.index,
                y=df[self.withdrawal_col],
                line=dict(color='blue')
            ))

            withdrawal_fig.update_layout(
                title="Net Withdrawals",
                xaxis_title="Date",
                yaxis_title="Net Withdrawals (BCF)"
            )

            # Prepare metrics
            metrics = [
                ("Current Storage", f"{selected_row[self.storage_col]:,.0f} BCF"),
                ("5-Year Max", f"{selected_row['hist_max']:,.0f} BCF"),
                ("5-Year Min", f"{selected_row['hist_min']:,.0f} BCF"),
                ("% Above 5Y Min", f"{selected_row['pct_above_min']:.1f}%"),
                ("Current Price", f"${selected_row[self.prices_col]:,.2f}"),
                ("Price 1Mo Later",
                 f"${selected_row['price_1mo']:,.2f}" if pd.notna(selected_row['price_1mo']) else "N/A"),
                ("1Mo Price Change",
                 f"{selected_row['return_1mo'] * 100:+.1f}%" if pd.notna(selected_row['return_1mo']) else "N/A"),
                ("3Mo Price Change",
                 f"{selected_row['return_3mo'] * 100:+.1f}%" if pd.notna(selected_row['return_3mo']) else "N/A")
            ]

            # Create metrics HTML
            metrics_html = [
                html.Div([
                    html.Div(metric[0], className='metric-label', style={'fontWeight': 'bold'}),
                    html.Div(metric[1], className='metric-value', style={'fontSize': '1.2em'})
                ], className='metric-item', style={
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'backgroundColor': '#f9f9f9'
                }) for metric in metrics
            ]

            # Create scatter plot using selected X and Y variables
            # Set defaults if not selected
            if not x_variable:
                x_variable = 'total_lower_48_storage' if 'total_lower_48_storage' in display_df.columns else \
                display_df.columns[0]
            if not y_variable:
                y_variable = 'return_1mo' if 'return_1mo' in display_df.columns else display_df.columns[1]

            # Ensure selected variables exist in the dataframe
            if x_variable not in display_df.columns or y_variable not in display_df.columns:
                scatter_fig = go.Figure()
                scatter_fig.add_annotation(
                    text="Selected variables not found in data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False
                )
                scatter_fig.update_layout(
                    title="Scatter Plot - Invalid Variable Selection",
                    xaxis_title=x_variable,
                    yaxis_title=y_variable
                )
            else:
                # Create scatter plot with filtered data
                valid_data = display_df.dropna(subset=[x_variable, y_variable])

                # Determine color variable (use signal_strength_decile if available)
                color_var = 'signal_strength_decile' if 'signal_strength_decile' in valid_data.columns else None

                scatter_fig = px.scatter(
                    valid_data,
                    x=x_variable,
                    y=y_variable,
                    color=color_var,
                    trendline='ols',
                    title=f"{y_variable} vs {x_variable} - {len(valid_data)} observations",
                    labels={
                        x_variable: x_variable.replace('_', ' ').title(),
                        y_variable: y_variable.replace('_', ' ').title(),
                        'signal_strength_decile': 'Signal Strength'
                    }
                )

                # If y_variable is a return, multiply by 100 for percentage
                if 'return' in y_variable.lower():
                    scatter_fig.update_layout(yaxis=dict(tickformat='.1%'))

                # Add current selection point if both variables are available
                if pd.notna(selected_row.get(x_variable)) and pd.notna(selected_row.get(y_variable)):
                    scatter_fig.add_trace(go.Scatter(
                        x=[selected_row[x_variable]],
                        y=[selected_row[y_variable]],
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='star'),
                        name='Selected Date',
                        showlegend=True
                    ))

            # Analysis summary with dynamic variable information
            storage_position = "high" if selected_row['pct_above_min'] > 50 else "low"
            expected_direction = "decrease" if selected_row['pct_above_min'] > 50 else "increase"

            filter_info = ""
            if not self.filtered_data.empty:
                filter_info = f"\n• Analysis based on {len(self.filtered_data)} filtered observations"

            # Add scatter plot variable info
            scatter_info = f"\n• Scatter plot shows {y_variable} vs {x_variable}"

            summary_text = f"""
            Based on the analysis for {selected_date.strftime('%Y-%m-%d')}:

            • Storage is at {selected_row['pct_above_min']:.1f}% above the 5-year minimum, indicating a {storage_position} storage position
            • Historical patterns suggest prices typically {expected_direction} when storage is at this level
            • Actual price change was {selected_row.get('return_1mo', 0) * 100:+.1f}% over the following month{filter_info}
            {scatter_info}
            """

            summary_html = html.Pre(summary_text, style={'whiteSpace': 'pre-wrap'})

            return (storage_fig, withdrawal_fig, metrics_html, scatter_fig,
                    summary_html, reset_month_range, reset_ssq_values)
