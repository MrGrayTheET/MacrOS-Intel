import pandas as pd
from dash import Output, Input, html
from plotly import graph_objects as go, express as px

from data.data_tables import TableClient
from dash import Input, Output, callback


class import_export_callbacks:

    def __init__(self, table_client:TableClient):
        self.table_client = table_client
        return

    def register_callbacks(self):
        table_client = self.table_client

        # Callback to update available options when commodity changes
        @callback(
           [Output('country-dropdown', 'options'),
            Output('country-dropdown', 'value'),
            Output('year-dropdown', 'options'),
            Output('year-dropdown', 'value')],
           [Input('commodity-dropdown', 'value'),
            Input('data-key_type-store', 'children')]
       )

        def update_commodity_data(self, selected_commodity, data_type):
            # Get new data for the selected commodity using commodity/data_type pattern
            df_commodity = table_client[f'{selected_commodity}/{data_type}']

            # Extract unique countries and years for the new commodity
            countries_new = df_commodity.index.get_level_values('Partner').unique().tolist()
            years_new = df_commodity.index.get_level_values('date').year.unique().tolist()
            years_new.sort()

            # Prepare dropdown options
            country_options = [{'label': country, 'value': country} for country in countries_new]
            year_options = [{'label': str(year), 'value': year} for year in years_new]

            # Set default values
            default_country = countries_new[0] if countries_new else None
            default_year = years_new[-1] if years_new else None

            return country_options, default_country, year_options, default_year


        # Callback for time series chart
        @callback(
            Output('time-series-chart', 'figure'),
            [Input('commodity-dropdown', 'value'),
             Input('country-dropdown', 'value'),
             Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date'),
            Input('data-key_type-store', 'children')]
        )
        def update_time_series(self, selected_commodity, selected_country, start_date, end_date, data_type):
           # Define colors
           colors = {
               'primary': '#64B5F6',
               'secondary': '#E57373',
               'background': '#121212',
               'card_background': '#1E1E1E',
               'text': '#FFFFFF',
               'text_secondary': '#B0B0B0',
               'accent': '#FFB74D',
               'grid': '#2A2A2A',
               'border': '#333333'
           }

           # Get commodity data using commodity/data_type pattern
           df_commodity = table_client[f'{selected_commodity}/{data_type}']

           # Filter data for selected country and date range
           if selected_country not in df_commodity.index.get_level_values('Partner'):
               # Return empty figure if country not available for this commodity
               fig = go.Figure()
               fig.update_layout(
                   title=f"No data available for {selected_country} in {selected_commodity.title()}",
                   plot_bgcolor=colors['card_background'],
                   paper_bgcolor=colors['card_background'],
                   font=dict(color=colors['text'])
               )
               return fig

           country_data = df_commodity.loc[selected_country].copy()

           # Convert date strings to datetime if needed
           if start_date and end_date:
               start_date = pd.to_datetime(start_date)
               end_date = pd.to_datetime(end_date)
               country_data = country_data[(country_data.index >= start_date) &
                                           (country_data.index <= end_date)]

           # Create time series plot
           fig = go.Figure()

           fig.add_trace(go.Scatter(
               x=country_data.index,
               y=country_data['value'],
               mode='lines+markers',
               name=f"{selected_country} ({selected_commodity.title()})",
               line=dict(color=colors['primary'], width=3),
               marker=dict(size=6, color=colors['primary'])
           ))

           fig.update_layout(
               title=f"Monthly {selected_commodity.title()} {data_type.title()} Values - {selected_country}",
               xaxis_title="Date",
               yaxis_title=f"{data_type.title()} Value",
               hovermode='x unified',
               plot_bgcolor=colors['card_background'],
               paper_bgcolor=colors['card_background'],
               font=dict(color=colors['text']),
               showlegend=False,
               title_font=dict(color=colors['text'], size=16)
           )

           fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'],
                            color=colors['text'], tickcolor=colors['text'])
           fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['grid'],
                            color=colors['text'], tickcolor=colors['text'])

           return fig


       # Callback for breakdown chart
        @callback(
           Output('breakdown-chart', 'figure'),
           [Input('commodity-dropdown', 'value'),
            Input('year-dropdown', 'value'),
            Input('chart-key_type-radio', 'value'),
            Input('data-key_type-store', 'children')]
       )
        def update_breakdown(selected_commodity, selected_year, chart_type, data_type):
           # Define colors
           colors = {
               'primary': '#64B5F6',
               'secondary': '#E57373',
               'background': '#121212',
               'card_background': '#1E1E1E',
               'text': '#FFFFFF',
               'text_secondary': '#B0B0B0',
               'accent': '#FFB74D',
               'grid': '#2A2A2A',
               'border': '#333333'
           }

           # Get commodity data using commodity/data_type pattern
           df_commodity = table_client[f'{selected_commodity}/{data_type}']

           # Filter data for selected year
           year_data = df_commodity[df_commodity.index.get_level_values('date').year == selected_year].copy()

           if year_data.empty:
               # Return empty figure if no data for this year
               fig = go.Figure()
               fig.update_layout(
                   title=f"No data available for {selected_year} in {selected_commodity.title()}",
                   plot_bgcolor=colors['card_background'],
                   paper_bgcolor=colors['card_background'],
                   font=dict(color=colors['text'])
               )
               return fig

           # Group by country and sum values
           country_totals = year_data.groupby('Partner')['value'].sum().sort_values(ascending=False)

           if chart_type == 'pie':
               fig = px.pie(
                   values=country_totals.values,
                   names=country_totals.index,
                   title=f"{selected_commodity.title()} {data_type.title()} Sources Distribution - {selected_year}",
                   color_discrete_sequence=px.colors.qualitative.Set3
               )
               fig.update_traces(textposition='inside', textinfo='percent+label',
                                 textfont=dict(color='white', size=12))

           else:  # bar chart
               fig = px.bar(
                   x=country_totals.index,
                   y=country_totals.values,
                   title=f"{selected_commodity.title()} {data_type.title()} Values by Country - {selected_year}",
                   color=country_totals.values,
                   color_continuous_scale='viridis'
               )
               fig.update_layout(
                   xaxis_title="Country",
                   yaxis_title=f"Total {data_type.title()} Value",
                   showlegend=False
               )
               fig.update_xaxes(tickangle=45, color=colors['text'], tickcolor=colors['text'])
               fig.update_yaxes(color=colors['text'], tickcolor=colors['text'])

           fig.update_layout(
               plot_bgcolor=colors['card_background'],
               paper_bgcolor=colors['card_background'],
               font=dict(color=colors['text']),
               title_font=dict(color=colors['text'], size=16)
           )

           return fig


       # Callback for summary statistics
        @callback(
           Output('summary-stats', 'children'),
           [Input('commodity-dropdown', 'value'),
            Input('country-dropdown', 'value'),
            Input('year-dropdown', 'value'),
            Input('data-key_type-store', 'children')]
       )
        def update_summary_stats(selected_commodity, selected_country, selected_year, data_type):
           # Define colors
           colors = {
               'primary': '#64B5F6',
               'secondary': '#E57373',
               'background': '#121212',
               'card_background': '#1E1E1E',
               'text': '#FFFFFF',
               'text_secondary': '#B0B0B0',
               'accent': '#FFB74D',
               'grid': '#2A2A2A',
               'border': '#333333'
           }

           # Get commodity data using commodity/data_type pattern
           df_commodity = table_client[f'{selected_commodity}/{data_type}']

           # Overall statistics for the commodity
           total_value = df_commodity['value'].sum()
           avg_monthly = df_commodity['value'].mean()

           # Country-specific statistics
           country_total = 0
           country_avg = 0
           country_share = 0
           if selected_country and selected_country in df_commodity.index.get_level_values('Partner'):
               country_data = df_commodity.loc[selected_country]
               country_total = country_data['value'].sum()
               country_avg = country_data['value'].mean()
               country_share = (country_total / total_value) * 100 if total_value > 0 else 0

           # Year-specific statistics
           year_data = df_commodity[df_commodity.index.get_level_values('date').year == selected_year]
           year_total = year_data['value'].sum()
           top_country = "N/A"
           num_sources = 0
           if not year_data.empty:
               top_country = year_data.groupby('Partner')['value'].sum().idxmax()
               num_sources = len(year_data.index.get_level_values('Partner').unique())

           return html.Div([
               html.Div([
                   html.Div([
                       html.H4(f"{selected_commodity.title()} - Overall Statistics", style={'color': colors['primary']}),
                       html.P(f"Total {data_type.title()}: ${total_value:,.0f}", style={'color': colors['text']}),
                       html.P(f"Average Monthly: ${avg_monthly:,.0f}", style={'color': colors['text']})
                   ], className="four columns"),

                   html.Div([
                       html.H4(f"{selected_country} Statistics", style={'color': colors['secondary']}),
                       html.P(f"Total {data_type.title()}: ${country_total:,.0f}", style={'color': colors['text']}),
                       html.P(f"Average Monthly: ${country_avg:,.0f}", style={'color': colors['text']}),
                       html.P(f"Share of Total: {country_share:.1f}%", style={'color': colors['text']})
                   ], className="four columns"),

                   html.Div([
                       html.H4(f"{selected_year} Statistics", style={'color': colors['accent']}),
                       html.P(f"Year Total: ${year_total:,.0f}", style={'color': colors['text']}),
                       html.P(f"Top Source: {top_country}", style={'color': colors['text']}),
                       html.P(f"Number of Sources: {num_sources}", style={'color': colors['text']})
                   ], className="four columns")
               ], className="row")
           ])
