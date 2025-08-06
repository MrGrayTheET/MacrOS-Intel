import dash
from dash import Input, Output, State, callback_context
import pandas as pd
from pages.esr.esr_utils import get_sample_esr_data, get_multi_year_esr_data, create_empty_figure
import plotly.express as px
from components.frames import EnhancedFrameGrid
from data.data_tables import ESRTableClient

table_client = ESRTableClient()



def sales_trends_chart_update(chart_id: str, **menu_values):
        """Update function for Sales Trends page"""
        commodity = menu_values.get('commodity', 'cattle')
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
        start_year = menu_values.get('start_year')
        end_year = menu_values.get('end_year')

        data = table_client.get_multi_year_esr_data(
            commodity=commodity, 
            start_year=start_year, 
            end_year=end_year
        )

        if countries:
            data = data[data['country'].isin(countries)]

        if data.empty:
            return create_empty_figure(f"{commodity.title()} - Sales Trends")

        # Chart-to-metric mapping
        if chart_id == 'esr_sales_trends_chart_0':
            metric = 'weeklyExports'
            metric_name = 'Weekly Exports'
        elif chart_id == 'esr_sales_trends_chart_1':
            metric = 'outstandingSales'
            metric_name = 'Outstanding Sales'
        elif chart_id == 'esr_sales_trends_chart_2':
            metric = 'grossNewSales'
            metric_name = 'Gross New Sales'
        else:
            metric = 'weeklyExports'
            metric_name = 'Weekly Exports'

        fig = px.line(
            data,
            x='weekEndingDate',
            y=metric,
            color='country',
            title=f"{commodity.title()} - {metric_name}",
            markers=True
        )

        fig.update_layout(
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

def country_analysis_chart_update(chart_id: str, **menu_values):
    """Update function for Country Analysis page"""
    commodity = menu_values.get('commodity', 'cattle')
    country = menu_values.get('country', 'Korea, South')
    start_year = menu_values.get('start_year')
    end_year = menu_values.get('end_year')

    data = table_client.get_multi_year_esr_data(
        commodity=commodity,
        country=country,
        start_year=start_year,
        end_year=end_year
    )
    data = data[data['country'] == country]

    if data.empty:
        return create_empty_figure(f"{commodity.title()} - {country} Analysis")

    # Chart-to-metric mapping
    if chart_id == 'esr_country_analysis_chart_0':
        metric = 'weeklyExports'
        metric_name = 'Weekly Exports'
    elif chart_id == 'esr_country_analysis_chart_1':
        metric = 'outstandingSales'
        metric_name = 'Outstanding Sales'
    else:
        metric = 'weeklyExports'
        metric_name = 'Weekly Exports'

    fig = px.line(
        data,
        x='weekEndingDate',
        y=metric,
        color='marketing_year',
        title=f"{commodity.title()} - {country} {metric_name} (5-Year)",
        markers=True
    )

    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def commitment_analysis_chart_update(chart_id: str, **menu_values):
    """Update function for Commitment Analysis page"""
    commodity = menu_values.get('commodity', 'cattle')
    start_year = menu_values.get('start_year')
    end_year = menu_values.get('end_year')
    countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])

    data = table_client.get_multi_year_esr_data(
        commodity=commodity,
        start_year=start_year,
        end_year=end_year
    )

    if countries:
        data = data[data['country'].isin(countries)]

    if data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - Commitment Analysis ({year_range})")

    # Chart-to-metric and chart-type mapping
    if chart_id == 'esr_commitment_analysis_chart_0':
        metric = 'currentMYTotalCommitment'
        metric_name = 'Current MY Total Commitment'
        chart_type = 'area'
    elif chart_id == 'esr_commitment_analysis_chart_1':
        metric = 'currentMYNetSales'
        metric_name = 'Current MY Net Sales'
        chart_type = 'line'
    elif chart_id == 'esr_commitment_analysis_chart_2':
        metric = 'nextMYOutstandingSales'
        metric_name = 'Next MY Outstanding Sales'
        chart_type = 'bar'
    elif chart_id == 'esr_commitment_analysis_chart_3':
        metric = 'nextMYNetSales'
        metric_name = 'Next MY Net Sales'
        chart_type = 'line'
    else:
        metric = 'currentMYTotalCommitment'
        metric_name = 'Current MY Total Commitment'
        chart_type = 'area'

    # Create chart based on type
    if chart_type == 'area':
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        fig = px.area(data, x='weekEndingDate', y=metric, color='country',
                      title=f"{commodity.title()} - {metric_name} ({year_range})")
    elif chart_type == 'bar':
        fig = px.bar(data, x='weekEndingDate', y=metric, color='country',
                     title=f"{commodity.title()} - {metric_name} ({year_range})")
    else:  # line
        fig = px.line(data, x='weekEndingDate', y=metric, color='country',
                      title=f"{commodity.title()} - {metric_name} ({year_range})", markers=True)

    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def comparative_analysis_chart_update(chart_id: str, **menu_values):
        """Update function for Comparative Analysis page"""
        commodity_a = menu_values.get('commodity_a', 'cattle')
        commodity_b = menu_values.get('commodity_b', 'corn')
        start_year = menu_values.get('start_year')
        end_year = menu_values.get('end_year')
        metric = menu_values.get('metric', 'weeklyExports')
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])

        # Determine commodity based on frame
        if 'comparison_frame1' in chart_id:
            commodity = commodity_a
            frame_label = "A"
        else:
            commodity = commodity_b
            frame_label = "B"

        data = table_client.get_multi_year_esr_data(
            commodity=commodity,
            start_year=start_year,
            end_year=end_year
        )

        if countries:
            data = data[data['country'].isin(countries)]

        if data.empty:
            return create_empty_figure(f"Commodity {frame_label}: {commodity.title()}")

        metric_name = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales'
        }.get(metric, metric.replace('_', ' ').title())

        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        fig = px.line(
            data,
            x='weekEndingDate',
            y=metric,
            color='country',
            title=f"Commodity {frame_label}: {commodity.title()} - {metric_name} ({year_range})",
            markers=True
        )

        fig.update_layout(
            height=350,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig