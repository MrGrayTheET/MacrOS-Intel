import dash
from dash import Input, Output, State, callback_context
import pandas as pd
from pages.esr.esr_utils import get_sample_esr_data, get_multi_year_esr_data, create_empty_figure
import plotly.express as px
from plotly import graph_objects as go
from components.frames import EnhancedFrameGrid
from data.data_tables import ESRTableClient

table_client = ESRTableClient()


def create_seasonal_summary_table(commodity, results, seasonal_metric):
    """Create seasonal summary table data from analysis results."""
    # Marketing year information
    marketing_years = {
        'cattle': ('January 1', 'December 31'),
        'hogs': ('January 1', 'December 31'),
        'pork': ('January 1', 'December 31'),
        'corn': ('September 1', 'August 31'),
        'wheat': ('September 1', 'August 31'),
        'soybeans': ('September 1', 'August 31')
    }
    
    my_start, my_end = marketing_years.get(commodity.lower(), ('September 1', 'August 31'))
    
    if 'error' in results:
        return [{
            'commodity': commodity.title(),
            'my_start': my_start,
            'my_end': my_end,
            'peak_weeks': 'No data available',
            'low_weeks': 'No data available',
            'seasonality': 0
        }]
    
    # Extract peak and low weeks
    peak_weeks = results.get('peak_weeks', [])
    low_weeks = results.get('low_weeks', [])
    seasonality_strength = results.get('seasonality_strength', 0)
    
    # Format week ranges
    peak_str = f"Weeks {min(peak_weeks)}-{max(peak_weeks)}" if peak_weeks else "N/A"
    low_str = f"Weeks {min(low_weeks)}-{max(low_weeks)}" if low_weeks else "N/A"
    
    return [{
        'commodity': commodity.title(),
        'my_start': my_start,
        'my_end': my_end,
        'peak_weeks': peak_str,
        'low_weeks': low_str,
        'seasonality': round(seasonality_strength, 4) if seasonality_strength else 0
    }]



def sales_trends_chart_update(chart_id: str, store_data=None, **menu_values):
        """Update function for Sales Trends page - supports store data"""
        commodity = menu_values.get('commodity', 'cattle')
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
        
        # Use store data if available, otherwise fall back to table client
        if store_data:
            try:
                data = pd.read_json(store_data, orient='records')
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
            except Exception as e:
                print(f"Error loading store data: {e}")
                data = pd.DataFrame()
        else:
            # Safely convert years to integers
            try:
                start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
            except (ValueError, TypeError):
                print(f"Error converting start_year: {menu_values.get('start_year')}")
                start_year = None
                
            try:
                end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
            except (ValueError, TypeError):
                print(f"Error converting end_year: {menu_values.get('end_year')}")
                end_year = None

            data = table_client.get_multi_year_esr_data(
                commodity=commodity, 
                start_year=start_year, 
                end_year=end_year
            )

        # Apply filtering
        if not data.empty and countries:
            data = data[data['country'].isin(countries)]
            
        # Apply year filtering if using store data
        if store_data and not data.empty:
            try:
                start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
                end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
                
                if start_year and end_year:
                    data = data[
                        (data['weekEndingDate'].dt.year >= start_year) & 
                        (data['weekEndingDate'].dt.year <= end_year)
                    ]
            except (ValueError, TypeError):
                pass

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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return fig

def country_analysis_chart_update(chart_id: str, store_data=None, **menu_values):
    """Update function for Country Analysis page - supports store data"""
    commodity = menu_values.get('commodity', 'cattle')
    country = menu_values.get('country', 'Korea, South')
    
    # Use store data if available, otherwise fall back to table client
    if store_data:
        try:
            data = pd.read_json(store_data, orient='records')
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            data = pd.DataFrame()
    else:
        # Safely convert years to integers
        try:
            start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
        except (ValueError, TypeError):
            start_year = None
            
        try:
            end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
        except (ValueError, TypeError):
            end_year = None

        data = table_client.get_multi_year_esr_data(
            commodity=commodity,
            country=country,
            start_year=start_year,
            end_year=end_year
        )
    
    # Apply filtering
    if not data.empty and country:
        data = data[data['country'] == country]
        
    # Apply year filtering if using store data
    if store_data and not data.empty:
        try:
            start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
            end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
            
            if start_year and end_year:
                data = data[
                    (data['weekEndingDate'].dt.year >= start_year) & 
                    (data['weekEndingDate'].dt.year <= end_year)
                ]
        except (ValueError, TypeError):
            pass

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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def dual_frame_commitment_analysis_chart_update(chart_id: str, store_data=None, **menu_values):
    """Update function for dual frame Commitment Analysis page - handles both metric chart and analytics charts."""
    commodity = menu_values.get('commodity', 'cattle')
    country_selection = menu_values.get('country_selection', 'Korea, South')
    countries = menu_values.get('countries', ['Korea, South'])
    commitment_metric = menu_values.get('commitment_metric', 'currentMYTotalCommitment')
    
    # Debug print for troubleshooting
    print(f"DEBUG dual_frame commitment_analysis: chart_id={chart_id}, commodity={commodity}, menu_values keys={list(menu_values.keys())}")
    
    # Ensure commodity is a string
    if not isinstance(commodity, str) or not commodity:
        commodity = 'cattle'
    
    # Safely convert years to integers
    try:
        start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
    except (ValueError, TypeError):
        start_year = None
        
    try:
        end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
    except (ValueError, TypeError):
        end_year = None

    # Get multi-year data
    data = table_client.get_multi_year_esr_data(
        commodity=commodity,
        start_year=start_year,
        end_year=end_year
    )

    if data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - Commitment Analysis ({year_range})")

    # Handle country selection
    if country_selection == 'ALL_COUNTRIES':
        # Sum of multiple countries
        filtered_data = data[data['country'].isin(countries)]
        if not filtered_data.empty:
            # Use ESRAnalyzer to aggregate multi-country data
            from models.commodity_analytics import ESRAnalyzer
            aggregated_data = ESRAnalyzer.aggregate_multi_country_data(filtered_data, countries)
            title_suffix = f"All Selected Countries ({len(countries)})"
            chart_data = aggregated_data
        else:
            return create_empty_figure(f"{commodity.title()} - No data for selected countries")
    else:
        # Single country
        chart_data = data[data['country'] == country_selection]
        title_suffix = country_selection

    if chart_data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - {title_suffix} ({year_range})")

    # Determine which chart is being updated based on frame0 and frame1 structure
    if 'esr_commitment_frame0_chart_0' in chart_id:
        # Frame 0, Chart 0 - selectable commitment metric
        metric_names = {
            'currentMYTotalCommitment': 'MY Total Commitment',
            'currentMYNetSales': 'MY Net Sales',
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'nextMYOutstandingSales': 'Next MY Outstanding Sales'
        }
        
        metric_name = metric_names.get(commitment_metric, commitment_metric.replace('_', ' ').title())
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        
        # Check if metric exists in data
        if commitment_metric not in chart_data.columns:
            return create_empty_figure(f"{commodity.title()} - {metric_name} not available ({year_range}) - {title_suffix}")
        
        fig = px.line(
            chart_data,
            x='weekEndingDate',
            y=commitment_metric,
            title=f'{commodity.title()} - {metric_name} ({year_range}) - {title_suffix}',
            markers=True
        )
        
        fig.update_layout(
            height=350,
            hovermode='x unified',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Week Ending Date",
            yaxis_title=metric_name
        )
        
        return fig
        
    elif ('esr_commitment_frame0_chart_1' in chart_id or 
          'esr_commitment_frame1_chart_0' in chart_id or 
          'esr_commitment_frame1_chart_1' in chart_id):
        # Analytics charts - commitment vs shipment analysis
        from models.commodity_analytics import ESRAnalyzer
        
        # Initialize analyzer with the data
        analyzer = ESRAnalyzer(chart_data.set_index('weekEndingDate'), 'grains')
        
        # Get commitment analysis results
        if country_selection == 'ALL_COUNTRIES':
            # For aggregated data, pass the countries list
            results = analyzer.commitment_vs_shipment_analysis(countries=countries, commodity=commodity)
        else:
            # For single country
            results = analyzer.commitment_vs_shipment_analysis(country=country_selection, commodity=commodity)
        
        if 'data' not in results:
            year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
            return create_empty_figure(f"{commodity.title()} - No Analysis Data ({year_range}) - {title_suffix}")
        
        results_data = results['data']
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        
        if 'esr_commitment_frame0_chart_1' in chart_id:
            # Sales Backlog Analysis
            if 'sales_backlog' in results_data.columns:
                fig = px.line(
                    results_data.reset_index(),
                    x='weekEndingDate',
                    y='sales_backlog',
                    title=f'{commodity.title()} - Sales Backlog Analysis ({year_range}) - {title_suffix}',
                    markers=True
                )
                
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Week Ending Date",
                    yaxis_title="Sales Backlog (Weeks)"
                )
                
                return fig
        
        elif 'esr_commitment_frame1_chart_0' in chart_id:
            # Commitment Utilization Rate
            if 'commitment_utilization' in results_data.columns:
                fig = px.line(
                    results_data.reset_index(),
                    x='weekEndingDate',
                    y='commitment_utilization',
                    title=f'{commodity.title()} - Commitment Utilization Rate ({year_range}) - {title_suffix}',
                    markers=True
                )
                
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Week Ending Date",
                    yaxis_title="Utilization Rate"
                )
                
                return fig
        
        elif 'esr_commitment_frame1_chart_1' in chart_id:
            # Export Fulfillment Rate
            if 'fulfillment_rate' in results_data.columns:
                fig = px.line(
                    results_data.reset_index(),
                    x='weekEndingDate',
                    y='fulfillment_rate',
                    title=f'{commodity.title()} - Export Fulfillment Rate ({year_range}) - {title_suffix}',
                    markers=True
                )
                
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Week Ending Date",
                    yaxis_title="Fulfillment Rate"
                )
                
                return fig
        
        # Fallback to empty figure if we get here
        return create_empty_figure(f"{commodity.title()} - Analytics Data Not Available ({year_range}) - {title_suffix}")
        
    # Fallback for unrecognized chart_id
    return create_empty_figure(f"{commodity.title()} - Unknown Chart ({chart_id})")

def new_commitment_analysis_chart_update(chart_id: str, store_data=None, **menu_values):
    """Update function for new Commitment Analysis page - handles both metric chart and analytics charts."""
    commodity = menu_values.get('commodity', 'cattle')
    country_selection = menu_values.get('country_selection', 'Korea, South')
    countries = menu_values.get('countries', ['Korea, South'])
    commitment_metric = menu_values.get('commitment_metric', 'currentMYTotalCommitment')
    
    # Debug print for troubleshooting
    print(f"DEBUG commitment_analysis: chart_id={chart_id}, commodity={commodity}, menu_values keys={list(menu_values.keys())}")
    
    # Ensure commodity is a string
    if not isinstance(commodity, str) or not commodity:
        commodity = 'cattle'
    
    # Safely convert years to integers
    try:
        start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
    except (ValueError, TypeError):
        start_year = None
        
    try:
        end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
    except (ValueError, TypeError):
        end_year = None

    # Get multi-year data
    data = table_client.get_multi_year_esr_data(
        commodity=commodity,
        start_year=start_year,
        end_year=end_year
    )

    if data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - Commitment Analysis ({year_range})")

    # Handle country selection
    if country_selection == 'ALL_COUNTRIES':
        # Sum of multiple countries
        filtered_data = data[data['country'].isin(countries)]
        if not filtered_data.empty:
            # Use ESRAnalyzer to aggregate multi-country data
            from models.commodity_analytics import ESRAnalyzer
            aggregated_data = ESRAnalyzer.aggregate_multi_country_data(filtered_data, countries)
            title_suffix = f"All Selected Countries ({len(countries)})"
            chart_data = aggregated_data
        else:
            return create_empty_figure(f"{commodity.title()} - No data for selected countries")
    else:
        # Single country
        chart_data = data[data['country'] == country_selection]
        title_suffix = country_selection

    if chart_data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - {title_suffix} ({year_range})")

    # Determine which chart is being updated
    if 'esr_commitment_frame1_chart_0' in chart_id:
        # First chart (Frame 1, Chart 0) - selectable commitment metric
        metric_names = {
            'currentMYTotalCommitment': 'MY Total Commitment',
            'currentMYNetSales': 'MY Net Sales', 
            'outstandingSales': 'MY Outstanding Sales',
            'nextMYOutstandingSales': 'Next MY Outstanding Sales'
        }
        
        metric_name = metric_names.get(commitment_metric, commitment_metric.replace('_', ' ').title())
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        
        if commitment_metric not in chart_data.columns:
            return create_empty_figure(f"Column '{commitment_metric}' not available")
        
        fig = px.line(
            chart_data,
            x='weekEndingDate',
            y=commitment_metric,
            title=f'{commodity.title()} - {metric_name} ({year_range}) - {title_suffix}',
            markers=True
        )
        
        fig.update_layout(
            height=400,
            hovermode='x unified',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Week Ending Date",
            yaxis_title=metric_name
        )
        
        return fig
        
    elif ('esr_commitment_frame1_chart_1' in chart_id or 
          'esr_commitment_frame2_chart_0' in chart_id or 
          'esr_commitment_frame2_chart_1' in chart_id):
        # Analytics charts - commitment vs shipment analysis
        from models.commodity_analytics import ESRAnalyzer
        
        # Initialize analyzer with the data
        analyzer = ESRAnalyzer(chart_data.set_index('weekEndingDate'), 'grains')
        
        # Get commitment analysis results
        if country_selection == 'ALL_COUNTRIES':
            results = analyzer.commitment_vs_shipment_analysis(countries=countries, commodity=commodity)
        else:
            results = analyzer.commitment_vs_shipment_analysis(country=country_selection, commodity=commodity)
        
        if 'error' in results:
            year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
            return create_empty_figure(f"{commodity.title()} - Analysis Error ({year_range}) - {title_suffix}")
        
        # Access the data from results['data']
        results_data = results.get('data', pd.DataFrame())
        if results_data.empty:
            year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
            return create_empty_figure(f"{commodity.title()} - No Analysis Data ({year_range}) - {title_suffix}")
        
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        
        if 'esr_commitment_frame1_chart_1' in chart_id:
            # Commitment Utilization Rate
            if 'commitment_utilization' in results_data.columns:
                fig = px.line(
                    results_data.reset_index(),
                    x='weekEndingDate',
                    y='commitment_utilization',
                    title=f'{commodity.title()} - Commitment Utilization Rate ({year_range}) - {title_suffix}',
                    markers=True
                )
                
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Week Ending Date",
                    yaxis_title="Utilization Rate"
                )
                
                return fig
        
        elif 'esr_commitment_frame2_chart_0' in chart_id:
            # Export Fulfillment Rate
            if 'fulfillment_rate' in results_data.columns:
                fig = px.line(
                    results_data.reset_index(),
                    x='weekEndingDate',
                    y='fulfillment_rate',
                    title=f'{commodity.title()} - Export Fulfillment Rate ({year_range}) - {title_suffix}',
                    markers=True
                )
                
                fig.update_layout(
                    height=350,
                    hovermode='x unified',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title="Week Ending Date",
                    yaxis_title="Fulfillment Rate"
                )
                
                return fig
        
        elif 'esr_commitment_frame2_chart_1' in chart_id:
            # Sales Backlog Analysis
            try:
                # Check if sales_backlog column exists
                if 'sales_backlog' in results_data.columns:
                    print(f"DEBUG: Found sales_backlog column in results_data")
                    
                    fig = px.line(
                        results_data.reset_index(),
                        x='weekEndingDate',
                        y='sales_backlog',
                        title=f'{str(commodity).title()} - Sales Backlog ({year_range}) - {title_suffix}',
                        markers=True
                    )
                    
                    fig.update_layout(
                        height=350,
                        hovermode='x unified',
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis_title="Week Ending Date",
                        yaxis_title="Sales Backlog (Weeks)"
                    )
                    
                    return fig
                else:
                    print(f"DEBUG: sales_backlog column not found. Available columns: {list(results_data.columns)}")
                    # Try sales_backlog_weeks instead
                    if 'sales_backlog_weeks' in results_data.columns:
                        fig = px.line(
                            results_data.reset_index(),
                            x='weekEndingDate',
                            y='sales_backlog_weeks',
                            title=f'{str(commodity).title()} - Sales Backlog ({year_range}) - {title_suffix}',
                            markers=True
                        )
                        
                        fig.update_layout(
                            height=350,
                            hovermode='x unified',
                            template='plotly_dark',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis_title="Week Ending Date",
                            yaxis_title="Sales Backlog (Weeks)"
                        )
                        
                        return fig
                    else:
                        return create_empty_figure(f"{str(commodity).title()} - Sales Backlog Column Missing")
            except Exception as e:
                print(f"DEBUG: Error in chart_3: {str(e)}")
                return create_empty_figure(f"{str(commodity).title()} - Sales Backlog Error: {str(e)}")
    
    # Fallback
    year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
    return create_empty_figure(f"{str(commodity).title()} - Chart Update Error ({year_range})")


def commitment_analysis_chart_update(chart_id: str, **menu_values):
    """Update function for Commitment Analysis page"""
    commodity = menu_values.get('commodity', 'cattle')
    countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
    
    # Safely convert years to integers
    try:
        start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
    except (ValueError, TypeError):
        start_year = None
        
    try:
        end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
    except (ValueError, TypeError):
        end_year = None

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
    year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
    
    if chart_type == 'area':
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def comparative_analysis_chart_update(chart_id: str, store_data=None, **menu_values):
        """Update function for Comparative Analysis page - supports store data"""
        commodity_a = menu_values.get('commodity_a', 'cattle')
        commodity_b = menu_values.get('commodity_b', 'corn')
        metric = menu_values.get('metric', 'weeklyExports')
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
        
        # Safely convert years to integers
        try:
            start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
        except (ValueError, TypeError):
            start_year = None
            
        try:
            end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
        except (ValueError, TypeError):
            end_year = None

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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return fig


def seasonal_analysis_chart_update(chart_id: str, store_data=None, **menu_values):
    """Update function for Seasonal Analysis page - supports store data"""
    commodity = menu_values.get('commodity', 'cattle')
    country_selection = menu_values.get('country_selection', 'Korea, South')
    countries = menu_values.get('countries', ['Korea, South'])
    seasonal_metric = menu_values.get('seasonal_metric', 'weeklyExports')
    
    # Safely convert years to integers
    try:
        start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
    except (ValueError, TypeError):
        start_year = None
        
    try:
        end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
    except (ValueError, TypeError):
        end_year = None

    # Use seasonal patterns analysis from ESRTableClient
    if country_selection == 'ALL_COUNTRIES':
        # Sum of multiple countries
        results = table_client.get_seasonal_patterns_analysis(
            commodity=commodity, 
            metric=seasonal_metric,
            start_year=start_year, 
            end_year=end_year,
            countries=countries
        )
        title_suffix = f"All Selected Countries ({len(countries)})"
    else:
        # Single country
        results = table_client.get_seasonal_patterns_analysis(
            commodity=commodity, 
            metric=seasonal_metric,
            start_year=start_year, 
            end_year=end_year,
            countries=[country_selection]
        )
        title_suffix = country_selection

    if 'error' in results:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - Seasonal Analysis ({year_range}) - {title_suffix}")

    # Get the processed data with marketing year weeks
    data = results.get('data', pd.DataFrame())
    
    if data.empty:
        year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
        return create_empty_figure(f"{commodity.title()} - Seasonal Analysis ({year_range}) - {title_suffix}")

    # Aggregate data by marketing year week
    if country_selection == 'ALL_COUNTRIES':
        # Sum across countries
        seasonal_data = data.groupby('my_week')[seasonal_metric].sum().reset_index()
        seasonal_data['country'] = 'All Countries'
    else:
        # Single country data
        country_data = data[data['country'] == country_selection]
        seasonal_data = country_data.groupby('my_week')[seasonal_metric].mean().reset_index()
        seasonal_data['country'] = country_selection

    # Create seasonal line chart
    metric_names = {
        'weeklyExports': 'Weekly Exports',
        'outstandingSales': 'Outstanding Sales', 
        'grossNewSales': 'Gross New Sales',
        'currentMYNetSales': 'Current MY Net Sales'
    }
    
    metric_name = metric_names.get(seasonal_metric, seasonal_metric.replace('_', ' ').title())
    year_range = f"{start_year}-{end_year}" if start_year and end_year else "Multi-Year"
    
    fig = px.line(
        seasonal_data,
        x='my_week',
        y=seasonal_metric,
        title=f'{commodity.title()} - {metric_name} Seasonal Pattern ({year_range}) - {title_suffix}',
        labels={
            'my_week': 'Marketing Year Week',
            seasonal_metric: metric_name
        }
    )
    
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_title="Marketing Year Week",
        yaxis_title=metric_name,
        showlegend=False
    )
    
    # Add markers to show individual weeks
    fig.update_traces(mode='lines+markers')
    
    return fig


def seasonal_analysis_table_update(store_data=None, **menu_values):
    """Update function for Seasonal Analysis table - supports store data"""
    commodity = menu_values.get('commodity', 'cattle')
    country_selection = menu_values.get('country_selection', 'Korea, South')
    countries = menu_values.get('countries', ['Korea, South'])
    seasonal_metric = menu_values.get('seasonal_metric', 'weeklyExports')
    
    # Safely convert years to integers
    try:
        start_year = int(menu_values.get('start_year')) if menu_values.get('start_year') else None
    except (ValueError, TypeError):
        start_year = None
        
    try:
        end_year = int(menu_values.get('end_year')) if menu_values.get('end_year') else None
    except (ValueError, TypeError):
        end_year = None

    # Use seasonal patterns analysis from ESRTableClient
    if country_selection == 'ALL_COUNTRIES':
        # Sum of multiple countries
        results = table_client.get_seasonal_patterns_analysis(
            commodity=commodity, 
            metric=seasonal_metric,
            start_year=start_year, 
            end_year=end_year,
            countries=countries
        )
    else:
        # Single country
        results = table_client.get_seasonal_patterns_analysis(
            commodity=commodity, 
            metric=seasonal_metric,
            start_year=start_year, 
            end_year=end_year,
            countries=[country_selection]
        )
    
    # Create table data
    table_data = create_seasonal_summary_table(commodity, results, seasonal_metric)
    return table_data