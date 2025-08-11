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



def sales_trends_chart_update(chart_ids, store_data=None, **menu_values):
        """Update function for Sales Trends page - handles multiple charts from single callback"""
        try:
            # Handle both single chart_id and list of chart_ids
            if isinstance(chart_ids, str):
                chart_ids = [chart_ids]
            
            # Get menu values
            countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
            country_display_mode = menu_values.get('country_display_mode', 'individual')
            date_range = menu_values.get('date_range', [])
            
            # Get column selections for each chart
            chart_0_column = menu_values.get('chart_0_column', 'weeklyExports')
            chart_1_column = menu_values.get('chart_1_column', 'outstandingSales')
            chart_2_column = menu_values.get('chart_2_column', 'grossNewSales')
            
            # Use store data
            if not store_data:
                error_figs = [create_empty_figure("No data available in store") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
            
            try:
                if isinstance(store_data, str):
                    import json
                    data = pd.DataFrame(json.loads(store_data))
                else:
                    data = pd.DataFrame(store_data)
                
                # Debug: Print available columns
                print(f"DEBUG sales_trends - Available columns in store data: {list(data.columns) if not data.empty else 'No data'}")
                print(f"DEBUG sales_trends - Data shape: {data.shape if not data.empty else 'No data'}")
                
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
            except Exception as e:
                print(f"Error loading store data: {e}")
                error_figs = [create_empty_figure(f"Error loading data: {str(e)}") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
                
            if data.empty:
                error_figs = [create_empty_figure("No data available") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
            
            # Filter by countries
            if countries and 'country' in data.columns:
                data = data[data['country'].isin(countries)]
            
            # Apply date range filter if provided
            if date_range and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                data = data[
                    (data['weekEndingDate'] >= start_date) & 
                    (data['weekEndingDate'] <= end_date)
                ]
            
            if data.empty:
                error_figs = [create_empty_figure("No data for selected criteria") for _ in chart_ids]
                return error_figs[0] if len(chart_ids) == 1 else error_figs
            
            # Column mapping for each chart
            column_mapping = {
                'esr_sales_trends_chart_0': chart_0_column,
                'esr_sales_trends_chart_1': chart_1_column,
                'esr_sales_trends_chart_2': chart_2_column
            }
            
            column_labels = {
                'weeklyExports': 'Weekly Exports',
                'outstandingSales': 'Outstanding Sales',
                'grossNewSales': 'Gross New Sales',
                'currentMYNetSales': 'Current MY Net Sales',
                'currentMYTotalCommitment': 'Current MY Total Commitment'
            }
            
            # Generate figures for each chart
            figures = []
            
            for chart_id in chart_ids:
                y_column = column_mapping.get(chart_id, 'weeklyExports')
                
                # Check if column exists
                if y_column not in data.columns:
                    figures.append(create_empty_figure(f"Column '{y_column}' not found in data"))
                    continue
                
                chart_title = column_labels.get(y_column, y_column.replace('_', ' ').title())
                
                # Handle country display mode
                if country_display_mode == 'sum' and len(countries) > 1:
                    # Sum all countries together
                    data_grouped = data.groupby('weekEndingDate')[y_column].sum().reset_index()
                    data_grouped['country'] = f"Sum of {', '.join(countries)}"
                    chart_data = data_grouped
                    
                    fig = px.line(
                        chart_data,
                        x='weekEndingDate',
                        y=y_column,
                        title=f"{chart_title} - {chart_data['country'].iloc[0]}",
                        markers=True
                    )
                    fig.update_traces(line=dict(color='#1f77b4', width=3))
                else:
                    # Individual countries
                    fig = px.line(
                        data,
                        x='weekEndingDate',
                        y=y_column,
                        color='country',
                        title=f"{chart_title} by Country",
                        markers=True
                    )
                
                # Update layout
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    xaxis_title='Week Ending Date',
                    yaxis_title=chart_title,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                figures.append(fig)
            
            # Return single figure or list of figures
            return figures[0] if len(chart_ids) == 1 else figures
            
        except Exception as e:
            print(f"Error in sales_trends_chart_update: {e}")
            error_figs = [create_empty_figure(f"Error: {str(e)}") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs

def country_analysis_chart_update(chart_ids, store_data=None, **menu_values):
    """
    Update function for Country Analysis page - supports multi-chart and store data with market year overlays.
    Enhanced to support both single chart_id and list of chart_ids for multi-chart updates.
    Supports dynamic country selection and market year overlay functionality.
    """
    try:
        # Handle both single chart_id and list of chart_ids
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        country_metric = menu_values.get('country_metric', 'weeklyExports')
        date_range = menu_values.get('date_range', [])
        start_year = menu_values.get('start_year')
        end_year = menu_values.get('end_year')
        
        print(f"DEBUG country_analysis: chart_ids={chart_ids}, countries={countries}")
        
        # Use store data
        if not store_data:
            error_figs = [create_empty_figure("No data available in store") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            error_figs = [create_empty_figure(f"Error loading data: {str(e)}") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
            
        if data.empty:
            error_figs = [create_empty_figure("No data available") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            error_figs = [create_empty_figure("No data for selected criteria") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Generate figures for each chart
        figures = []
        
        for i, chart_id in enumerate(chart_ids):
            fig = create_country_analysis_chart(
                data, country_metric, countries, country_display_mode, 
                start_year, end_year, chart_id, i
            )
            figures.append(fig)
        
        # Return single figure or list of figures
        return figures[0] if len(chart_ids) == 1 else figures
        
    except Exception as e:
        print(f"Error in country_analysis_chart_update: {e}")
        error_figs = [create_empty_figure(f"Error: {str(e)}") for _ in chart_ids]
        return error_figs[0] if len(chart_ids) == 1 else error_figs

def dual_frame_commitment_analysis_chart_update(chart_ids, store_data=None, **menu_values):
    """
    Update function for dual frame Commitment Analysis page - handles multiple charts.
    Enhanced to support both single chart_id and list of chart_ids for multi-chart updates.
    Frame 0 Chart 0: Store data with column select
    Other charts: Analytics with sales_backlog, fulfillment_rate, commitment_utilization
    """
    try:
        # Handle both single chart_id and list of chart_ids
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        commitment_metric = menu_values.get('commitment_metric', 'currentMYTotalCommitment')
        date_range = menu_values.get('date_range', [])
        
        print(f"DEBUG commitment_analysis: chart_ids={chart_ids}, countries={countries}")
        
        # Use store data
        if not store_data:
            error_figs = [create_empty_figure("No data available in store") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            # Debug: Print available columns for commitment analysis
            print(f"DEBUG commitment_analysis - Available columns: {list(data.columns) if not data.empty else 'No data'}")
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            error_figs = [create_empty_figure(f"Error loading data: {str(e)}") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
            
        if data.empty:
            error_figs = [create_empty_figure("No data available") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            error_figs = [create_empty_figure("No data for selected criteria") for _ in chart_ids]
            return error_figs[0] if len(chart_ids) == 1 else error_figs
        
        # Generate figures for each chart
        figures = []
        
        for chart_id in chart_ids:
            if 'esr_commitment_frame0_chart_0' in chart_id:
                # Frame 0, Chart 0 - selectable commitment metric from store data
                fig = create_store_based_commitment_chart(
                    data, commitment_metric, countries, country_display_mode, chart_id
                )
            else:
                # Other charts - analytics (sales_backlog, fulfillment_rate, commitment_utilization)
                fig = create_analytics_commitment_chart(
                    data, countries, country_display_mode, chart_id
                )
            
            figures.append(fig)
        
        # Return single figure or list of figures
        return figures[0] if len(chart_ids) == 1 else figures
        
    except Exception as e:
        print(f"Error in dual_frame_commitment_analysis_chart_update: {e}")
        error_figs = [create_empty_figure(f"Error: {str(e)}") for _ in chart_ids]
        return error_figs[0] if len(chart_ids) == 1 else error_figs


def create_store_based_commitment_chart(data, commitment_metric, countries, country_display_mode, chart_id):
    """Create chart for Frame 0 Chart 0 using store data with column selection"""
    try:
        # Check if column exists
        if commitment_metric not in data.columns:
            return create_empty_figure(f"Column '{commitment_metric}' not found in data")
        
        # Create chart title
        column_labels = {
            'currentMYTotalCommitment': 'Current MY Total Commitment',
            'currentMYNetSales': 'Current MY Net Sales',
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'nextMYOutstandingSales': 'Next MY Outstanding Sales'
        }
        chart_title = column_labels.get(commitment_metric, commitment_metric.replace('_', ' ').title())
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Sum all countries together
            data_grouped = data.groupby('weekEndingDate')[commitment_metric].sum().reset_index()
            data_grouped['country'] = f"Sum of {', '.join(countries)}"
            chart_data = data_grouped
            
            fig = px.line(
                chart_data,
                x='weekEndingDate',
                y=commitment_metric,
                title=f"{chart_title} - {chart_data['country'].iloc[0]}",
                markers=True
            )
            fig.update_traces(line=dict(color='#1f77b4', width=3))
        else:
            # Individual countries
            fig = px.line(
                data,
                x='weekEndingDate',
                y=commitment_metric,
                color='country',
                title=f"{chart_title} by Country",
                markers=True
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=350,
            xaxis_title='Week Ending Date',
            yaxis_title=chart_title,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_store_based_commitment_chart: {e}")
        return create_empty_figure(f"Error: {str(e)}")


def create_analytics_commitment_chart(data, countries, country_display_mode, chart_id):
    """Create analytics charts for sales_backlog, fulfillment_rate, commitment_utilization"""
    try:
        # Perform commitment vs shipment analysis
        from models.commodity_analytics import ESRAnalyzer
        
        # Handle country display mode for analysis
        analysis_data = data.copy()
        if country_display_mode == 'sum' and len(countries) > 1:
            # Aggregate data for multiple countries
            # Sum numeric columns by date
            numeric_cols = ['weeklyExports', 'outstandingSales', 'grossNewSales', 
                          'currentMYNetSales', 'currentMYTotalCommitment']
            available_cols = [col for col in numeric_cols if col in analysis_data.columns]
            
            if available_cols:
                grouped = analysis_data.groupby('weekEndingDate')[available_cols].sum().reset_index()
                grouped['country'] = f"Sum of {', '.join(countries)}"
                analysis_data = grouped
        
        # Create ESR analyzer instance with proper initialization
        # Determine commodity type
        commodity_type = 'livestock'  # Default for cattle, hogs, pork
        # Note: This could be enhanced to detect grain vs livestock from data
        
        analyzer = ESRAnalyzer(analysis_data, commodity_type=commodity_type)
        
        # Perform analysis
        analysis_results = analyzer.commitment_vs_shipment_analysis()
        
        # Handle the case where analysis_results is a dict with 'data' key
        if isinstance(analysis_results, dict):
            if 'error' in analysis_results:
                return create_empty_figure(f"Analytics Error: {analysis_results['error']}")
            
            # Get the actual data DataFrame
            analysis_data = analysis_results.get('data', pd.DataFrame())
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        else:
            # If it's already a DataFrame
            analysis_data = analysis_results
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        
        # Determine which analytics metric to display based on chart_id
        if 'chart_1' in chart_id:
            # Frame 0, Chart 1 - Sales Backlog
            metric = 'sales_backlog'
            title = 'Sales Backlog Analysis'
        elif 'frame1_chart_0' in chart_id:
            # Frame 1, Chart 0 - Commitment Utilization
            metric = 'commitment_utilization' 
            title = 'Commitment Utilization Rate'
        elif 'frame1_chart_1' in chart_id:
            # Frame 1, Chart 1 - Fulfillment Rate
            metric = 'fulfillment_rate'
            title = 'Export Fulfillment Rate'
        else:
            # Default to sales backlog
            metric = 'sales_backlog'
            title = 'Sales Backlog Analysis'
        
        # Check if metric column exists in analysis results
        if metric not in analysis_data.columns:
            return create_empty_figure(f"Analytics metric '{metric}' not available")
        
        # Create chart
        if country_display_mode == 'sum' and len(countries) > 1:
            fig = px.line(
                analysis_data,
                x='weekEndingDate',
                y=metric,
                title=f"{title} - Sum of {', '.join(countries)}",
                markers=True
            )
            fig.update_traces(line=dict(color='#ff7f0e', width=3))
        else:
            fig = px.line(
                analysis_data,
                x='weekEndingDate',
                y=metric,
                color='country' if 'country' in analysis_data.columns else None,
                title=f"{title} by Country",
                markers=True
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=350,
            xaxis_title='Week Ending Date',
            yaxis_title=title,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_analytics_commitment_chart: {e}")
        return create_empty_figure(f"Analytics Error: {str(e)}")


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


def commitment_metric_chart(chart_id, store_data=None, **menu_values):
    """
    Create chart for Frame 0 Chart 0 using store data with commitment metric selection.
    Based on manifest objectives for commitment analysis.
    """
    try:
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        commitment_metric = menu_values.get('commitment_metric', 'currentMYTotalCommitment')
        date_range = menu_values.get('date_range', [])
        
        # Use store data
        if not store_data:
            return [create_empty_figure("No data available in store")]
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            return [create_empty_figure(f"Error loading data: {str(e)}")]
            
        if data.empty:
            return [create_empty_figure("No data available")]
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            return [create_empty_figure("No data for selected criteria")]
        
        # Check if column exists
        if commitment_metric not in data.columns:
            return [create_empty_figure(f"Column '{commitment_metric}' not found in data")]
        
        # Create chart title
        column_labels = {
            'currentMYTotalCommitment': 'Current MY Total Commitment',
            'currentMYNetSales': 'Current MY Net Sales',
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales',
            'grossNewSales': 'Gross New Sales',
            'nextMYOutstandingSales': 'Next MY Outstanding Sales'
        }
        chart_title = column_labels.get(commitment_metric, commitment_metric.replace('_', ' ').title())
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Sum all countries together
            data_grouped = data.groupby('weekEndingDate')[commitment_metric].sum().reset_index()
            data_grouped['country'] = f"Sum of {', '.join(countries)}"
            chart_data = data_grouped
            
            fig = px.line(
                chart_data,
                x='weekEndingDate',
                y=commitment_metric,
                title=f"{chart_title} - {chart_data['country'].iloc[0]}",
                markers=True
            )
            fig.update_traces(line=dict(color='#1f77b4', width=3))
        else:
            # Individual countries
            fig = px.line(
                data,
                x='weekEndingDate',
                y=commitment_metric,
                color='country',
                title=f"{chart_title} by Country",
                markers=True
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title='Week Ending Date',
            yaxis_title=chart_title,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return [fig]
        
    except Exception as e:
        print(f"Error in commitment_metric_chart: {e}")
        return [create_empty_figure(f"Error: {str(e)}")]


def commitment_analytics_chart(chart_ids, store_data=None, **menu_values):
    """
    Create analytics charts using ESRAnalyzer for sales_backlog, fulfillment_rate, commitment_utilization.
    Based on manifest objectives for commitment analysis.
    Handles multiple chart IDs and returns multiple figures.
    """
    try:
        # Get menu values
        countries = menu_values.get('countries', ['Korea, South', 'Japan'])
        country_display_mode = menu_values.get('country_display_mode', 'individual')
        date_range = menu_values.get('date_range', [])
        
        # Handle both single chart_id and list of chart_ids for error cases
        def handle_error_return(error_msg, chart_ids):
            if isinstance(chart_ids, list) and len(chart_ids) > 1:
                return [create_empty_figure(error_msg) for _ in chart_ids]
            else:
                return create_empty_figure(error_msg)
        
        # Use store data
        if not store_data:
            return handle_error_return("No data available in store", chart_ids)
        
        try:
            if isinstance(store_data, str):
                import json
                data = pd.DataFrame(json.loads(store_data))
            else:
                data = pd.DataFrame(store_data)
            
            data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
        except Exception as e:
            print(f"Error loading store data: {e}")
            return handle_error_return(f"Error loading data: {str(e)}", chart_ids)
            
        if data.empty:
            return handle_error_return("No data available", chart_ids)
        
        # Filter by countries
        if countries and 'country' in data.columns:
            data = data[data['country'].isin(countries)]
        
        # Apply date range filter if provided
        if date_range and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            data = data[
                (data['weekEndingDate'] >= start_date) & 
                (data['weekEndingDate'] <= end_date)
            ]
        
        if data.empty:
            return handle_error_return("No data for selected criteria", chart_ids)
        
        # Perform commitment vs shipment analysis
        from models.commodity_analytics import ESRAnalyzer
        
        # Handle country display mode for analysis
        analysis_data = data.copy()
        if country_display_mode == 'sum' and len(countries) > 1:
            # Aggregate data for multiple countries
            numeric_cols = ['weeklyExports', 'outstandingSales', 'grossNewSales', 
                          'currentMYNetSales', 'currentMYTotalCommitment']
            available_cols = [col for col in numeric_cols if col in analysis_data.columns]
            
            if available_cols:
                grouped = analysis_data.groupby('weekEndingDate')[available_cols].sum().reset_index()
                grouped['country'] = f"Sum of {', '.join(countries)}"
                analysis_data = grouped
        
        # Create ESR analyzer instance
        commodity_type = 'livestock'  # Default for cattle, hogs, pork
        analyzer = ESRAnalyzer(analysis_data, commodity_type=commodity_type)
        
        # Perform analysis
        analysis_results = analyzer.commitment_vs_shipment_analysis()
        
        # Handle the case where analysis_results is a dict with 'data' key
        if isinstance(analysis_results, dict):
            if 'error' in analysis_results:
                return create_empty_figure(f"Analytics Error: {analysis_results['error']}")
            
            # Get the actual data DataFrame
            analysis_data = analysis_results.get('data', pd.DataFrame())
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        else:
            # If it's already a DataFrame
            analysis_data = analysis_results
            if analysis_data.empty:
                return create_empty_figure("No analytics data available")
        
        # Handle both single chart_id and list of chart_ids
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        
        # Create figures for each chart
        figures = []
        
        for chart_id in chart_ids:
            # Determine which analytics metric to display based on chart_id
            if 'chart_1' in chart_id or 'frame0_chart_1' in chart_id:
                # Frame 0, Chart 1 - Sales Backlog
                metric = 'sales_backlog'
                title = 'Sales Backlog Analysis'
            elif 'frame1_chart_0' in chart_id:
                # Frame 1, Chart 0 - Commitment Utilization
                metric = 'commitment_utilization' 
                title = 'Commitment Utilization Rate'
            elif 'frame1_chart_1' in chart_id:
                # Frame 1, Chart 1 - Fulfillment Rate
                metric = 'fulfillment_rate'
                title = 'Export Fulfillment Rate'
            else:
                # Default to sales backlog
                metric = 'sales_backlog'
                title = 'Sales Backlog Analysis'
            
            # Check if metric column exists in analysis results
            if metric not in analysis_data.columns:
                figures.append(create_empty_figure(f"Analytics metric '{metric}' not available"))
                continue
            
            # Prepare data for charting - ensure weekEndingDate is available
            chart_data = analysis_data.copy()
            if 'weekEndingDate' not in chart_data.columns and chart_data.index.name in ['weekEndingDate', None]:
                # If weekEndingDate is the index, reset it to a column
                chart_data = chart_data.reset_index()
                if 'index' in chart_data.columns and 'weekEndingDate' not in chart_data.columns:
                    chart_data = chart_data.rename(columns={'index': 'weekEndingDate'})
            
            # Ensure weekEndingDate is datetime
            if 'weekEndingDate' in chart_data.columns:
                chart_data['weekEndingDate'] = pd.to_datetime(chart_data['weekEndingDate'])
            else:
                # Create a simple date range if weekEndingDate is missing
                chart_data['weekEndingDate'] = pd.date_range('2024-01-01', periods=len(chart_data), freq='W')
            
            # Create chart
            if country_display_mode == 'sum' and len(countries) > 1:
                fig = px.line(
                    chart_data,
                    x='weekEndingDate',
                    y=metric,
                    title=f"{title} - Sum of {', '.join(countries)}",
                    markers=True
                )
                fig.update_traces(line=dict(color='#ff7f0e', width=3))
            else:
                fig = px.line(
                    chart_data,
                    x='weekEndingDate',
                    y=metric,
                    color='country' if 'country' in chart_data.columns else None,
                    title=f"{title} by Country",
                    markers=True
                )
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title='Week Ending Date',
                yaxis_title=title,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            figures.append(fig)
        
        # Return single figure or list of figures based on input
        if len(figures) == 1:
            return figures[0]
        else:
            return figures
        
    except Exception as e:
        print(f"Error in commitment_analytics_chart: {e}")
        # Handle both single and multiple chart error cases
        if isinstance(chart_ids, list) and len(chart_ids) > 1:
            return [create_empty_figure(f"Analytics Error: {str(e)}") for _ in chart_ids]
        else:
            return create_empty_figure(f"Analytics Error: {str(e)}")


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


def create_country_analysis_chart(data, country_metric, countries, country_display_mode, 
                                  start_year, end_year, chart_id, chart_index=0):
    """Create country analysis chart with market year overlays and multi-country support"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Check if column exists
        if country_metric not in data.columns:
            return create_empty_figure(f"Column '{country_metric}' not found in data")
        
        # Create title mapping
        metric_labels = {
            'weeklyExports': 'Weekly Exports',
            'outstandingSales': 'Outstanding Sales', 
            'grossNewSales': 'Gross New Sales',
            'currentMYNetSales': 'Current MY Net Sales',
            'currentMYTotalCommitment': 'Current MY Total Commitment'
        }
        metric_name = metric_labels.get(country_metric, country_metric.title())
        
        # Create marketing year column if it doesn't exist
        if 'marketing_year' not in data.columns:
            data = data.copy()
            # USDA marketing year (Oct-Sep for most commodities)
            data['marketing_year'] = data['weekEndingDate'].apply(
                lambda x: x.year if x.month < 10 else x.year + 1
            )
        
        # Apply year filtering for market year overlays
        if start_year and end_year:
            data = data[
                (data['marketing_year'] >= start_year) & 
                (data['marketing_year'] <= end_year)
            ]
        
        if data.empty:
            return create_empty_figure(f"No data available for {metric_name}")
        
        # Handle country display mode
        if country_display_mode == 'sum' and len(countries) > 1:
            # Sum all countries together
            aggregated_data = data.groupby(['weekEndingDate', 'marketing_year'])[country_metric].sum().reset_index()
            aggregated_data['country'] = f"Sum of {', '.join(countries[:3])}" + ("..." if len(countries) > 3 else "")
            
            # Create chart for summed data
            fig = px.line(
                aggregated_data,
                x='weekEndingDate',
                y=country_metric, 
                color='marketing_year',
                title=f'{metric_name} - Market Year Overlays (Summed Countries)',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        else:
            # Individual countries
            if chart_index == 0:
                # First chart - show all years overlaid
                fig = px.line(
                    data,
                    x='weekEndingDate',
                    y=country_metric,
                    color='marketing_year',
                    line_dash='country',  # Distinguish countries by line style
                    title=f'{metric_name} - Market Year Overlays by Country',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
            else:
                # Second chart - focus on country comparison for current year
                current_my = data['marketing_year'].max()
                current_data = data[data['marketing_year'] == current_my]
                
                fig = px.line(
                    current_data,
                    x='weekEndingDate',
                    y=country_metric,
                    color='country',
                    title=f'{metric_name} - Current Marketing Year ({current_my}) Country Comparison',
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Dark2
                )
        
        # Update layout
        fig.update_layout(
            height=450,
            hovermode='x unified',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Week Ending Date",
            yaxis_title=metric_name,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Date: %{x}<br>' + 
                         f'{metric_name}: %{{y:,.0f}}<br>' +
                         '<extra></extra>'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating country analysis chart: {e}")
        return create_empty_figure(f"Error creating chart: {str(e)}")