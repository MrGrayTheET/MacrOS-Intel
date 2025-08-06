"""
Integration Example: Using Time Series Models with ESR Dashboard
===============================================================

This example demonstrates how to integrate the time series analysis models
with your existing ESR dashboard and callback functions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Import your models
from models.commodity_analytics import create_esr_analyzer, compare_country_exports
from models.timeseries_analysis import TimeSeriesAnalyzer

# Import existing ESR utilities
from pages.esr.esr_utils import get_sample_esr_data

def enhanced_esr_analysis_example():
    """
    Example of enhanced ESR analysis using the new time series models.
    This shows how to integrate with your existing ESR callback functions.
    """
    # Load sample ESR data (using your existing utility)
    commodity = 'cattle'
    year = 2024
    countries = ['Korea, South', 'Japan', 'China', 'Mexico', 'Canada']
    
    esr_data = get_sample_esr_data(commodity, year, countries)
    
    print(f"Loaded ESR data: {len(esr_data)} records")
    print(f"Countries: {esr_data['country'].unique()}")
    print(f"Date range: {esr_data['weekEndingDate'].min()} to {esr_data['weekEndingDate'].max()}")
    
    # Create ESR analyzer
    analyzer = create_esr_analyzer(esr_data, commodity_type='livestock')
    
    # 1. Seasonal Pattern Analysis
    print("\n=== SEASONAL ANALYSIS ===")
    seasonal_patterns = analyzer.analyze_seasonal_patterns('weeklyExports', group_by='country')
    
    for country, patterns in seasonal_patterns.items():
        if 'error' not in patterns:
            print(f"\n{country}:")
            print(f"  Peak weeks: {patterns['peak_weeks']}")
            print(f"  Seasonality strength: {patterns['seasonality_strength']:.2f}")
            print(f"  Average growth rate: {patterns['average_growth_rate']:.1%}")
    
    # 2. Stationarity Testing
    print("\n=== STATIONARITY TESTING ===")
    stationarity_results = analyzer.test_stationarity('weeklyExports', group_by='country')
    
    for country, result in stationarity_results.items():
        if 'adf' in result:
            is_stationary = result['adf']['is_stationary']
            p_value = result['adf']['p_value']
            print(f"{country}: {'Stationary' if is_stationary else 'Non-stationary'} (p={p_value:.3f})")
    
    # 3. Seasonal Normalization
    print("\n=== SEASONAL NORMALIZATION ===")
    normalized_exports = analyzer.seasonal_normalize('weeklyExports', method='stl', group_by='country')
    print(f"Normalized data shape: {normalized_exports.shape}")
    print(f"Original std: {esr_data['weeklyExports'].std():.2f}")
    print(f"Normalized std: {normalized_exports.std():.2f}")
    
    # 4. Commitment vs Shipment Analysis  
    print("\n=== COMMITMENT ANALYSIS ===")
    for country in countries[:3]:  # Analyze top 3 countries
        commitment_analysis = analyzer.commitment_vs_shipment_analysis(country=country)
        if 'error' not in commitment_analysis and 'fulfillment_rate' in commitment_analysis:
            fulfillment = commitment_analysis['fulfillment_rate']
            print(f"{country}:")
            print(f"  Avg fulfillment rate: {fulfillment['mean']:.1%}")
            print(f"  Trend: {'Improving' if fulfillment['trend'] > 0 else 'Declining'}")
    
    # 5. Country Performance Ranking
    print("\n=== COUNTRY RANKINGS ===")
    rankings = analyzer.country_performance_ranking('weeklyExports', 'recent_year')
    print("Top 3 countries by total exports:")
    for i, (country, row) in enumerate(rankings.head(3).iterrows(), 1):
        print(f"{i}. {country}: {row['total']:,.0f} (Market Share: {row['market_share_pct']:.1f}%)")
    
    # 6. Outlier Detection
    print("\n=== ANOMALY DETECTION ===")
    anomalies = analyzer.detect_export_anomalies('weeklyExports', method='seasonal_adjusted', group_by='country')
    
    total_anomalies = anomalies['is_anomaly'].sum() if not anomalies.empty else 0
    print(f"Detected {total_anomalies} export anomalies across all countries")
    
    if not anomalies.empty:
        top_anomalies = anomalies[anomalies['is_anomaly']].nlargest(3, 'anomaly_score')
        print("Top 3 anomalies:")
        for _, anomaly in top_anomalies.iterrows():
            print(f"  {anomaly['date'].strftime('%Y-%m-%d')}: {anomaly['country']} "
                  f"(Score: {anomaly['anomaly_score']:.2f})")
    
    return analyzer, esr_data


def enhanced_chart_update_example(chart_id: str, **menu_values):
    """
    Example of how to enhance your existing chart update functions
    with time series analysis capabilities.
    
    This could replace or enhance functions like:
    - sales_trends_chart_update
    - country_analysis_chart_update
    - commitment_analysis_chart_update
    """
    
    # Extract parameters (same as your existing functions)
    commodity = menu_values.get('commodity', 'cattle')
    countries = menu_values.get('countries', ['Korea, South', 'Japan', 'China'])
    analysis_type = menu_values.get('analysis_type', 'original')  # New parameter
    
    # Get ESR data
    current_year = pd.Timestamp.now().year
    data = get_sample_esr_data(commodity, current_year, countries)
    
    if data.empty:
        return create_empty_figure(f"{commodity.title()} - No Data")
    
    # Create analyzer
    analyzer = create_esr_analyzer(data, commodity_type='livestock')
    
    # Chart-specific logic
    if 'sales_trends' in chart_id:
        return create_enhanced_sales_trends_chart(analyzer, analysis_type, countries)
    elif 'country_analysis' in chart_id:
        return create_enhanced_country_chart(analyzer, analysis_type)
    elif 'commitment_analysis' in chart_id:
        return create_enhanced_commitment_chart(analyzer, analysis_type)
    else:
        return create_enhanced_default_chart(analyzer, analysis_type)


def create_enhanced_sales_trends_chart(analyzer, analysis_type, countries):
    """Enhanced sales trends chart with time series analysis."""
    
    fig = go.Figure()
    
    for country in countries:
        country_data = analyzer.data[analyzer.data['country'] == country]
        
        if analysis_type == 'seasonal_normalized':
            # Show seasonally adjusted data
            normalized = analyzer.seasonal_normalize('weeklyExports', group_by='country')
            country_normalized = normalized[analyzer.data['country'] == country]
            
            fig.add_trace(go.Scatter(
                x=country_data.index,
                y=country_normalized,
                name=f'{country} (Seasonal Adj.)',
                line=dict(dash='solid')
            ))
        
        elif analysis_type == 'trend_decomposed':
            # Show trend component
            try:
                decomp = analyzer.decompose_series('weeklyExports', group_by='country')
                country_trend = decomp[decomp['country'] == country]['weeklyExports_trend']
                
                fig.add_trace(go.Scatter(
                    x=country_data.index,
                    y=country_trend,
                    name=f'{country} (Trend)',
                    line=dict(dash='solid')
                ))
            except:
                # Fallback to original data
                fig.add_trace(go.Scatter(
                    x=country_data.index,
                    y=country_data['weeklyExports'],
                    name=f'{country} (Original)',
                    line=dict(dash='solid')
                ))
        
        else:  # original
            fig.add_trace(go.Scatter(
                x=country_data.index,
                y=country_data['weeklyExports'],
                name=f'{country}',
                line=dict(dash='solid')
            ))
    
    # Add anomaly markers if requested
    if analysis_type == 'with_anomalies':
        anomalies = analyzer.detect_export_anomalies('weeklyExports', method='iqr', group_by='country')
        if not anomalies.empty:
            anomaly_data = anomalies[anomalies['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=anomaly_data['date'],
                y=anomaly_data['original_value'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
    
    fig.update_layout(
        title=f'Enhanced Sales Trends Analysis ({analysis_type})',
        xaxis_title='Date',
        yaxis_title='Weekly Exports',
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_enhanced_country_chart(analyzer, analysis_type):
    """Enhanced country analysis with time series insights."""
    
    # Country performance ranking
    rankings = analyzer.country_performance_ranking('weeklyExports')
    top_countries = rankings.head(5)
    
    if analysis_type == 'performance_ranking':
        fig = px.bar(
            x=top_countries.index,
            y=top_countries['total'],
            title='Country Performance Ranking',
            labels={'x': 'Country', 'y': 'Total Weekly Exports'}
        )
        
        # Add market share annotations
        for i, (country, row) in enumerate(top_countries.iterrows()):
            fig.add_annotation(
                x=country,
                y=row['total'],
                text=f"{row['market_share_pct']:.1f}%",
                showarrow=False,
                yshift=10
            )
    
    elif analysis_type == 'volatility_analysis':
        fig = px.scatter(
            x=top_countries['average'],
            y=top_countries['volatility'],
            text=top_countries.index,
            title='Export Volume vs Volatility',
            labels={'x': 'Average Weekly Exports', 'y': 'Volatility (CV)'}
        )
        fig.update_traces(textposition="top center")
    
    else:  # seasonal_patterns
        fig = go.Figure()
        
        seasonal_data = analyzer.analyze_seasonal_patterns('weeklyExports', group_by='country')
        
        for country in top_countries.index[:3]:  # Top 3 countries
            if country in seasonal_data and 'error' not in seasonal_data[country]:
                weekly_means = pd.Series(seasonal_data[country]['weekly_patterns']['mean'])
                
                fig.add_trace(go.Scatter(
                    x=weekly_means.index,
                    y=weekly_means.values,
                    name=country,
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title='Seasonal Export Patterns by Country',
            xaxis_title='Marketing Year Week',
            yaxis_title='Average Weekly Exports'
        )
    
    fig.update_layout(height=400)
    return fig


def create_enhanced_commitment_chart(analyzer, analysis_type):
    """Enhanced commitment analysis with advanced metrics."""
    
    fig = go.Figure()
    
    # Multi-series commitment analysis
    commitment_results = analyzer.multi_series_analysis(
        ['weeklyExports', 'outstandingSales', 'currentMYTotalCommitment'], 
        'country'
    )
    
    if analysis_type == 'fulfillment_efficiency':
        # Show export fulfillment efficiency by country
        for country, stats in commitment_results['group_statistics'].items():
            if 'weeklyExports' in stats and 'outstandingSales' in stats:
                fulfillment_rate = (stats['weeklyExports']['mean'] / 
                                  stats['outstandingSales']['mean'] * 100)
                
                fig.add_trace(go.Bar(
                    x=[country],
                    y=[fulfillment_rate],
                    name=country
                ))
        
        fig.update_layout(
            title='Export Fulfillment Efficiency by Country',
            xaxis_title='Country',
            yaxis_title='Fulfillment Rate (%)'
        )
    
    elif analysis_type == 'commitment_correlation':
        # Show correlation between commitment metrics
        countries = list(commitment_results['correlations'].keys())
        if countries:
            country = countries[0]  # Show first country's correlation
            corr_data = commitment_results['correlations'][country]
            
            # Create correlation heatmap data
            metrics = list(corr_data.keys())
            z_values = [[corr_data[m1].get(m2, 0) for m2 in metrics] for m1 in metrics]
            
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=metrics,
                y=metrics,
                colorscale='RdYlBu',
                zmid=0
            ))
            
            fig.update_layout(
                title=f'Commitment Metrics Correlation - {country}',
                width=500,
                height=400
            )
    
    else:  # time_series_comparison
        # Show time series of multiple commitment metrics
        data = analyzer.data
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['weeklyExports'],
            name='Weekly Exports',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['outstandingSales'],
            name='Outstanding Sales',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Export vs Outstanding Sales Comparison',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Weekly Exports', side='left'),
            yaxis2=dict(title='Outstanding Sales', side='right', overlaying='y')
        )
    
    fig.update_layout(height=400)
    return fig


def create_enhanced_default_chart(analyzer, analysis_type):
    """Default enhanced chart for any unspecified chart types."""
    
    # Multi-series summary
    summary_stats = analyzer.summary_statistics(['weeklyExports', 'outstandingSales'])
    
    fig = px.bar(
        summary_stats,
        x='column',
        y='mean',
        error_y='std',
        title='Summary Statistics - Key ESR Metrics'
    )
    
    fig.update_layout(
        xaxis_title='Metric',
        yaxis_title='Average Value',
        height=400
    )
    
    return fig


def create_empty_figure(title: str):
    """Create empty figure for error states."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(title=title, height=400)
    return fig


if __name__ == "__main__":
    print("Running enhanced ESR analysis example...")
    analyzer, data = enhanced_esr_analysis_example()
    
    print(f"\nAnalysis complete! Analyzer created with {len(data)} records.")
    print(f"Available methods: {[m for m in dir(analyzer) if not m.startswith('_')]}")
    
    print("\nTo integrate with your existing callbacks:")
    print("1. Import: from models.integration_example import enhanced_chart_update_example")
    print("2. Replace your chart update functions with enhanced versions")
    print("3. Add 'analysis_type' parameter to your menus for different visualizations")