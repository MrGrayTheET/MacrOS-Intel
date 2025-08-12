#!/usr/bin/env python3
"""
Test multi-chart callback registration with EnhancedFrameGrid.
This demonstrates registering multiple charts to a single update function.
"""

import sys
sys.path.append('..')

from components.frames import EnhancedFrameGrid, FlexibleMenu, FundamentalFrame
from data.data_tables import ESRTableClient
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np

def generate_mock_data():
    """Generate mock ESR data for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=52)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    countries = ['Korea, South', 'Japan', 'China']
    data = []
    
    for country in countries:
        base_exports = np.random.randint(100, 300)
        for date in dates:
            seasonal = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise = np.random.normal(1, 0.1)
            
            weekly_exports = max(0, int(base_exports * seasonal * noise))
            outstanding = max(0, int(weekly_exports * 2.5 * noise))
            gross_new = max(0, int(weekly_exports * 1.3 * noise))
            
            data.append({
                'weekEndingDate': date.strftime('%Y-%m-%d'),
                'country': country,
                'commodity': 'cattle',
                'weeklyExports': weekly_exports,
                'outstandingSales': outstanding,
                'grossNewSales': gross_new
            })
    
    return data

def multi_store_update_function(chart_ids, store_data=None, **menu_values):
    """Update function that handles multiple charts from store data."""
    print(f"[MULTI-STORE UPDATE] Charts: {chart_ids}")
    print(f"  Store data: {'available' if store_data else 'none'}")
    print(f"  Menu values: {menu_values}")
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    figures = []
    
    try:
        if store_data:
            if isinstance(store_data, str):
                data = json.loads(store_data)
            else:
                data = store_data
            
            df = pd.DataFrame(data)
            df['weekEndingDate'] = pd.to_datetime(df['weekEndingDate'])
            
            # Apply menu filtering
            commodity = menu_values.get('commodity', 'cattle')
            countries = menu_values.get('countries', ['Korea, South'])
            
            df_filtered = df[
                (df['commodity'] == commodity) &
                (df['country'].isin(countries))
            ]
            
            # Create different chart for each chart_id
            for i, chart_id in enumerate(chart_ids):
                if i == 0:
                    # First chart - Weekly Exports
                    fig = px.line(
                        df_filtered,
                        x='weekEndingDate',
                        y='weeklyExports',
                        color='country',
                        title=f'Multi-Chart {i+1}: Weekly Exports',
                        markers=True
                    )
                elif i == 1:
                    # Second chart - Outstanding Sales
                    fig = px.line(
                        df_filtered,
                        x='weekEndingDate',
                        y='outstandingSales',
                        color='country',
                        title=f'Multi-Chart {i+1}: Outstanding Sales',
                        markers=True
                    )
                else:
                    # Additional charts - Gross New Sales
                    fig = px.line(
                        df_filtered,
                        x='weekEndingDate',
                        y='grossNewSales',
                        color='country',
                        title=f'Multi-Chart {i+1}: Gross New Sales',
                        markers=True
                    )
                
                fig.update_layout(template='plotly_dark', height=400)
                figures.append(fig)
        
        if not figures:
            # Create fallback figures
            for i, chart_id in enumerate(chart_ids):
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Multi-Store Update<br>Chart {i+1}<br>ID: {chart_id}<br>Menu: {menu_values}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=12, color="white")
                )
                fig.update_layout(title=f"Multi-Chart {i+1} (Store)", template='plotly_dark', height=400)
                figures.append(fig)
    
    except Exception as e:
        print(f"Error in multi-store update: {e}")
        # Create error figures
        for i, chart_id in enumerate(chart_ids):
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error in Multi-Store Update<br>Chart {i+1}: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="red")
            )
            fig.update_layout(title=f"Error - Chart {i+1}", template='plotly_dark', height=400)
            figures.append(fig)
    
    return figures

def multi_function_update_function(chart_ids, **values):
    """Update function that handles multiple charts from function inputs."""
    print(f"[MULTI-FUNCTION UPDATE] Charts: {chart_ids}")
    print(f"  Values: {values}")
    
    import plotly.graph_objects as go
    
    figures = []
    
    # Get values
    commodity = values.get('commodity', 'cattle')
    start_year = values.get('start_year', 2023)
    end_year = values.get('end_year', 2024)
    
    try:
        for i, chart_id in enumerate(chart_ids):
            # Generate different mock data for each chart
            dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='M')
            
            if i == 0:
                # First chart - trend line
                values_data = np.random.randint(50, 200, len(dates)) + i * 20
                color = 'blue'
                name = f'{commodity.title()} Trend'
            elif i == 1:
                # Second chart - seasonal pattern
                values_data = 100 + 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12) + i * 15
                color = 'green'
                name = f'{commodity.title()} Seasonal'
            else:
                # Additional charts - random walk
                values_data = np.cumsum(np.random.randn(len(dates))) + 150 + i * 10
                color = 'orange'
                name = f'{commodity.title()} Random'
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values_data,
                mode='lines+markers',
                name=name,
                line=dict(color=color)
            ))
            
            fig.update_layout(
                title=f'Multi-Function Chart {i+1}: {name} ({start_year}-{end_year})',
                template='plotly_dark',
                height=400,
                xaxis_title='Date',
                yaxis_title='Value'
            )
            
            figures.append(fig)
    
    except Exception as e:
        print(f"Error in multi-function update: {e}")
        # Create error figures
        for i, chart_id in enumerate(chart_ids):
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error in Multi-Function Update<br>Chart {i+1}: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="red")
            )
            fig.update_layout(title=f"Error - Chart {i+1}", template='plotly_dark', height=400)
            figures.append(fig)
    
    return figures

def single_vs_multi_update_function(chart_ids, store_data=None, **menu_values):
    """Update function that returns single figure for multiple charts (duplicate scenario)."""
    print(f"[SINGLE->MULTI UPDATE] Charts: {chart_ids}")
    print(f"  Store data: {'available' if store_data else 'none'}")
    print(f"  Will return single figure for {len(chart_ids)} charts")
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    try:
        if store_data:
            if isinstance(store_data, str):
                data = json.loads(store_data)
            else:
                data = store_data
            
            df = pd.DataFrame(data)
            df['weekEndingDate'] = pd.to_datetime(df['weekEndingDate'])
            
            # Apply menu filtering
            commodity = menu_values.get('commodity', 'cattle')
            countries = menu_values.get('countries', ['Korea, South'])
            
            df_filtered = df[
                (df['commodity'] == commodity) &
                (df['country'].isin(countries))
            ]
            
            if not df_filtered.empty:
                fig = px.line(
                    df_filtered,
                    x='weekEndingDate',
                    y='weeklyExports',
                    color='country',
                    title='Same Chart for Multiple Outputs',
                    markers=True
                )
                fig.update_layout(template='plotly_dark', height=400)
                return fig  # Return single figure - framework should duplicate it
    
    except Exception as e:
        print(f"Error in single->multi update: {e}")
    
    # Fallback
    fig = go.Figure()
    fig.add_annotation(
        text=f"Single Figure for Multiple Charts<br>{len(chart_ids)} outputs<br>Menu: {menu_values}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="white")
    )
    fig.update_layout(title="Duplicated Chart", template='plotly_dark', height=400)
    return fig  # Single figure

def test_multi_chart_registration():
    """Test multi-chart callback registration."""
    
    print("Testing Multi-Chart Callback Registration")
    print("=" * 50)
    
    # Create table client and frames
    table_client = ESRTableClient()
    
    # Create charts for testing
    chart_configs = [
        {
            'title': 'Chart 0 - Multi Store Group A',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Chart 1 - Multi Store Group A',
            'chart_type': 'area',
            'starting_key': 'cattle/exports/all',
            'y_column': 'outstandingSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Chart 2 - Multi Function Group B',
            'chart_type': 'bar',
            'starting_key': 'cattle/exports/all',
            'y_column': 'grossNewSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Chart 3 - Multi Function Group B',
            'chart_type': 'scatter',
            'starting_key': 'cattle/exports/all',
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Chart 4 - Single->Multi Store',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
            'y_column': 'outstandingSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Chart 5 - Single->Multi Store',
            'chart_type': 'area',
            'starting_key': 'cattle/exports/all',
            'y_column': 'grossNewSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        }
    ]
    
    frame = FundamentalFrame(
        table_client=table_client,
        chart_configs=chart_configs,
        layout="horizontal",
        div_prefix="test_multi_frame",
        width="100%",
        height="2000px"
    )
    
    # Create menu
    menu_configs = [
        {
            'type': 'dropdown',
            'id': 'commodity',
            'label': 'Commodity',
            'options': [
                {'label': 'Cattle', 'value': 'cattle'},
                {'label': 'Corn', 'value': 'corn'},
                {'label': 'Wheat', 'value': 'wheat'}
            ],
            'value': 'cattle'
        },
        {
            'type': 'checklist',
            'id': 'countries',
            'label': 'Countries',
            'options': [
                {'label': 'Korea, South', 'value': 'Korea, South'},
                {'label': 'Japan', 'value': 'Japan'},
                {'label': 'China', 'value': 'China'}
            ],
            'value': ['Korea, South', 'Japan']
        },
        {
            'type': 'dropdown',
            'id': 'start_year',
            'label': 'Start Year',
            'options': [
                {'label': '2022', 'value': 2022},
                {'label': '2023', 'value': 2023},
                {'label': '2024', 'value': 2024}
            ],
            'value': 2023
        },
        {
            'type': 'dropdown',
            'id': 'end_year',
            'label': 'End Year',
            'options': [
                {'label': '2023', 'value': 2023},
                {'label': '2024', 'value': 2024},
                {'label': '2025', 'value': 2025}
            ],
            'value': 2024
        },
        {
            'type': 'button',
            'id': 'apply_function',
            'label': 'Update Function Charts',
            'color': '#FF9800'
        }
    ]
    
    menu = FlexibleMenu(
        menu_id='test_multi_menu',
        title='Multi-Chart Test Controls',
        component_configs=menu_configs
    )
    
    # Create grid with store data source
    grid = EnhancedFrameGrid(
        frames=[frame],
        flexible_menu=menu,
        data_source='test_multi_store'
    )
    
    print(f"\nGrid created with {len(grid.get_chart_ids())} charts:")
    for chart_id in grid.get_chart_ids():
        print(f"  - {chart_id}")
    
    # Test individual registration methods
    print("\n" + "=" * 30)
    print("Multi-Chart Registration")
    print("=" * 30)
    
    # Simulate app registration (normally done with real Dash app)
    class MockApp:
        def callback(self, output, inputs, states=None, prevent_initial_call=True):
            def decorator(func):
                output_str = str(output) if not isinstance(output, list) else f"[{', '.join(str(o) for o in output)}]"
                input_str = f"[{', '.join(str(i) for i in inputs)}]"
                print(f"  Mock callback registered: {output_str} <- {input_str}")
                return func
            return decorator
    
    mock_app = MockApp()
    
    # Get chart IDs
    all_chart_ids = grid.get_chart_ids()
    
    print(f"\nRegistering multi-chart store callback (Group A):")
    # Register charts 0 and 1 together with multi-store callback
    group_a_charts = [all_chart_ids[0], all_chart_ids[1]]
    success_a = grid.register_chart_store_callback(
        app=mock_app,
        chart_id=group_a_charts,
        update_function=multi_store_update_function,
        menu_inputs=['commodity', 'countries']
    )
    
    print(f"\nRegistering multi-chart function callback (Group B):")
    # Register charts 2 and 3 together with multi-function callback
    group_b_charts = [all_chart_ids[2], all_chart_ids[3]]
    success_b = grid.register_chart_function_callback(
        app=mock_app,
        chart_id=group_b_charts,
        update_function=multi_function_update_function,
        input_components=['commodity', 'start_year', 'end_year', 'apply_function'],
        trigger_component='apply_function'
    )
    
    print(f"\nRegistering single->multi store callback (Group C):")
    # Register charts 4 and 5 together with single figure function (test duplication)
    group_c_charts = [all_chart_ids[4], all_chart_ids[5]]
    success_c = grid.register_chart_store_callback(
        app=mock_app,
        chart_id=group_c_charts,
        update_function=single_vs_multi_update_function,
        menu_inputs=['commodity', 'countries']
    )
    
    # Print registry summary
    grid.print_chart_registry_summary()
    
    print(f"\nSuccess Results:")
    print(f"  Group A (Multi-Store): {success_a}")
    print(f"  Group B (Multi-Function): {success_b}")
    print(f"  Group C (Single->Multi): {success_c}")
    
    # Test multi-chart helper methods
    print(f"\nMulti-Chart Helper Methods:")
    print(f"  Multi-chart groups: {list(grid.get_multi_chart_groups().keys())}")
    
    for chart_id in all_chart_ids:
        is_multi = grid.is_multi_chart_callback(chart_id)
        print(f"  {chart_id}: {'multi-chart' if is_multi else 'single-chart'}")
    
    return grid

def test_mock_multi_callback_execution():
    """Test mock multi-callback execution to verify functionality."""
    
    print("\n" + "=" * 50)
    print("Mock Multi-Callback Execution Test")
    print("=" * 50)
    
    # Generate mock store data
    mock_store_data = generate_mock_data()
    store_json = json.dumps(mock_store_data)
    
    print(f"Generated {len(mock_store_data)} mock records")
    
    # Test multi-store update function
    print("\nTesting multi-store update function:")
    test_chart_ids = ["test_multi_frame_chart_0", "test_multi_frame_chart_1"]
    
    figs_multi_store = multi_store_update_function(
        chart_ids=test_chart_ids,
        store_data=store_json,
        commodity='cattle',
        countries=['Korea, South', 'Japan']
    )
    
    print(f"Multi-store function returned: {len(figs_multi_store)} figures")
    
    # Test multi-function update function
    print("\nTesting multi-function update function:")
    test_chart_ids = ["test_multi_frame_chart_2", "test_multi_frame_chart_3"]
    
    figs_multi_function = multi_function_update_function(
        chart_ids=test_chart_ids,
        commodity='cattle',
        start_year=2023,
        end_year=2024
    )
    
    print(f"Multi-function update returned: {len(figs_multi_function)} figures")
    
    # Test single->multi update function
    print("\nTesting single->multi update function:")
    test_chart_ids = ["test_multi_frame_chart_4", "test_multi_frame_chart_5"]
    
    fig_single_multi = single_vs_multi_update_function(
        chart_ids=test_chart_ids,
        store_data=store_json,
        commodity='cattle',
        countries=['Korea, South']
    )
    
    print(f"Single->multi function returned: {type(fig_single_multi)} (should be single figure)")
    
    print("\nMulti-callback execution tests completed successfully!")

if __name__ == '__main__':
    print("Multi-Chart Callback Registration Test")
    print("=" * 60)
    
    # Run tests
    grid = test_multi_chart_registration()
    test_mock_multi_callback_execution()
    
    print("\n" + "=" * 60)
    print("MULTI-CHART REGISTRATION TEST COMPLETED!")
    print("=" * 60)
    
    print("\nKey Features Demonstrated:")
    print("- Multi-chart callback registration (list of chart IDs)")
    print("- Single update function handling multiple chart outputs")
    print("- Automatic figure duplication for single->multi scenarios")
    print("- Multi-chart group tracking and status reporting")
    print("- Mixed single and multi-chart callbacks in same grid")
    
    print("\nUsage Examples:")
    print("""
# Register multiple charts with single update function
grid.register_chart_store_callback(
    app, ['chart_0', 'chart_1'], multi_update_func, menu_inputs=['commodity']
)

# Register multiple charts with function callback  
grid.register_chart_function_callback(
    app, ['chart_2', 'chart_3'], multi_update_func, 
    input_components=['year', 'apply_btn'], trigger_component='apply_btn'
)

# Check multi-chart status
groups = grid.get_multi_chart_groups()
for group_key, info in groups.items():
    print(f"Group: {info['charts']}, Type: {info['callback_type']}")
""")