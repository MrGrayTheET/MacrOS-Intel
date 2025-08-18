#!/usr/bin/env python3
"""
Test individual chart callback registration with EnhancedFrameGrid.
This demonstrates registering some charts with store callbacks and others with function callbacks.
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

def store_update_function(chart_id, store_data=None, **menu_values):
    """Update function that uses store data."""
    print(f"[STORE UPDATE] {chart_id}")
    print(f"  Store data: {'available' if store_data else 'none'}")
    print(f"  Menu values: {menu_values}")
    
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
            
            # Determine metric based on chart
            if 'chart_0' in chart_id:
                metric = 'weeklyExports'
                title = 'Weekly Exports (Store Data)'
            elif 'chart_1' in chart_id:
                metric = 'outstandingSales'
                title = 'Outstanding Sales (Store Data)'
            else:
                metric = 'weeklyExports'
                title = 'Store Chart'
            
            if not df_filtered.empty:
                fig = px.line(
                    df_filtered,
                    x='weekEndingDate',
                    y=metric,
                    color='country',
                    title=title,
                    markers=True
                )
                fig.update_layout(template='plotly_dark', height=400)
                return fig
    
    except Exception as e:
        print(f"Error in store update: {e}")
    
    # Fallback
    fig = go.Figure()
    fig.add_annotation(
        text=f"Store Update<br>{chart_id}<br>Menu: {menu_values}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="white")
    )
    fig.update_layout(title=f"{chart_id} - Store Mode", template='plotly_dark', height=400)
    return fig

def function_update_function(chart_id, **values):
    """Update function that uses traditional function-based approach."""
    print(f"[FUNCTION UPDATE] {chart_id}")
    print(f"  Values: {values}")
    
    import plotly.graph_objects as go
    
    # Simulate some processing based on values
    commodity = values.get('commodity', 'cattle')
    start_year = values.get('start_year', 2023)
    end_year = values.get('end_year', 2024)
    
    # Generate some mock data based on inputs
    dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='M')
    values_data = np.random.randint(50, 200, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values_data,
        mode='lines+markers',
        name=f'{commodity.title()} Data',
        line=dict(color='orange')
    ))
    
    fig.update_layout(
        title=f'Function-Based Chart: {commodity.title()} ({start_year}-{end_year})',
        template='plotly_dark',
        height=400,
        xaxis_title='Date',
        yaxis_title='Value'
    )
    
    return fig

def test_individual_chart_registration():
    """Test individual chart callback registration."""
    
    print("Testing Individual Chart Callback Registration")
    print("=" * 50)
    
    # Create table client and frames
    table_client = ESRTableClient()
    
    chart_configs = [
        {
            'title': 'Chart 0 - Store Driven',
            'chart_type': 'line',
            'starting_key': 'cattle/exports/all',
            'y_column': 'weeklyExports',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Chart 1 - Store Driven',
            'chart_type': 'area',
            'starting_key': 'cattle/exports/all',
            'y_column': 'outstandingSales',
            'x_column': 'weekEndingDate',
            'width': '100%',
            'height': 400
        },
        {
            'title': 'Chart 2 - Function Driven',
            'chart_type': 'bar',
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
        div_prefix="test_individual_frame",
        width="100%",
        height="1300px"
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
            'label': 'Update Function Chart',
            'color': '#FF9800'
        }
    ]
    
    menu = FlexibleMenu(
        menu_id='test_individual_menu',
        title='Individual Test Controls',
        component_configs=menu_configs
    )
    
    # Create grid with store data source
    grid = EnhancedFrameGrid(
        frames=[frame],
        flexible_menu=menu,
        data_source='test_individual_store'
    )
    
    print(f"\nGrid created with {len(grid.get_chart_ids())} charts:")
    for chart_id in grid.get_chart_ids():
        print(f"  - {chart_id}")
    
    # Test individual registration methods
    print("\n" + "=" * 30)
    print("Individual Chart Registration")
    print("=" * 30)
    
    # Simulate app registration (normally done with real Dash app)
    class MockApp:
        def callback(self, output, inputs, states=None, prevent_initial_call=True):
            def decorator(func):
                print(f"  Mock callback registered: {output} <- {inputs}")
                return func
            return decorator
    
    mock_app = MockApp()
    
    # Register first two charts with store callbacks
    chart_0 = grid.get_chart_ids()[0]  # test_individual_frame_chart_0
    chart_1 = grid.get_chart_ids()[1]  # test_individual_frame_chart_1
    chart_2 = grid.get_chart_ids()[2]  # test_individual_frame_chart_2
    
    print(f"\nRegistering store callbacks:")
    
    success_0 = grid.register_chart_store_callback(
        app=mock_app,
        chart_id=chart_0,
        update_function=store_update_function,
        menu_inputs=['commodity', 'countries']
    )
    
    success_1 = grid.register_chart_store_callback(
        app=mock_app,
        chart_id=chart_1,
        update_function=store_update_function,
        menu_inputs=['commodity', 'countries']
    )
    
    print(f"\nRegistering function callback:")
    
    success_2 = grid.register_chart_function_callback(
        app=mock_app,
        chart_id=chart_2,
        update_function=function_update_function,
        input_components=['commodity', 'start_year', 'end_year', 'apply_function'],
        trigger_component='apply_function'
    )
    
    # Print registry summary
    grid.print_chart_registry_summary()
    
    # Test individual chart info
    print(f"\nDetailed Chart Information:")
    print("-" * 30)
    
    for chart_id in grid.get_chart_ids():
        info = grid.get_chart_info(chart_id)
        print(f"\n{chart_id}:")
        print(f"  Callback Type: {info.get('callback_type', 'None')}")
        print(f"  Registered: {info.get('callback_registered', False)}")
        print(f"  Input IDs: {info.get('input_ids', [])}")
        print(f"  Output IDs: {info.get('output_ids', [])}")
    
    # Test helper methods
    print(f"\nHelper Method Results:")
    print(f"  All Chart IDs: {grid.get_chart_ids()}")
    print(f"  Store Chart IDs: {grid.get_store_chart_ids()}")
    print(f"  Function Chart IDs: {grid.get_function_chart_ids()}")
    print(f"  Unregistered Chart IDs: {grid.get_unregistered_chart_ids()}")
    
    return grid

def test_mock_callback_execution():
    """Test mock callback execution to verify functionality."""
    
    print("\n" + "=" * 50)
    print("Mock Callback Execution Test")
    print("=" * 50)
    
    # Generate mock store data
    mock_store_data = generate_mock_data()
    store_json = json.dumps(mock_store_data)
    
    print(f"Generated {len(mock_store_data)} mock records")
    
    # Test store update function
    print("\nTesting store update function:")
    test_chart_id = "test_individual_frame_chart_0"
    
    fig_store = store_update_function(
        chart_id=test_chart_id,
        store_data=store_json,
        commodity='cattle',
        countries=['Korea, South', 'Japan']
    )
    
    print(f"Store function returned: {type(fig_store)}")
    
    # Test function update function
    print("\nTesting function update function:")
    test_chart_id = "test_individual_frame_chart_2"
    
    fig_function = function_update_function(
        chart_id=test_chart_id,
        commodity='cattle',
        start_year=2023,
        end_year=2024
    )
    
    print(f"Function update returned: {type(fig_function)}")
    
    print("\nCallback execution tests completed successfully!")

if __name__ == '__main__':
    print("Individual Chart Callback Registration Test")
    print("=" * 60)
    
    # Run tests
    grid = test_individual_chart_registration()
    test_mock_callback_execution()
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL REGISTRATION TEST COMPLETED!")
    print("=" * 60)
    
    print("\nKey Features Demonstrated:")
    print("- Individual chart callback registration")
    print("- Mixed store and function callbacks in same grid")
    print("- Chart registry tracking and status reporting")
    print("- Flexible input/output ID management")
    print("- Separate update functions for different callback types")
    
    print("\nUsage Example:")
    print("""
# Create grid with some charts
grid = EnhancedFrameGrid(frames=[frame], flexible_menu=menu, data_source='my_store')

# Register individual charts
grid.register_chart_store_callback(
    app, 'chart_0', store_update_func, menu_inputs=['commodity', 'countries']
)

grid.register_chart_function_callback(
    app, 'chart_1', function_update_func, 
    input_components=['year', 'apply_btn'], trigger_component='apply_btn'
)

# Check status
grid.print_chart_registry_summary()
""")