from dash import html, dcc


def create_dd_menu(table_client, menu_id='enhanced-psd-menu', header_text=None, key_filter="psd", style={}):
    """
    Enhanced PSD menu with additional filtering and information.

    Args:
        table_client: Your TableClient instance
        menu_id: ID for the menu container

    Returns:
        html.Div: Enhanced menu div with PSD dropdown and info
        component_ids : Dictionary containing values of menu component_id(s)
    """
    # Get all available keys
    try:
        if hasattr(table_client, 'available_keys'):
            all_keys = table_client.available_keys()
        else:
            all_keys = []
    except:
        all_keys = []

    # Filter and categorize PSD keys
    psd_keys = [key for key in all_keys if key_filter in key.lower()]

    # Try to categorize PSD keys by commodity/region
    categorized_options = []
    commodities = set()

    for key in psd_keys:
        # Extract potential commodity name
        key_parts = key.lower().split('/')
        commodity = key_parts[0]

        commodities.add(commodity)

        display_name = commodity
        categorized_options.append({
            'label': f"{commodity}",
            'value': key,
        })

    # Sort options by label
    categorized_options.sort(key=lambda x: (x['label']))

    menu_div = html.Div([
        html.H4("PSD Data Selection",
                style={
                    'textAlign': 'center',
                    'marginBottom': '15px',
                    'color': '#333'
                }),

        # Info about available datasets
        html.Div([
            html.P(f"Found {len(psd_keys)} PSD datasets",
                   style={'fontSize': '12px', 'color': '#666', 'textAlign': 'center'}),
            html.P(f"Commodities: {', '.join(sorted(commodities))}",
                   style={'fontSize': '10px', 'color': '#999', 'textAlign': 'center'})
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Select PSD Dataset:",
                       style={
                           'fontWeight': 'bold',
                           'marginBottom': '5px',
                           'display': 'block'
                       }),

            dcc.Dropdown(
                id=f'{menu_id}-dropdown',
                options=categorized_options,
                value=categorized_options[0]['value'] if categorized_options else None,
                placeholder="Select a PSD dataset...",
                style={'marginBottom': '10px'},
                searchable=True
            ),

            html.Button(
                "Load Selected Data",
                id=f'{menu_id}-load-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#28a745',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%',
                    'fontWeight': 'bold',
                    'marginBottom': '10px'
                }
            ),

            html.Button(
                "Refresh Keys",
                id=f'{menu_id}-refresh-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#6c757d',
                    'color': 'white',
                    'padding': '8px 16px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'width': '100%',
                    'fontSize': '12px'
                }
            )
        ])
    ],
        id=menu_id,
        style={
            'width': '350px',
            'padding': '20px',
            'border': '2px solid #28a745',
            'borderRadius': '10px',
            'backgroundColor': '#f8f9fa',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.15)',
            'margin': '20px'
        })

    input_components = {'dd':f'{menu_id}-dropdown', 'load-btn':f'{menu_id}-load-btn', 'refresh-btn':f'{menu_id}-refresh-btn'}

    return menu_div, input_components
