import dash
from dash import Input, Output, State

def register_global_callbacks(app):
    """Register simplified global application callbacks"""

    @app.callback(
        Output('global-alerts', 'children'),
        Input('url', 'pathname'),
        prevent_initial_call=True
    )
    def update_global_alerts(pathname):
        """Handle global alerts and notifications"""
        # Add any global alert logic here
        return ""

    @app.callback(
        Output('global-settings', 'data'),
        Input('app-state', 'data'),
        State('global-settings', 'data'),
        prevent_initial_call=True
    )
    def sync_global_settings(app_state, current_settings):
        """Sync application state with global settings"""
        if app_state and app_state.get('settings'):
            settings = current_settings or {}
            settings.update(app_state['settings'])
            return settings
        return current_settings


