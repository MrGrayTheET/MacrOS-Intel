from typing import Dict, List, Tuple, Callable, Any
from dash import callback, Output, Input, State, no_update
import inspect

class CallbackRegistry:
    """
    A registry pattern for managing callbacks in complex dashboards.
    This provides better organization and reusability.
    """

    def __init__(self):
        self.registered_callbacks = {}
        self.callback_groups = {}

    def register(self,
                 name: str,
                 outputs: List[Tuple[str, str]],
                 inputs: List[Tuple[str, str]],
                 states: List[Tuple[str, str]] = None,
                 prevent_initial_call: bool = True,
                 group: str = None):
        """
        Decorator to register a callback function.

        Usage:
            @registry.register(
                name='update_chart',
                outputs=[('chart-1', 'figure')],
                inputs=[('button-1', 'n_clicks')],
                group='charts'
            )
            def update_chart_callback(n_clicks):
                return new_figure
        """

        def decorator(func: Callable):
            callback_config = {
                'function': func,
                'outputs': outputs,
                'inputs': inputs,
                'states': states or [],
                'prevent_initial_call': prevent_initial_call,
                'group': group
            }

            self.registered_callbacks[name] = callback_config

            if group:
                if group not in self.callback_groups:
                    self.callback_groups[group] = []
                self.callback_groups[group].append(name)

            return func

        return decorator

    def create_callback(self, name: str):
        """Create and register a Dash callback from registered config."""
        if name not in self.registered_callbacks:
            raise ValueError(f"Callback '{name}' not found in registry")

        config = self.registered_callbacks[name]

        outputs = [Output(comp_id, prop) for comp_id, prop in config['outputs']]
        inputs = [Input(comp_id, prop) for comp_id, prop in config['inputs']]
        states = [State(comp_id, prop) for comp_id, prop in config['states']]

        @callback(
            *outputs,
            *inputs,
            *states,
            prevent_initial_call=config['prevent_initial_call']
        )
        def wrapper(*args, **kwargs):
            try:
                return config['function'](*args, **kwargs)
            except Exception as e:
                print(f"Error in callback '{name}': {str(e)}")
                return [no_update] * len(config['outputs'])

        return wrapper

    def create_all_callbacks(self):
        """Create all registered callbacks."""
        for name in self.registered_callbacks:
            self.create_callback(name)

    def create_group_callbacks(self, group: str):
        """Create callbacks for a specific group."""
        if group not in self.callback_groups:
            raise ValueError(f"Group '{group}' not found")

        for name in self.callback_groups[group]:
            self.create_callback(name)

    def get_callback_info(self, name: str = None, group: str = None):
        """Get information about registered callbacks."""
        if name:
            return self.registered_callbacks.get(name)
        elif group:
            return {name: self.registered_callbacks[name]
                    for name in self.callback_groups.get(group, [])}
        else:
            return self.registered_callbacks
