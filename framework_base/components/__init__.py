"""
UI Components
=============

Reusable dashboard components for building commodity analytics interfaces.
Includes advanced chart containers, layout managers, and interactive controls.

Classes:
    FundamentalFrame: Advanced chart container with data binding
    FlexibleMenu: Dynamic control panel system
    EnhancedFrameGrid: Advanced layout manager with callback registration
    
Utilities:
    callback_utils: Callback management and registry utilities
    chart_components: Chart generation and configuration utilities
"""

from .frames import (
    FundamentalFrame,
    FlexibleMenu, 
    EnhancedFrameGrid
)

from .chart_components import (
    ChartGenerator,
    create_line_chart,
    create_bar_chart,
    create_area_chart,
    create_scatter_chart
)

try:
    from .callback_utils import (
        CallbackRegistry,
        register_callback,
        get_callback_inputs,
        get_callback_outputs
    )
    HAS_CALLBACK_UTILS = True
except ImportError:
    HAS_CALLBACK_UTILS = False

__all__ = [
    'FundamentalFrame', 'FlexibleMenu', 'EnhancedFrameGrid',
    'ChartGenerator', 'create_line_chart', 'create_bar_chart', 
    'create_area_chart', 'create_scatter_chart'
]

if HAS_CALLBACK_UTILS:
    __all__.extend([
        'CallbackRegistry', 'register_callback', 
        'get_callback_inputs', 'get_callback_outputs'
    ])