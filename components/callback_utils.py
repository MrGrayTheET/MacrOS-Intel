"""
Callback utilities for handling component data transformations
"""

from typing import List, Union, Any


def parse_searchable_checklist_value(value: Union[str, List[str], None]) -> List[str]:
    """
    Parse the value from a searchable checklist component.
    
    The searchable checklist uses '||' as delimiter to avoid conflicts with 
    country names that contain commas (like "Korea, South").
    
    Args:
        value: Value from searchable checklist (||- delimited string, list, or None)
    
    Returns:
        List of selected values
    """
    if value is None:
        return []
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        if not value.strip():
            return []
        
        # Use || delimiter to split values
        if '||' in value:
            return [item.strip() for item in value.split('||') if item.strip()]
        else:
            # Fallback for single item or legacy format
            return [value.strip()] if value.strip() else []
    
    return []


def format_countries_for_callback(countries_value: Any, default: List[str] = None) -> List[str]:
    """
    Format countries value for use in ESR callbacks.
    
    Handles both traditional checklist format (list) and searchable checklist format (string).
    
    Args:
        countries_value: Value from country selection component
        default: Default countries to use if value is empty
    
    Returns:
        List of country names
    """
    default = default or ['Korea, South']
    
    if countries_value is None:
        return default
    
    # Parse the value
    countries = parse_searchable_checklist_value(countries_value)
    
    # Return default if no countries selected
    if not countries:
        return default
    
    return countries


def prepare_menu_values_for_esr_callback(menu_values: dict) -> dict:
    """
    Prepare menu values for ESR callbacks by parsing searchable checklist values.
    
    Args:
        menu_values: Dictionary of menu input values
    
    Returns:
        Dictionary with parsed values
    """
    processed_values = menu_values.copy()
    
    # Parse countries if present
    if 'countries' in processed_values:
        processed_values['countries'] = format_countries_for_callback(
            processed_values['countries'], 
            default=['Korea, South', 'Japan', 'China']
        )
    
    return processed_values


# For backward compatibility and easier importing
def get_countries_list(value: Any, default: List[str] = None) -> List[str]:
    """Alias for format_countries_for_callback"""
    return format_countries_for_callback(value, default)