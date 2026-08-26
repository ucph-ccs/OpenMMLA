"""Validation utilities for OpenMMLA."""


def validate_unix_timestamp(timestamp: float) -> bool:
    """Validate that a timestamp is a reasonable Unix timestamp.

    Args:
        timestamp: the timestamp to validate

    Returns:
        True if the timestamp is a valid Unix timestamp, False otherwise.
    """
    # unix timestamps should be positive and within reasonable bounds
    # January 1, 1970 00:00:00 UTC = 0
    # January 1, 2100 00:00:00 UTC = 4102444800
    min_timestamp = 0
    max_timestamp = 4102444800  # year 2100

    if isinstance(timestamp, (int, float)) and min_timestamp < timestamp < max_timestamp:
        return True
    else:
        return False
