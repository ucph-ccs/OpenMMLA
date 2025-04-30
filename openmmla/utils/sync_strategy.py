from typing import Any
from enum import Enum, auto


class SyncStrategy(Enum):
    """Enumeration of available synchronization strategies."""
    EARLIEST = auto()  # Choose the earliest time bucket
    NEAREST = auto()  # Choose the nearest time bucket
    LATEST = auto()  # Choose the latest time bucket


class TimeBucketSynchronizer:
    """Utility class for handling time bucket synchronization strategies."""

    @staticmethod
    def find_closest_time_bucket(
            current_time: float,
            time_buckets: dict[float, Any],
            base_id: str,
            time_range: float,
            strategy: SyncStrategy = SyncStrategy.EARLIEST
    ) -> float | None:
        """Find the closest time bucket based on the specified strategy.
        
        Args:
            current_time: The current time to find closest bucket for
            time_buckets: Dictionary of time buckets and their contents
            base_id: ID of the base sending the data
            time_range: Maximum time difference allowed for synchronization
            strategy: The synchronization strategy to use
            
        Returns:
            The selected time bucket timestamp, or None if no valid bucket found
        """
        if not time_buckets:
            return None

        # Calculate time differences for all buckets
        time_differences = {
            key: abs(current_time - key)
            for key in time_buckets.keys()
        }

        # Filter valid buckets based on time difference and exclude buckets where base_id exists
        valid_buckets = {
            key: diff for key, diff in time_differences.items()
            if diff <= time_range and base_id not in time_buckets[key]
        }

        if not valid_buckets:
            return None

        # Apply the selected strategy
        if strategy == SyncStrategy.EARLIEST:
            return min(valid_buckets.keys())
        elif strategy == SyncStrategy.NEAREST:
            return min(valid_buckets.keys(), key=lambda k: valid_buckets[k])
        elif strategy == SyncStrategy.LATEST:
            return max(valid_buckets.keys())
        else:
            raise ValueError(f"Unknown synchronization strategy: {strategy}")

    @staticmethod
    def get_expired_buckets(
            current_time: float,
            time_buckets: dict[float, Any],
            expiry_time: float
    ) -> list[float]:
        """Get list of expired time buckets.
        
        Args:
            current_time: Current time to check against
            time_buckets: Dictionary of time buckets and their contents
            expiry_time: Maximum age of a bucket before it's considered expired
            
        Returns:
            List of expired bucket timestamps
        """
        return [
            t for t in time_buckets.keys()
            if current_time - t > expiry_time
        ]
