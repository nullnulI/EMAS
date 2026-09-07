"""MAP-THOR benchmark integration for EMAS.

The benchmark package owns episode discovery, official scene initialization,
action-level metric tracking, batch aggregation, and durable result files.
The production hybrid entry point is intentionally left unchanged.
"""

from .episode import MapThorEpisode

__all__ = ["MapThorEpisode"]
