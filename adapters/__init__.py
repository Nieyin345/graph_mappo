"""Reference capacity and channel adapters.

Keep package initialization lightweight. Some optional scenario/entity adapters
depend on packages that are not part of this repository, so callers should
import those modules explicitly when their dependencies are available.
"""

from adapters.unified_channel import RateModelInput, UnifiedLinkModel, UnifiedQKDRateModel

__all__ = ["RateModelInput", "UnifiedLinkModel", "UnifiedQKDRateModel"]
