"""Neural-network model components.

The initializer intentionally has no eager imports. Model modules are large and
Torch-dependent; importing the namespace should stay cheap and avoid creating
an unnecessary dependency edge between package discovery and the full model.
"""

__all__: list[str] = []
