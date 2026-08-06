"""Single source of the package version."""

from typing import Final

__version__: Final[str] = "2.0.0"

#: Stamped into every evaluation submission so a result can be traced to the
#: code that produced it. Bumped whenever scoring changes in a way that would
#: alter a weight vector.
SCORING_VERSION: Final[str] = "prometheon-scoring/2.0.0"

__all__ = ["SCORING_VERSION", "__version__"]
