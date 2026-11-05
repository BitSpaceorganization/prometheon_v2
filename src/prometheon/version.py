"""Single source of the package version and the scoring version."""

from typing import Final

__version__: Final[str] = "3.0.0"

#: Names the reward rules a weight vector was computed under. It goes into every
#: signed evaluation record, into the epoch the hub signs, and into every
#: validator config, so a vector can be traced to its rules and a config written
#: for different rules is refused at load.
#:
#: Bump it in the same change as anything that would make two validators
#: compute different weights from identical inputs. It is the only copy: an
#: earlier version kept a second constant here that nothing read, while the one
#: that was published sat unbumped through three weight-changing releases.
SCORING_VERSION: Final[str] = "prometheon-scoring/3.0"

__all__ = ["SCORING_VERSION", "__version__"]
