"""Single source of the package version, the scoring version and the submission fee."""

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

#: The model submission fee: alpha per credit. A published rule, not a setting.
#: The hub prices credits at exactly this, so a miner who could change it
#: locally would send an amount the hub does not count the way they expect.
#: README.md and docs/miner.md state it, and a contract test holds them to it.
SUBMISSION_FEE_ALPHA: Final[int] = 50

__all__ = ["SCORING_VERSION", "SUBMISSION_FEE_ALPHA", "__version__"]
