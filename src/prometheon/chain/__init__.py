"""Everything that touches the chain.

Layered so only one file imports the SDK:

- :mod:`~prometheon.chain.metagraph`: the typed snapshot, hotkey → UID.
- :mod:`~prometheon.chain.commitment`: the 128-byte model commitment codec.
- :mod:`~prometheon.chain.weights`: submission gates and the u16 conversion.
- :mod:`~prometheon.chain.subtensor`: the SDK adapter, lazily imported.

Importing this package does not import bittensor, so the scoring and evaluation
paths can be exercised without the SDK present.
"""

from prometheon.chain.commitment import (
    COMMITMENT_MAX_BYTES,
    ModelCommitment,
    commitment_size_bytes,
    decode_commitment,
    encode_commitment,
    publish_commitment,
    read_commitment,
)
from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.subtensor import (
    connect,
    detect_capabilities,
    read_hyperparameters,
    read_subnet_owner_hotkey,
    submit_set_weights,
    sync_metagraph_view,
)
from prometheon.chain.weights import (
    CHAIN_WEIGHT_UNITS,
    ChainCapabilities,
    ChainHyperparameters,
    ChainWeightVector,
    WeightAllocation,
    assert_submission_allowed,
    resolve_allocations,
    to_u16_chain_vector,
)

__all__ = [
    "CHAIN_WEIGHT_UNITS",
    "COMMITMENT_MAX_BYTES",
    "ChainCapabilities",
    "ChainHyperparameters",
    "ChainWeightVector",
    "MetagraphView",
    "ModelCommitment",
    "WeightAllocation",
    "assert_submission_allowed",
    "commitment_size_bytes",
    "connect",
    "decode_commitment",
    "detect_capabilities",
    "encode_commitment",
    "publish_commitment",
    "read_commitment",
    "read_hyperparameters",
    "read_subnet_owner_hotkey",
    "resolve_allocations",
    "submit_set_weights",
    "sync_metagraph_view",
    "to_u16_chain_vector",
]
