"""Evaluating miners' deployed models over the day's corpus.

Three pieces, in the order a cycle uses them:

- :mod:`~prometheon.evaluation.corpus` builds the shared, deterministically
  ordered corpus and the per-miner views that exclude a miner's own test items.
- :mod:`~prometheon.evaluation.runner` calls one miner's ``/moderate`` endpoint
  in batches and scores every failure as incorrect rather than skipping it.
- :mod:`~prometheon.evaluation.result` holds the accumulated per-miner result
  and the per-item records published with it.

Nothing here decides emissions. Scoring consumes these results, and because the
two are apart, a third party can recompute a cycle's weights from the published
evaluation without re-running any inference.
"""

from prometheon.evaluation.corpus import (
    EvaluationCorpus,
    ItemSource,
    LabelledItem,
    MinerCorpusCounts,
    build_corpus,
    corpus_order_key,
    iter_batches,
)
from prometheon.evaluation.result import (
    ItemVerdict,
    MinerEvaluation,
    MinerEvaluationBuilder,
)
from prometheon.evaluation.runner import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRANSPORT_RETRIES,
    MinerTarget,
    build_request_payload,
    evaluate_miner,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TRANSPORT_RETRIES",
    "EvaluationCorpus",
    "ItemSource",
    "ItemVerdict",
    "LabelledItem",
    "MinerCorpusCounts",
    "MinerEvaluation",
    "MinerEvaluationBuilder",
    "MinerTarget",
    "build_corpus",
    "build_request_payload",
    "corpus_order_key",
    "evaluate_miner",
    "iter_batches",
]
