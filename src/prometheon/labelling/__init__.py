"""Ground-truth labelling: what the corpus actually says, per the policy.

A validator labels the day's corpus once, with an LLM reading
``content_policy.md``, and every miner is then scored against those labels. The
labels are the measurement, so this package is written around one assumption:
the content being labelled is hostile. Making subversion unlikely is not
enough. A labelling call that was subverted has to be *detectable*.

- :mod:`prometheon.labelling.prompt` frames the content as data and pins the
  response schema.
- :mod:`prometheon.labelling.batch` enforces that schema and contains a bad
  call by splitting it, down to excluding a single item.
- :mod:`prometheon.labelling.client` is the transport, injectable end to end.
"""

from prometheon.labelling.batch import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SINGLE_ITEM_ATTEMPTS,
    LabelledCorpus,
    label_items,
    parse_label_response,
)
from prometheon.labelling.client import (
    ChatCompletionClient,
    ChatResponse,
    OpenAICompatibleClient,
)
from prometheon.labelling.prompt import (
    LabelItem,
    build_system_prompt,
    build_user_prompt,
    new_fence,
    render_items,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SINGLE_ITEM_ATTEMPTS",
    "ChatCompletionClient",
    "ChatResponse",
    "LabelItem",
    "LabelledCorpus",
    "OpenAICompatibleClient",
    "build_system_prompt",
    "build_user_prompt",
    "label_items",
    "new_fence",
    "parse_label_response",
    "render_items",
]
