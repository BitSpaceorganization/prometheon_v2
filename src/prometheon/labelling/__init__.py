"""Ground-truth labelling: what each post actually is, per the policy.

A validator labels the day's posts once, with an LLM reading
``content_policy.md``, and every miner is then scored against those labels. The
labels are the measurement, so this package is written around one assumption:
the content being labelled is hostile.

- :mod:`prometheon.labelling.prompt` frames each post as data and pins the
  response schema.
- :mod:`prometheon.labelling.labeller` asks about one post per request, a
  bounded number at a time, enforces the schema, and excludes a post that
  never labels cleanly rather than guessing.
- :mod:`prometheon.labelling.client` is the transport, with retries on rate
  limits and server errors, injectable end to end.
"""

from prometheon.labelling.client import (
    ChatCompletionClient,
    ChatResponse,
    OpenAICompatibleClient,
)
from prometheon.labelling.labeller import (
    DEFAULT_ATTEMPTS,
    DEFAULT_MAX_IN_FLIGHT,
    LabelledCorpus,
    label_items,
    parse_label_response,
)
from prometheon.labelling.prompt import (
    LabelItem,
    build_system_prompt,
    build_user_prompt,
    new_fence,
    render_item,
)

__all__ = [
    "DEFAULT_ATTEMPTS",
    "DEFAULT_MAX_IN_FLIGHT",
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
    "render_item",
]
