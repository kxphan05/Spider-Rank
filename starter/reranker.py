"""Cross-encoder reranking of the fused candidate pool (spec Pillar I).

The competition spec's Pillar I names a "Multi-Route Retrieval -> LLM Semantic
Ranking" pipeline, and until now this project had only the first half:
retrieval is BM25 + dense fused by weighted RRF, then reordered by categorical
attribute agreement and MMR. Nothing learned or generative ever looked at a
(query, product) pair jointly. That gap is CLAUDE.md #6.

`Qwen/Qwen3-Reranker-0.6B` fills it. It is a causal LM repurposed as a binary
relevance judge: the pair is formatted into a chat template ending at the
assistant turn, and the score is the model's probability of emitting "yes"
rather than "no" at the next position. That means **one forward pass per pair
and no token generation**, which is the only reason this is tractable here --
generation on this hardware runs at under 2 tokens/second, while a
prefill-only pass over a few hundred tokens is roughly a second.

Why 0.6B and not something larger: measured on the target machine (i5-8365U,
no CUDA, ~4.6 GB free RAM), a 2B *generative* model needs ~8.4 s of wall time
for a single 41-token scoring call. An 8B model would need roughly 180 hours
for one 200-sample evaluation, which does not merely make it slow, it makes it
unmeasurable -- and this project does not ship unmeasured changes. A 0.6B
cross-encoder scoring the pool head is the largest version of this idea that
can actually be A/B'd on the full public set.

Degrades like every other optional component here (CLAUDE.md #15): if the
weights are missing the agent logs and runs without reranking, rather than
failing to start.
"""
from __future__ import annotations

import logging

from .paths import model_cache_dir

logger = logging.getLogger(__name__)

RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

# Characters of product text shown to the reranker. Cost here is dominated by
# prefill, which is linear in prompt length, so this is the main runtime knob.
# Long enough to carry the title plus the first feature or two, which is where
# material/colour/closure information actually lives in this catalog.
MAX_DOC_CHARS = 400

# Pairs scored per forward pass. Kept small: peak RSS matters more than
# throughput on a 4.6 GB-free machine, and padding waste grows with batch size
# when document lengths are uneven.
BATCH_SIZE = 4

INSTRUCTION = (
    "Given a shopper's request, judge whether the product satisfies it."
)


def _format_pair(query: str, document: str) -> str:
    return (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n'
        f"<|im_start|>user\n<Instruct>: {INSTRUCTION}\n<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


class QwenReranker:
    """Binary-relevance cross-encoder over (query, product) pairs.

    Raises on construction if the weights are unavailable, so `Agent.__init__`
    can catch and degrade the same way it does for the dense index and the
    masked LM.
    """

    def __init__(self, model_name: str = RERANK_MODEL_NAME, cache_dir: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        cache_dir = cache_dir or model_cache_dir()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self._model.eval()
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        if self._yes_id is None or self._no_id is None:
            raise RuntimeError("reranker tokenizer lacks yes/no tokens")
        logger.info("QwenReranker: loaded %s", model_name)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        """P(relevant) for each document, in input order."""
        torch = self._torch
        out: list[float] = []
        for start in range(0, len(documents), BATCH_SIZE):
            chunk = documents[start:start + BATCH_SIZE]
            prompts = [_format_pair(query, doc[:MAX_DOC_CHARS]) for doc in chunk]
            encoded = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                logits = self._model(**encoded).logits[:, -1, :]
            # Left padding means the final position is the real last token for
            # every row in the batch, which is what makes batching safe here.
            pair = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=1)
            probs = torch.log_softmax(pair.float(), dim=1).exp()
            out.extend(probs[:, 1].tolist())
        return out
