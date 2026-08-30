"""Cross-encoder reranking of the fused candidate pool (spec Pillar I).

The competition spec's Pillar I names a "Multi-Route Retrieval -> LLM Semantic
Ranking" pipeline, and until now this project had only the first half:
retrieval is BM25 + dense fused by weighted RRF, then reordered by categorical
attribute agreement and MMR. Nothing learned or generative ever looked at a
(query, product) pair jointly, until this module.

`Qwen/Qwen3-Reranker-0.6B` fills it. It is a causal LM repurposed as a binary
relevance judge: the pair is formatted into a chat template ending at the
assistant turn, and the score is the model's probability of emitting "yes"
rather than "no" at the next position. That means **one forward pass per pair
and no token generation**, which is the only reason this is tractable here --
generation on this hardware runs at under 2 tokens/second, while a
prefill-only pass over a few hundred tokens is roughly a second.

Two backends ship, and the split is a measured decision rather than
indecision.

`CrossEncoderReranker` (ms-marco-MiniLM-L-6-v2, 22M parameters, ~90 MB) is the
**shipped, measured** stage. `QwenReranker` (Qwen3-Reranker-0.6B) is kept for
the report and the demo as the LLM-scale variant.

The reason is arithmetic, not preference. Measured on the target machine
(i5-8365U, no CUDA, ~4.6 GB free RAM), the Qwen reranker scores a
(query, product) pair in **~27 s** loaded as fp32 through `transformers`. Its
judgements are good -- 0.999 for an exact match, 0.679 for a same-category
alternative, 0.001 for off-category -- but reranking a 10-item slate across
~1000 evaluation turns is roughly 75 hours for a *single* configuration, so it
cannot be A/B'd, let alone swept. An unmeasurable component is one this
project will not ship on by default -- plausible retrieval changes have gone
unverified here before and cost score.

The same argument already excluded larger models: an 8B generative model
implies ~180 hours per evaluation on this hardware.

A classic cross-encoder is the version of this idea that can actually be
measured, and it closes the same structural gap -- nothing in the pipeline
previously scored a (query, product) pair *jointly*, which is what
distinguishes a reranker from the bi-encoder dense leg.

Degrades like every other optional component here: if the
weights are missing the agent logs and runs without reranking, rather than
failing to start.
"""
from __future__ import annotations

import logging

from .paths import model_cache_dir
from .config import (BATCH_SIZE, MAX_DOC_CHARS, MINILM_MAX_LENGTH, MINILM_MODEL_NAME, 
    RERANK_MODEL_NAME)

logger = logging.getLogger(__name__)




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


class CrossEncoderReranker:
    """Classic cross-encoder relevance scoring (ms-marco-MiniLM-L-6-v2).

    Same interface as QwenReranker and the same structural role -- query and
    document go through the model *together*, so the score can depend on their
    interaction, unlike the dense leg where each is embedded independently and
    compared by cosine. That joint scoring is the whole point of a reranking
    stage, and this project ran without one for most of its life.

    Trained on MS MARCO passage ranking, so it is a purpose-built reranker
    rather than a general LM steered into the role, which is also why it is
    roughly three orders of magnitude cheaper per pair than the Qwen backend.
    """

    def __init__(self, model_name: str = MINILM_MODEL_NAME, cache_dir: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            model_name, max_length=MINILM_MAX_LENGTH, cache_folder=cache_dir or model_cache_dir()
        )
        logger.info("CrossEncoderReranker: loaded %s", model_name)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        """Relevance logit per document, in input order.

        Returned unnormalized: the consumer only ranks by these, and RRF
        fusion in `Agent._rerank` uses the resulting *order*, never the
        magnitudes.
        """
        if not documents:
            return []
        pairs = [(query, doc[:MAX_DOC_CHARS]) for doc in documents]
        return [float(value) for value in self._model.predict(pairs)]
