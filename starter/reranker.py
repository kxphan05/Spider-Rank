"""Cross-encoder reranking of the fused candidate pool (spec Pillar I).

Two backends: `CrossEncoderReranker` (ms-marco-MiniLM-L-6-v2, ~90MB) is the
shipped, measured stage. `QwenReranker` (Qwen3-Reranker-0.6B, scores "yes" vs
"no" in one forward pass) is kept for the report -- at ~27s/pair on this
hardware it's too slow to sweep or A/B, so it doesn't ship by default.

Degrades like every other optional component: missing weights log and run
without reranking rather than failing to start.
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

    Raises on construction if weights are unavailable, so Agent.__init__ can
    catch and degrade the same way it does for the dense index/masked LM.
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

    Query and document go through the model together, unlike the dense leg
    where each is embedded independently and compared by cosine.
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
