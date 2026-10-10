"""Feature extraction shared by training and the API (same code path for both).

A ticket becomes one sparse row built from five blocks:

  1. word TF-IDF            on the normalised text
  2. char n-gram TF-IDF     on the normalised text      (robust to typos / spelling variation)
  3. char n-gram TF-IDF     on the ROMANISED text       (Sinhala <-> Singlish, Tamil <-> Tanglish share features)
  4. concept lexicon counts on the romanised text        (domain keywords, see lexicon.py)
  5. concept lexicon counts on the romanised text with known boilerplate removed

Block 5 exists because tickets are mostly boilerplate ("I have kept a copy of the confirmation",
"mama phone eken baluwe") wrapped around one or two sentences that carry the real issue.
Boilerplate is learned from the training data: a word trigram that occurs in tickets of many
different categories cannot be about any one category, so it is treated as filler and masked
before the lexicon is applied.
"""
from __future__ import annotations

import collections
import re
import unicodedata

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .lexicon import CONCEPTS, concept_counts, concept_matches
from .romanize import romanize

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9'-]+|[^\sa-z0-9]")


def normalize(subject: str | None, text: str | None) -> str:
    """Join subject + body, Unicode-normalise (NFC keeps combining marks), lowercase, collapse whitespace."""
    joined = (subject or "") + " \n " + (text or "")
    return _WS.sub(" ", unicodedata.normalize("NFC", joined).lower()).strip()


def tokenize(rom_text: str) -> list[str]:
    return _TOKEN.findall(rom_text)


def learn_fillers(token_lists, labels, min_tickets: int = 6, min_categories: int = 5) -> set[str]:
    """Word trigrams seen in >= min_tickets tickets spread over >= min_categories categories."""
    seen = collections.defaultdict(collections.Counter)
    for toks, label in zip(token_lists, labels):
        for gram in set(zip(toks, toks[1:], toks[2:])):
            seen[gram][label] += 1
    return {
        " ".join(gram)
        for gram, cats in seen.items()
        if sum(cats.values()) >= min_tickets and len(cats) >= min_categories
    }


def mask_fillers(toks: list[str], fillers: set[str]) -> str:
    keep = [True] * len(toks)
    for i in range(len(toks) - 2):
        if f"{toks[i]} {toks[i + 1]} {toks[i + 2]}" in fillers:
            keep[i] = keep[i + 1] = keep[i + 2] = False
    return " ".join(tok if k else "|" for tok, k in zip(toks, keep))


class Featurizer:
    """fit() on training tickets, transform() on any tickets. Pickled inside the model artifact."""

    def __init__(self, use_word=True, use_char=True, use_rom=True, use_lexicon=True, use_masked_lexicon=True,
                 lexicon_weight: float = 0.5):
        self.use_word = use_word
        self.use_char = use_char
        self.use_rom = use_rom
        self.use_lexicon = use_lexicon
        self.use_masked_lexicon = use_masked_lexicon
        self.lexicon_weight = lexicon_weight
        self.word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, min_df=2)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.rom_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.fillers: set[str] = set()

    # -- helpers ---------------------------------------------------------
    def _lexicon_block(self, rom_texts) -> list[np.ndarray]:
        blocks = []
        if self.use_lexicon:
            blocks.append(np.array([concept_counts(t) for t in rom_texts], dtype=np.float32))
        if self.use_masked_lexicon:
            masked = [mask_fillers(tokenize(t), self.fillers) for t in rom_texts]
            blocks.append(np.array([concept_counts(t) for t in masked], dtype=np.float32))
        return blocks

    def _stack(self, texts, rom_texts, fit: bool):
        parts = []
        if self.use_word:
            parts.append(self.word_vec.fit_transform(texts) if fit else self.word_vec.transform(texts))
        if self.use_char:
            parts.append(self.char_vec.fit_transform(texts) if fit else self.char_vec.transform(texts))
        if self.use_rom:
            parts.append(self.rom_vec.fit_transform(rom_texts) if fit else self.rom_vec.transform(rom_texts))
        for block in self._lexicon_block(rom_texts):
            parts.append(sp.csr_matrix(block.reshape(len(rom_texts), -1) * self.lexicon_weight))
        return sp.hstack(parts).tocsr()

    # -- public API ------------------------------------------------------
    def fit_transform(self, texts: list[str], labels) -> sp.csr_matrix:
        rom_texts = [romanize(t) for t in texts]
        if self.use_masked_lexicon:
            self.fillers = learn_fillers([tokenize(t) for t in rom_texts], labels)
        return self._stack(texts, rom_texts, fit=True)

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        rom_texts = [romanize(t) for t in texts]
        return self._stack(texts, rom_texts, fit=False)

    def fired_concepts(self, text: str) -> list[str]:
        """Concepts found in the boilerplate-masked ticket (used by the demo's explanation panel)."""
        masked = mask_fillers(tokenize(romanize(text)), self.fillers)
        return [c for c, v in zip(CONCEPTS, concept_counts(masked)) if v > 0]

    def matched_keywords(self, text: str) -> list[dict]:
        """Which words triggered which concept, after boilerplate masking (demo explanation panel)."""
        masked = mask_fillers(tokenize(romanize(text)), self.fillers)
        return [{"concept": c, "words": w} for c, w in concept_matches(masked).items()]
