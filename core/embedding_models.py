from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

AVAILABLE_EMBEDDINGS: set[str] = {
    "cointegrated/rubert-tiny2",
    "DeepPavlov/rubert-base-cased",
    "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
    "DeepPavlov/rubert-base-cased-sentence",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
}


class DummyEmbeddingModel:
    def __init__(self):
        np.random.seed(42)

    @staticmethod
    def generate(text: str | list[str], dim: int = 312) -> np.ndarray | Iterable[np.ndarray]:
        if isinstance(text, str):
            return np.random.rand(dim)
        else:
            return np.random.rand(len(text), dim)


class EmbeddingModel:
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        self.model = SentenceTransformer(model_name)

    def generate(self, text: str | Iterable[str]) -> np.ndarray | Iterable[np.ndarray]:
        return self.model.encode(text)
