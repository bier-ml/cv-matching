import numpy as np

from core.embedding_models import DummyEmbeddingModel
from core.models.base_model import BaseMatchingModel


class CosineModel(BaseMatchingModel):
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    @staticmethod
    def cosine_similarity(a, b):
        return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

    def predict(self, vacancy: str | np.ndarray, cv: str | np.ndarray) -> float:
        if self.embedding_model is None:
            self.embedding_model = DummyEmbeddingModel()

        if isinstance(vacancy, str):
            vacancy = self.embedding_model.generate(vacancy)

        if isinstance(cv, str):
            cv = self.embedding_model.generate(cv)

        return self.cosine_similarity(cv, vacancy)
