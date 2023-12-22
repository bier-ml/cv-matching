import io
from typing import Union, Optional

import pandas as pd
from docx import Document

from core.embedding_models import DummyEmbeddingModel
from core.models.cosine_model import CosineModel
from core.models.tools import change_type_to_list


def read_docx(file: Union[bytes, bytearray]) -> str:
    doc = Document(io.BytesIO(file.read()))
    result = [p.text for p in doc.paragraphs]
    return "\n\n".join(result)


def get_relevant_vacancies(
    cv_text: str, n_recommendations: int = 3, model_class=CosineModel
) -> list[str]:
    embedding_model = DummyEmbeddingModel()
    model = model_class(embedding_model=embedding_model)
    cv_emb = embedding_model.generate(cv_text)

    dataset = pd.read_csv("data/vac.csv")[["vacancy_name", "embeddings"]]
    dataset["embeddings"] = dataset.embeddings.apply(lambda x: change_type_to_list(x))
    dataset["similarity"] = dataset.embeddings.apply(lambda x: model.predict(cv_emb, x))

    sorted_vacancies = dataset.sort_values(by="similarity", ascending=False)
    return sorted_vacancies.vacancy_name.head(n_recommendations).tolist()


def check_relevance(
    cv_text: str,
    job_name: str,
    job_description: Optional[str] = "",
    model_class=CosineModel,
) -> float:
    embedding_model = DummyEmbeddingModel()
    model = model_class(embedding_model=embedding_model)

    cv_emb = embedding_model.generate(cv_text)
    vac_text = "\n".join([job_name, job_description])
    vac_emb = embedding_model.generate(vac_text)

    similarity = model.predict(cv_emb, vac_emb)
    return similarity
