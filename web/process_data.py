import io
from typing import Union

from docx import Document


def read_docx(file: Union[bytes, bytearray]) -> str:
    doc = Document(io.BytesIO(file.read()))
    result = [p.text for p in doc.paragraphs]
    return "\n\n".join(result)
