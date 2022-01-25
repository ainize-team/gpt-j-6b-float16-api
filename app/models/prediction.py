from typing import List
from pydantic import BaseModel


class TextGenerationResult(BaseModel):
    generated_text: List[str]
