from fastapi import APIRouter
from starlette.requests import Request

from core.config import TASK
from core.commons import Tasks
from models.payload import TextGenerationPredictPayload
from models.prediction import TextGenerationResult

router = APIRouter()


@router.post("/text-generation", name="text-generation")
def post_text_generation(
        request: Request,
        block_data: TextGenerationPredictPayload
) -> TextGenerationResult:
    model = request.app.state.model
    prediction = model.predict(block_data)
    return prediction
