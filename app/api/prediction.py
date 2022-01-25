from fastapi import APIRouter
from starlette.requests import Request

from models.payload import TextGenerationPredictPayload
from models.prediction import TextGenerationResult

router = APIRouter()


@router.post("/text-generation", name="text-generation", response_model=TextGenerationResult,
             responses={413: {"description": "Error: Request Entity Too Large"}}, )
def post_text_generation(
        request: Request,
        block_data: TextGenerationPredictPayload
) -> TextGenerationResult:
    model = request.app.state.model
    prediction = model.predict(block_data)
    return prediction
