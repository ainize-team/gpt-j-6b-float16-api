from fastapi import APIRouter
from api import prediction

api_router = APIRouter()
api_router.include_router(prediction.router, tags=["prediction"])
