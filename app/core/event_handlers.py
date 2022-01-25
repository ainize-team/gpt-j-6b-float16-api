from typing import Callable

from fastapi import FastAPI
from loguru import logger

from core.config import MODEL_NAME_OR_PATH, TASK, IS_FP16, REVISION
from models.nlp import TextGenerationModel
from core.commons import Tasks


def _startup_model(app: FastAPI) -> None:
    model_name_or_path = MODEL_NAME_OR_PATH
    task = TASK
    is_fp16 = IS_FP16
    revision = REVISION
    logger.info(f"Revision {revision}")
    if task == Tasks.TEXT_GENERATION.value:
        model_instance = TextGenerationModel(model_name_or_path, revision, is_fp16)
    else:
        raise ValueError(f"{task} is not supported")
    app.state.model = model_instance


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)

    return shutdown
