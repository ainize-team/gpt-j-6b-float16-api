from typing import Callable

from fastapi import FastAPI
from loguru import logger

from core.config import MODEL_NAME_OR_PATH, TASK, IS_FP16, REVISION, LOW_CPU_MEM_USAGE
from models.nlp import TextGenerationModel
from core.commons import Tasks


def _startup_model(app: FastAPI) -> None:
    logger.info(f"Model Name Or Path: {MODEL_NAME_OR_PATH}")
    logger.info(f"Revision: {REVISION}")
    if TASK == Tasks.TEXT_GENERATION.value:
        model_instance = TextGenerationModel(MODEL_NAME_OR_PATH, REVISION, IS_FP16, LOW_CPU_MEM_USAGE)
    else:
        raise ValueError(f"{TASK} is not supported")
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
