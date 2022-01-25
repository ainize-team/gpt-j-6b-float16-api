from fastapi import FastAPI
from core.config import API_PREFIX, APP_NAME, APP_VERSION
from api.router import api_router
from core.event_handlers import start_app_handler, stop_app_handler


def get_app() -> FastAPI:
    fast_api_app = FastAPI(title=APP_NAME, version=APP_VERSION)
    fast_api_app.include_router(api_router, prefix=API_PREFIX)
    fast_api_app.add_event_handler("startup", start_app_handler(fast_api_app))
    fast_api_app.add_event_handler("shutdown", stop_app_handler(fast_api_app))
    return fast_api_app


app = get_app()
