from starlette.config import Config

APP_VERSION = "1.0.0"
APP_NAME = "GPT-J-6B Float16"
API_PREFIX = "/api"
TASK = "TEXT_GENERATION"

config = Config(".env")

MODEL_NAME_OR_PATH: str = config("MODEL_NAME_OR_PATH")
IS_FP16: bool = config("IS_FP16", cast=bool, default=False)
REVISION: str = config("REVISION", cast=str, default=None)
LOW_CPU_MEM_USAGE: bool = config("LOW_CPU_MEM_USAGE", cast=bool, default=False)
