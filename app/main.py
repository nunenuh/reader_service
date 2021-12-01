import uvicorn
from api.routes.api import router as api_router
from core.config import API_PREFIX, DEBUG, PROJECT_NAME, VERSION
from core import config
from core.events import create_start_app_handler
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
import os
from tests import reader_test


def get_application() -> FastAPI:
    application = FastAPI(title=PROJECT_NAME, debug=DEBUG, version=VERSION)
    application.include_router(api_router, prefix=API_PREFIX)
    pre_load = False
    if pre_load:
        application.add_event_handler("startup", create_start_app_handler(application))
    return application


app = get_application()
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(config.STATIC_URL, StaticFiles(directory=config.STATIC_DIR), name=config.STATIC_DIR)

if __name__ == "__main__":
    reader_test.run_test()
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, debug=False)
