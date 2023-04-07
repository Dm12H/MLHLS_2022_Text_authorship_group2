from fastapi import FastAPI, Request, Form, Header
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.types import Scope, Receive, Send

from app.logs import log_request, log_server_startup, set_logs
from app.config import get_model_names
from app.app_models.model_manager import ModelDep, TransformDep, ModelHolder
from app.app_models.inference import predict_text

import logging
import logging.config
from typing import Annotated
from contextlib import asynccontextmanager
from uuid import uuid4, UUID


set_logs()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with log_server_startup(logger):
        ModelHolder.load_from_settings()
    yield


app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory="app/forms/temp")

app.mount("/stat", StaticFiles(directory="app/forms/stat"), name="stat")

class Middleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        id = uuid4()
        scope["headers"].append((b"x-request-id", str(id).encode()))
        async with log_request(logger, id=id, request=Request(scope)):
            await self.app(scope, receive, send)


app.add_middleware(Middleware)

# @app.middleware('http')
# async def middleware(request: Request, call_next):
#     id = session_counter.get_session_id()
#     async with log_request(logger, id=id, request=request):
#         resp = await call_next(request)
#     return resp


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('forms.html',
                                      {"request": request,
                                       "models": get_model_names()})


@app.post("/upload_text/")
async def upload_text(x_request_id: Annotated[UUID, Header()], 
                      model: ModelDep, 
                      transformer: TransformDep,
                      text: Annotated[str, Form(max_length=5000)]):
    
    return predict_text(id=x_request_id, 
                        model=model,
                        transformer=transformer,
                        text=text)