from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


from app.logs import log_connection, log_duration
from app.config import set_logs, get_model_names
from app.app_models.model_manager import ModelDep, TransformDep
from app.session_id import SessionId
from app.app_models.inference import predict_text


import logging
import logging.config
import time
from typing import Annotated


app = FastAPI()

set_logs()
logger = logging.getLogger(__name__)

session_counter = SessionId()

templates = Jinja2Templates(directory="app/forms/temp")

app.mount("/stat", StaticFiles(directory="app/forms/stat"), name="stat")


@app.middleware('http')
async def log_request(request: Request, call_next):
    id = session_counter.get_session_id()
    log_connection(logger, id=id, request=request)
    start = time.time()
    resp = await call_next(request)
    duration = (time.time() - start) * 1000
    log_duration(logger, id=id, duration=duration)
    return resp


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('forms.html',
                                      {"request": request,
                                       "models": get_model_names()})


@app.post("/upload_text/")
async def upload_text(model: ModelDep, 
                      transformer: TransformDep,
                      text: Annotated[str, Form(max_length=5000)]):
    
    return predict_text(model=model,
                        transformer=transformer,
                        text=text)