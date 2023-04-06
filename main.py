from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated
from text_authorship.ta_model.stacking import TASTack2Deploy
from text_authorship.ta_model.data_preparation import TATransformer
from service_tools.session_id import SessionId
from service_tools.logs import log_conection, log_duration

import pickle
import pandas as pd
import logging
import logging.config
import yaml
import time

model: TASTack2Deploy = pickle.load(open('tastack_deploy.pkl', 'rb'))
transformer: TATransformer = pickle.load(open('tatransformer.pkl', 'rb'))

with open('logconfig.yml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

session_counter = SessionId()

app = FastAPI()

templates = Jinja2Templates(directory="forms/temp")

app.mount("/stat", StaticFiles(directory="forms/stat"), name="stat")


@app.middleware('http')
async def log_request(request: Request, call_next):
    session_id = session_counter.get_session_id()
    log_conection(__name__, id=session_id, request=request)
    start = time.time()
    resp = await call_next(request)
    duration = (time.time() - start) * 1000
    log_duration(__name__, id=session_id, duration=duration)
    return resp


@app.get("/")
async def root(request: Request, message='Insert Text'):
    return templates.TemplateResponse('forms.html',
                                      {"request": request,
                                      "message": message})


@app.post("/upload_text/")
async def upload_text(name: Annotated[str, Form(max_length=5000)]):
    item = pd.DataFrame({'text': [name]})
    transformed_item = transformer.transform(item)
    return model.predict(transformed_item)[0]
