from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated
from text_authorship.ta_model.stacking import TASTack2Deploy
from text_authorship.ta_model.data_preparation import TATransformer
import pickle
import pandas as pd

model: TASTack2Deploy = pickle.load(open('tastack_deploy.pkl', 'rb'))
transformer: TATransformer = pickle.load(open('tatransformer.pkl', 'rb'))

app = FastAPI()

templates = Jinja2Templates(directory="forms/temp")

app.mount("/stat", StaticFiles(directory="forms/stat"), name="stat")

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


