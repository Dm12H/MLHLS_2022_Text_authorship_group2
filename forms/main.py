from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
app = FastAPI()

templates = Jinja2Templates(directory="temp")

app.mount("/stat", StaticFiles(directory="stat"), name="stat")

@app.get("/")
async def root(request: Request, message='Insert Text'):
    return templates.TemplateResponse('forms.html',
                                      {"request": request,
                                      "message": message})
@app.get("/model") # по-моему просто просмотр файлов
async def get_model(request: Request):
    model = ["a","b"]
    # model = os.listdir('stat/model')
    return templates.TemplateResponse('model.html',
                                      {"request": request,
                                      "model": model})
@app.post("/upload_text")
async def upload_text(request: Request, name: str = Form(...)):
    model = ["a","b"]
    return templates.TemplateResponse('model.html',
                                      {"request": request,"model": model})
    #templates.TemplateResponse('forms.html',
                                      #{"request": request,})



    #{'result': 'load!'}

