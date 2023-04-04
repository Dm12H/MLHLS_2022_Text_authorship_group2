from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="temp")

app.mount("/stat", StaticFiles(directory="stat"), name="stat")

@app.get("/")
async def root(request: Request, message='Insert Text'):
    return templates.TemplateResponse('forms.html',
                                      {"request": request,
                                      "message": message})


@app.post("/upload_text")
async def upload_text(request: Request, name: str = Form(...)):

    return {'result': 'load!'}


