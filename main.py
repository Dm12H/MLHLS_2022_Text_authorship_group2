from fastapi import FastAPI, Request, Form, Header, Depends, UploadFile, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.exceptions import HTTPException
from starlette.types import Scope, Receive, Send

from app.logs import log_request, log_server_startup, set_logs
from app.config import get_model_names, check_trainable
from app.app_models.model_manager import ModelDep, TransformDep, ModelHolder
from app.app_models.inference import predict_text, select_best_pred
from app.app_models.retrain_model import retrain_model
from app.utils.visualization import draw_barplot
from app.monitoring import create_instrumentator, record_metric

import logging.config
from typing import Annotated
from contextlib import asynccontextmanager
from uuid import uuid4, UUID
import uvicorn

from prepare_dataset import prepare_dataset
from train_model import train_model


set_logs()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with log_server_startup(logger):
        ModelHolder.load_from_settings()
    yield


app = FastAPI(lifespan=lifespan)

int = create_instrumentator()
int.instrument(app).expose(app)

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


@app.post("/upload_text/", response_class=HTMLResponse)
async def upload_text(request: Request,
                      x_request_id: Annotated[UUID, Header()],
                      model: ModelDep, 
                      transformer: TransformDep):

    form = await request.form()
    fields = dict(form)
    texts = [fields[t] for t in list(form)[:-1]]
    probabilities = predict_text(id=x_request_id,
                                 model=model,
                                 transformer=transformer,
                                 text=texts)
    figs = [draw_barplot(probs)
            for _, probs
            in probabilities.iterrows()]
    author_names = [select_best_pred(probs)
                    for _, probs
                    in probabilities.iterrows()]
    return templates.TemplateResponse("prediction.html",
                                      {"request": request,
                                       "author_names": author_names,
                                       "barplots": figs})


@app.post("/check_retrainable/")
async def upload_dataset(request: Request, model: Annotated[str, Form()]):
    if check_trainable(model):
        return templates.TemplateResponse("model_load.html",
                                          {"request": request,
                                           "model": model})
    else:
        response = FileResponse("app/forms/stat/model_select_err.html")
        return response


@app.post("/retrain/")
async def retrain(x_request_id: Annotated[UUID, Header()],
                        model: Annotated[str, Form()],
                        archive: UploadFile):
    try:
        retrain_model(x_request_id, model, archive)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="error occured while reading the archive")
    finally:
        await archive.close()
    return {"result": "ok"}


@app.post("/record_answer/")
async def record_answer(answer=Form(media_type="multipart/form-data"),
                        metric=Depends(record_metric())):
    metric.labels(answer).inc()
    return {"recorded_answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8898)
