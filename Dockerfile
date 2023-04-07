FROM python:3.10-buster
ENV PROJECT_PATH="/ta_project"
ENV VENV_PATH=$PROJECT_PATH/venv \
    MODEL_PATH=$PROJECT_PATH/text_authorship/ta_model \
    APP_PATH=$PROJECT_PATH/app
RUN mkdir -p $APP_PATH && mkdir -p $MODEL_PATH
COPY text_authorship/ta_model/ $MODEL_PATH
COPY app/ $APP_PATH
COPY main.py tastack_deploy.pkl tatransformer.pkl logreg.pkl settings.yml $PROJECT_PATH/
RUN useradd -ms /bin/bash modelserver &&  \
    chown -R modelserver $PROJECT_PATH
USER modelserver
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r $MODEL_PATH/requirements.txt && \
    python3 -m pip install --no-cache-dir fastapi[all] -U
WORKDIR $PROJECT_PATH
CMD uvicorn --host 0.0.0.0 --port 8898 main:app
