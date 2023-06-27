FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-venv
ENV PROJECT_PATH="/ta_project"
ENV VENV_PATH=$PROJECT_PATH/venv \
    MODEL_PATH=$PROJECT_PATH/text_authorship/ta_model \
    APP_PATH=$PROJECT_PATH/app
RUN mkdir -p $APP_PATH && mkdir -p $MODEL_PATH && touch $PROJECT_PATH/logerrors.log
COPY text_authorship/ta_model/ $MODEL_PATH
COPY app/ $APP_PATH
COPY bert_pretrained/ $PROJECT_PATH/bert_pretrained
COPY main.py tastack_deploy.pkl tatransformer.pkl logreg.pkl settings.yml $PROJECT_PATH/
RUN useradd -ms /bin/bash modelserver &&  \
    chown -R modelserver $PROJECT_PATH
USER modelserver
RUN python3.10 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r $MODEL_PATH/requirements.txt
RUN python3 -m pip install --no-cache-dir fastapi[all] prometheus_fastapi_instrumentator plotly -U
WORKDIR $PROJECT_PATH
CMD uvicorn --host 0.0.0.0 --port 8898 main:app
