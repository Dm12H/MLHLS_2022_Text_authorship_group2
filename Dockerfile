FROM python:3.10-buster
ENV PROJECT_PATH="/ta_project"
ENV VENV_PATH=$PROJECT_PATH/venv
RUN mkdir -p $PROJECT_PATH/server && mkdir $PROJECT_PATH/ta_model
COPY forms/ $PROJECT_PATH/server
COPY text_authorship/ta_model/ $PROJECT_PATH/ta_model
RUN useradd -Ms /bin/bash modelserver &&  \
    chown -R modelserver $PROJECT_PATH
USER modelserver
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r $PROJECT_PATH/ta_model/requirements.txt && \
    python3 -m pip install --no-cache-dir fastapi[all] -U
WORKDIR $PROJECT_PATH/server
CMD uvicorn --host 0.0.0.0 --port 8898 main:app
