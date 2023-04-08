## Загрузка и запуск проекта
Cобранный docker-образ расположен [здесь](https://hub.docker.com/repository/docker/authorshipauth/text_authorship_group2/general) \
Для запуска использовать команду:
```
docker pull authorshipauth/text_authorship_group2:v1
docker run --rm --name <container_name> -p <local-port>:8898 -it authorshipauth/text_authorship_group2:v1
```

### Структура Dockerfile
```dockerfile
# исходный образ, на основе которого мы собираем свой
FROM python:3.10-buster
# задаем переменные среды для удобства работы
# переменные среды задаются только после создания слоя.
# поэтому для того, чтобы использовать адрес исходного каталога,
# приходится использовать две команды ENV
ENV PROJECT_PATH="/ta_project"
ENV VENV_PATH=$PROJECT_PATH/venv \
    MODEL_PATH=$PROJECT_PATH/text_authorship/ta_model \
    APP_PATH=$PROJECT_PATH/app
# Создаем директории и копируем только необходимое
# подмножество исходников проекта
RUN mkdir -p $APP_PATH && mkdir -p $MODEL_PATH && touch $PROJECT_PATH/logerrors.log
COPY text_authorship/ta_model/ $MODEL_PATH
COPY app/ $APP_PATH
COPY main.py tastack_deploy.pkl tatransformer.pkl logreg.pkl settings.yml $PROJECT_PATH/
# создаем пользователя, от имени котого будет запущен сервис.
# это позволяет разграничить права
RUN useradd -ms /bin/bash modelserver &&  \
    chown -R modelserver $PROJECT_PATH \
# дальнейшие команды исполняются от имени modelserver
USER modelserver
# создаем виртуальную среду и переключаемся на неё. у пользователя нет прав
# устанавливать пакеты в системный дистрибутив питона
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
# установка пакетов, необходимых для работы. Можно исполнить одной командой,
# сокращая число создаваемых слоев,но, учитывая размеры образа,
# экономия не существенная. Разбивка упрощает кеширование и сборку
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r $MODEL_PATH/requirements.txt
RUN python3 -m pip install --no-cache-dir fastapi[all] plotly -U
# переходим в директорию проекта и запускаем сервис
WORKDIR $PROJECT_PATH
CMD uvicorn --host 0.0.0.0 --port 8898 main:app
```

