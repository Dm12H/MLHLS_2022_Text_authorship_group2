# Проект по определению авторства текста на основе его стиля

## участники:
- [Владимир Голод](https://github.com/Vigolod)
- [Максим Демидов](https://github.com/Dm12H)
- [Никита Праздников](https://github.com/kuchen1911)

## Данные 
Вручную собранный датасет текстов русскоязычных писателей,
находящихся в открытом доступе. Начальный размер выборки - собрание сочинений 15 авторов XVIII-XX века.

## [План работ](checkpoint_1/README.md)
## [Описание данных](checkpoint_2/README.md)
## [Разведочный анализ](checkpoint_3/README.md)

## Структура проекта

- В папке `experiments` находятся ноутбуки с актуальными ноутбуками
- папка `text_authorship/ta_model` содержит основную структуру проекта:
    * `base_models` содержит функции для обучения актуальных моделей
    * `data_extraction` подготавливает датасет из сырых документов и обрабатывает загрузку-выгрузку
    * `data_preparation` собирает все признаки, использующиеся в моделях
    * `model_selection` содержит функции для кроссвалидации и анализа результатов
    * `stacking` содержит инструменты для энсэмблинга
    * `logreg` содержит инструменты для логистической регрессии

## Структура сервиса

- Файл `main.py` содержит основной файл для запуска сервиса на `fastapi`
- папка `app` содержит реализации основных функций сервиса, а также логирование:
    * папка `app_models` содержит в себе модули `inference` и `model_manager`, отвечающие за выгрузку и применение моделей
    * папка `forms` содержит необходимые HTML-шаблоны и CSS файлы
    * папка `utils` содержит визуализацию и другие мелкие инструменты
    * модуль `config` отвечает за выгрузку конфигураций в настройки сервера
    * модуль `logs` отвечает за логирование
    * модуль `session_id` содержит потоко-безопасный генератор уникальных id (в данный момент не используется сервисом)
- конфигурационный файл `settings.yml` содержит конфигурацию сервиса:
    * параметры `transformer_path` и `model_paths` содержат пути к сериализованным трансформеру и моделям, соответственно
    * `log_config` содержит настройки для логирования

## Запуск сервиса

При запуске сервиса на локальной машине (не из docker образа) нужно загрузить [архив](https://drive.google.com/drive/folders/1w05x8hz_RO8Pn_oDCySCi0soXTbj9nm2?usp=sharing) и распаковать лежащие внутри `.pkl` файлы в корневую директорию проекта\
установка зависимостей:\
`pip install -r requirements.txt`\
запуск сервиса:\
`uvicorn main:app` (при желании можно указать конкретный порт)\
после появления в консоли лога вида `server started after <time-spent>` можно открыть сервис в браузере (по умолчанию `localhost:8000`)

### Функционал

Сервис по введенному тексту определеят автора (на выбор предлагаются две модели), а также выводит barplot, показывающий, на стиль каких авторов по мнению модели больше всего похож введенный текст (список поддерживаемых авторов можно найти в папке `checkpoint_2`).

### Примечание

Чтобы посмотреть на то, как логи выводят ошибки, можно попробовать испортить файл `settings.yml` (например, добавить в списке моделей еще одну с несуществующим файлом `nothing_here.pkl`). Логи ошибок появятся в файле `logerrors.log`. Конфигурацию логирования при этом лучше не менять.

## Загрузка и запуск проекта из образа
Cобранный docker-образ расположен [здесь](https://hub.docker.com/repository/docker/authorshipauth/text_authorship_group2/general) \
Для запуска использовать команду:
```
docker pull authorshipauth/text_authorship_group2:v1
docker run --rm --name <container_name> -p <local-port>:8898 -it authorshipauth/text_authorship_group2:v1
```

## Структура Dockerfile
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