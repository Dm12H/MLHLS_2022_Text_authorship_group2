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
    * `stacking` содержит инструменты для энсэмблинга'

## Запуск проекта
Для запуска неободимо скачать подготовленный [датасет](https://drive.google.com/drive/folders/1S7ZPEsi2yiW5C7TP-1ICO1pZJp0YUXQ9?usp=share_link)\
установка зависимостей:\
`pip install -r requirements.txt`
скрипт для запуска обучения модели\
`./main.py --dataset_dir=<path-to-downloaded-dataset> --model=<logreg|stacking>`
скрипт для подготовки датасета:\
`./prepare_dataset.py --data_dir=<path-to-raw-data-folder> --output_dir=<path-to-output-folder>`
