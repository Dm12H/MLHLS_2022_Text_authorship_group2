from tempfile import TemporaryDirectory
from py7zr import SevenZipFile
from fastapi import UploadFile
from prepare_dataset import prepare_dataset
import os
from app.config import get_model_path
from train_model import train_model
from app.app_models.model_manager import ModelHolder
import logging
from ..logs import log_retraining
from uuid import UUID
from io import BytesIO


logger = logging.getLogger(__name__)


def retrain_model(id: UUID, model: str, archive: UploadFile):
    with log_retraining(logger, id, model):
        with TemporaryDirectory() as tmp:
            with SevenZipFile(BytesIO(archive.file.read())) as zip_archive:
                zip_archive.extractall(tmp)
            prepare_dataset(tmp, tmp, symbol_lim=1500)
            df_path = os.path.join(tmp, 'prepared_df.csv')
            model_path = get_model_path(model)
            train_model(df_path, model, pkl=model_path)
            ModelHolder.force_load_model(model, model_path)