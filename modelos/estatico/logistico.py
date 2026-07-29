import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from settings.settings import settings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from utils.file_helpers import load_csv_to_df, save_baseline_model, load_baseline_model, save_dict_to_json


class ModelPipelineManager:
    def __init__(self, features_path: Path, model_path: Path, metrics_path: Path) -> None:
        assert features_path.exists()

        self.features_path = features_path
        self.model_path = model_path
        self.metrics_path = metrics_path

    def _load_features(self) -> pd.DataFrame:
        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} started - path={self.features_path}')

        df: pd.DataFrame = load_csv_to_df(self.features_path)
        assert not df.empty

        df[settings.FEATURE_COLUMNS] = df[settings.FEATURE_COLUMNS].fillna(0.0)

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - rows={len(df)} columns={df.columns}')

        return df

    def _build_pipeline(self) -> Pipeline:
        pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])

        assert 'scaler' in pipeline.named_steps
        assert 'clf' in pipeline.named_steps

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - steps={pipeline.named_steps}')

        return pipeline

    def _temporal_split(self, df: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert 'month' in df.columns and 'customer_id' in df.columns
        assert test_months > 0

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} started - test_months={test_months}')
    
        max_month: int = int(df['month'].max())
        cutoff: int = max_month - test_months
    
        train: pd.DataFrame = df[df['month'] <= cutoff].copy()
        test: pd.DataFrame = df[df['month'] > cutoff].copy()
    
        assert len(train) > 0 and len(test) > 0

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - train_rows={len(train)} test_rows={len(test)}')
    
        return train, test

    @classmethod
    def _calculate_metrics(cls, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
        assert len(y_true) == len(y_prob)
        assert len(y_true) > 0

        logging.debug(f'{cls.__class__.__name__}.{sys._getframe().f_code.co_name} started')

        auc: float = float(roc_auc_score(y_true, y_prob))
        gini: float = 2.0 * auc - 1.0

        fpr: np.ndarray
        tpr: np.ndarray
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        ks: float = np.max(np.abs(tpr - fpr))

        metrics: dict[str, float] = {
            'auc': auc,
            'gini': gini,
            'ks': ks
        }

        assert 0.0 <= metrics['auc'] <= 1.0

        logging.debug(f'{cls.__class__.__name__}.{sys._getframe().f_code.co_name} completed - auc={auc}, gini={gini} ks={ks}')
        
        return metrics

    def train(self) -> tuple[Pipeline, dict[str, float]]:
        assert self.features_path.exists()

        logging.info(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} started')

        df = self._load_features()
        train_df, test_df = self._temporal_split(df, settings.TEST_MONTHS)

        X_train: pd.DataFrame = train_df[settings.FEATURE_COLUMNS]
        y_train: np.ndarray = train_df[settings.TARGET_COLUMN].values
        X_test: pd.DataFrame = test_df[settings.FEATURE_COLUMNS]
        y_test: np.ndarray = test_df[settings.TARGET_COLUMN].values

        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)

        y_prob: np.ndarray = pipeline.predict_proba(X_test)[:, 1]
        metrics = self._calculate_metrics(y_test, y_prob)

        logging.info(f"{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - auc={metrics.get('auc', -1)}")

        return pipeline, metrics

    def save(self, model: Pipeline, metrics: dict[str, float]) -> None:
        save_baseline_model(model, self.model_path)
        save_dict_to_json(metrics, self.metrics_path)

        logging.info(f"{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - model_path={self.model_path} metrics_path={self.metrics_path}")

    def load(self) -> Pipeline:
        model = load_baseline_model(self.model_path)
        logging.info(f"{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - model_path={self.model_path}")
        return model
    
    def train_to_save(self) -> Pipeline:
        model, metrics = self.train()
        self.save(model, metrics)

        return model


def initialize(force_retrain: bool = False):
    assert isinstance(force_retrain, bool)

    try:
        manager: ModelPipelineManager = ModelPipelineManager(settings.FEATURES_PATH, settings.LOGISTICS_MODEL_PATH, settings.LOGISTICS_MODEL_METRICS_PATH)

        if force_retrain or not settings.LOGISTICS_MODEL_PATH.exists():
            logging.info(f"{sys._getframe().f_code.co_name} started - model not found or force_retrain=True training new model")
            return manager.train_to_save()

        logging.info(f"{sys._getframe().f_code.co_name} started - loading existing model from disk")

        return manager.load()

    except FileNotFoundError as e:
        logging.error(f"initialize failed - missing dependency: {e}")
        raise
    
    except Exception as e:
        logging.error(f"initialize failed - reason={e}")
        raise