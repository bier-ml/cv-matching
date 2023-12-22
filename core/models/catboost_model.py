from catboost import CatBoostRegressor
from sklearn import metrics

from core.models.linear_regression import LinearRegressionModel


def brier_score(y_true, y_pred):
    return -metrics.brier_score_loss(y_true, y_pred)


class CatboostRegressionModel(LinearRegressionModel):
    def __init__(self, metric=metrics.mean_squared_error, embedding_model=None):
        super().__init__(metric, embedding_model)
        self.metric = metric
        self.model = CatBoostRegressor()
        self.embedding_model = embedding_model
