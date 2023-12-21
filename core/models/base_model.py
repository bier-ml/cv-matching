from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseMatchingModel(ABC):
    @abstractmethod
    def train(self, dataset: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def predict(self, vacancy: str | np.ndarray, cv: str | np.ndarray) -> float:
        pass
