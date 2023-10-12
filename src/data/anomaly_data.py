from typing import *

import hkkang_utils.data as data_utils
import numpy as np


@data_utils.dataclass
class AnomalyData:
    cause: str  # anomaly cause
    attributes: List[str]
    values: List[List[float]]  # shape: (time, attribute)
    normal_regions: List[int]
    abnormal_regions: List[int]

    @property
    def values_as_np(self) -> np.ndarray:
        return np.array(self.values)
