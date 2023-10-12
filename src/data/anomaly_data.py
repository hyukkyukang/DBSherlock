from typing import *

import hkkang_utils.data as data_utils
import numpy as np


@data_utils.dataclass
class AnomalyData:
    cause: str  # the name of each performance anomaly
    attributes: List[str] # list of attribute names
    values: List[List[float]]  # shape: (time, attribute)
    normal_regions: List[int] # list of normal region indices
    abnormal_regions: List[int] # list of abnormal region indices

    @property
    def values_as_np(self) -> np.ndarray:
        return np.array(self.values)
