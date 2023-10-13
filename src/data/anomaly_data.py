from typing import *

import hkkang_utils.data as data_utils
import numpy as np


@data_utils.dataclass
class AnomalyData:
    cause: str  # the name of each performance anomaly
    attributes: List[str]  # list of attribute names
    values: List[List[float]]  # shape: (time, attribute)
    normal_regions: List[int]  # list of normal region indices
    abnormal_regions: List[int]  # list of abnormal region indices

    @property
    def values_as_np(self) -> np.ndarray:
        return np.array(self.values)

    @property
    def valid_normal_regions(self) -> List[int]:
        """Get all region size"""
        if self.normal_regions:
            return self.normal_regions
        return [
            i for i in range(len(self.attributes)) if i not in self.abnormal_regions
        ]

    @property
    def training_data(self) -> np.ndarray:
        """Get training data"""
        valid_regions = self.valid_normal_regions + self.abnormal_regions
        training_indices = [i for i in range(len(self.values)) if i in valid_regions]
        return self.values_as_np[training_indices:]
