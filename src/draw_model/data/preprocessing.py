import numpy as np
import os
import math
class Preprocessing:
    def get_data(file_name: str) -> tuple:
        data = np.load(file_name)
        return data