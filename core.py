import numpy as np

class Vector:
    def __init__(self,data):
        self.data = np.array(data, dtype=float)
    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"
    
class Matrix:
    def __init__(self,data):
        self.data = np.asarray(data, dtype=float)
    def __repr__(self) -> str:
        return f"Matrix({self.data.tolist()})"