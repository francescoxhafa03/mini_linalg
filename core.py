class Vector:
  def __init__(self, data: list[float]):
    self.data = list(data)
  def __repr__(self)-> str:
    return f"Vector({self.data})"
class Matrix:
  def __init__(self, data: list[list[float]]):
    self.data = [list(r) for r in data]
  def __repr__(self)-> str:
    return f"Matrix({self.data})"
