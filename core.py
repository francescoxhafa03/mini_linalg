class Vector:
    def __init__(self, data: list[float]):
        self.data = list(data)

    def __repr__(self) -> str:
        return f"Vector({self.data})"


class Matrix:
    def __init__(self, data: list[list[float]]):
        self.data = [list(r) for r in data]

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def mat_vec(self, v: Vector) -> Vector:
        if not self.data or len(self.data[0]) != len(v.data):
            raise ValueError("Incompatible dimensions")

        result = [
            sum(a * b for a, b in zip(row, v.data))
            for row in self.data
        ]
        return Vector(result)

    def mat_mat(self, B: "Matrix") -> "Matrix":
        if not self.data or not B.data:
            raise ValueError("Matrices cannot be empty")

        if len(self.data[0]) != len(B.data):
            raise ValueError("Incompatible dimensions")

        columns = list(zip(*B.data))
        result = [
            [
                sum(a * b for a, b in zip(row, col))
                for col in columns
            ]
            for row in self.data
        ]
        return Matrix(result)

    def batch_mat_vec(self, vectors: list[Vector]) -> list[Vector]:
        return [self.mat_vec(v) for v in vectors]

    def batch_mat_mat(self, matrices: list["Matrix"]) -> list["Matrix"]:
        return [self.mat_mat(B) for B in matrices]