
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

    # Matrix-Vector multiplication
    def mat_vec(self, v: Vector) -> Vector:

        if not self.data:
            raise ValueError("Matrix cannot be empty")

        if any(len(row) != len(self.data[0]) for row in self.data):
            raise ValueError("Matrix must be rectangular")

        if len(self.data[0]) != len(v.data):
            raise ValueError("Incompatible dimensions")

        result = []

        for row in self.data:
            total = 0

            for i in range(len(row)):
                total += row[i] * v.data[i]

            result.append(total)

        return Vector(result)

    # Matrix-Matrix multiplication
    def mat_mat(self, B: "Matrix") -> "Matrix":

        if not self.data or not B.data:
            raise ValueError("Matrices cannot be empty")

        if any(len(row) != len(self.data[0]) for row in self.data):
            raise ValueError("Matrix A must be rectangular")

        if any(len(row) != len(B.data[0]) for row in B.data):
            raise ValueError("Matrix B must be rectangular")

        if len(self.data[0]) != len(B.data):
            raise ValueError("Incompatible dimensions")

        result = []

        for i in range(len(self.data)):
            row = []

            for j in range(len(B.data[0])):
                total = 0

                for k in range(len(B.data)):
                    total += self.data[i][k] * B.data[k][j]

                row.append(total)

            result.append(row)

        return Matrix(result)

    # Batch Matrix-Vector multiplication
    def batch_mat_vec(self, vectors: list[Vector]) -> list[Vector]:
        return [self.mat_vec(v) for v in vectors]

    # Batch Matrix-Matrix multiplication
    def batch_mat_mat(self, matrices: list["Matrix"]) -> list["Matrix"]:
        return [self.mat_mat(B) for B in matrices]
