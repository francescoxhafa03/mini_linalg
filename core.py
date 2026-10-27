
class Vector:
    def __init__(self, data):
        self.data = list(data)

    def __repr__(self):
        return f"Vector({self.data})"


class Matrix:
    def __init__(self, data):
        self.data = [list(r) for r in data]

    def __repr__(self):
        return f"Matrix({self.data})"

    # Matrix-Vector multiplication
    def mat_vec(self, x):

        if len(self.data[0]) != len(x.data):
            raise ValueError("Incompatible dimensions")

        result = []

        for row in self.data:
            total = 0

            for i in range(len(row)):
                total += row[i] * x.data[i]

            result.append(total)

        return Vector(result)

    # Matrix-Matrix multiplication
    def mat_mat(self, B):

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

