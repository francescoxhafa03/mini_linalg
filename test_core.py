
from mini_linalg.core import Matrix, Vector


def test_mat_vec():
    A = Matrix([[1, 2], [3, 4]])
    x = Vector([5, 6])

    result = A.mat_vec(x)

    assert result.data == [17, 39]


def test_mat_mat():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    result = A.mat_mat(B)

    assert result.data == [[19, 22], [43, 50]]

