import numpy as np

M1 = np.array([[2, -7, 8, -4],
               [0, -1, 4, -1],
               [3, -4, 2, -1],
               [-9, 1, -4, 6]])
B1 = np.array([57, 24, 28, 12])

M2 = np.array([[-22, -2, -6, 6],
               [3, -17, -3, 7],
               [2, 6, -17, 5],
               [-1, -8, 8, 23]])
B2 = np.array([96, -26, 35, -234])

eps = 0.01


def gauss_jordan(A, B):
    n = len(A)
    A = A.astype(float)
    B = B.astype(float)

    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            B[j] -= factor * B[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (B[i] - np.sum(A[i, i + 1:] * x[i + 1:])) / A[i, i]

    return x


def seidel(A, B, eps):
    n = len(A)
    x = np.zeros(n)
    while True:
        x_new = np.zeros(n)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, 4))
            x_new[i] = (B[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
    return x


print("Результат за методом Гаусса-Жордана: ", gauss_jordan(M1, B1))
print("Результат за методом Зейделя: ", seidel(M2, B2, eps))
