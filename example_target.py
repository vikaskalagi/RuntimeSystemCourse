# matrix_mul.py
# This is a sample matrix multiplication code on which we will run our py runtime analyser code.
def mat_mult(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def main():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = mat_mult(A, B)
    print("Result:", C)
