from pathlib import Path
import sys
# try package-style import first (good for editors/linters); fall back to sys.path-based import if missing
try:
    from probes.python_probe.probe import get_runtime_info
except Exception:
    sys.path.append(str(Path(__file__).resolve().parents[2] / "probes/python_probe"))
    from probe import get_runtime_info

def matmul(A, B):
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

if __name__ == "__main__":
    N = 50
    A = [[i+j for j in range(N)] for i in range(N)]
    B = [[i*j for j in range(N)] for i in range(N)]

    get_runtime_info(matmul, A, B)
