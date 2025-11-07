def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def main():
    nums = [3, 5, 7]
    results = [factorial(n) for n in nums]
    print("Factorials:", results)
