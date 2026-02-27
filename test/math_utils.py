"""Number theory and sequence utilities."""

from .math_utils import multiply, gcd

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial undefined for negative")
    result = 1
    for i in range(2, n + 1):
        result = multiply(result, i)
    return result

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)
