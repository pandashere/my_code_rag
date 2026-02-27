"""Geometry calculations using math_utils."""

from .math_utils import power, square_root, add, multiply

def calculate_circle_area(radius: float) -> float:
    pi = 3.14159
    return multiply(pi, power(radius, 2))

def calculate_rectangle_perimeter(length: float, width: float) -> float:
    return multiply(2, add(length, width))

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return square_root(add(power(dx, 2), power(dy, 2)))
