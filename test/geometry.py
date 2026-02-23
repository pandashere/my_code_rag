"""
Geometry calculations module.
Uses math_utils for calculations.
"""

from .math_utils import power, square_root, add, multiply


def calculate_circle_area(radius: float) -> float:
    """Calculate area of a circle given radius."""
    pi = 3.14159265359
    return multiply(pi, power(radius, 2))


def calculate_circle_circumference(radius: float) -> float:
    """Calculate circumference of a circle given radius."""
    pi = 3.14159265359
    return multiply(2, multiply(pi, radius))


def calculate_rectangle_area(length: float, width: float) -> float:
    """Calculate area of a rectangle."""
    return multiply(length, width)


def calculate_rectangle_perimeter(length: float, width: float) -> float:
    """Calculate perimeter of a rectangle."""
    return multiply(2, add(length, width))


def calculate_triangle_area(base: float, height: float) -> float:
    """Calculate area of a triangle given base and height."""
    return multiply(0.5, multiply(base, height))


def calculate_hypotenuse(a: float, b: float) -> float:
    """Calculate hypotenuse of a right triangle using Pythagorean theorem."""
    a_squared = power(a, 2)
    b_squared = power(b, 2)
    sum_squares = add(a_squared, b_squared)
    return square_root(sum_squares)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate distance between two points in 2D space."""
    dx = subtract(x2, x1)
    dy = subtract(y2, y1)
    return calculate_hypotenuse(dx, dy)


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def calculate_sphere_volume(radius: float) -> float:
    """Calculate volume of a sphere given radius."""
    pi = 3.14159265359
    r_cubed = power(radius, 3)
    return multiply(4/3, multiply(pi, r_cubed))


def calculate_sphere_surface_area(radius: float) -> float:
    """Calculate surface area of a sphere given radius."""
    pi = 3.14159265359
    r_squared = power(radius, 2)
    return multiply(4, multiply(pi, r_squared))


def calculate_cylinder_volume(radius: float, height: float) -> float:
    """Calculate volume of a cylinder."""
    pi = 3.14159265359
    base_area = multiply(pi, power(radius, 2))
    return multiply(base_area, height)


def calculate_cube_volume(side: float) -> float:
    """Calculate volume of a cube."""
    return power(side, 3)


def calculate_cube_surface_area(side: float) -> float:
    """Calculate surface area of a cube."""
    face_area = power(side, 2)
    return multiply(6, face_area)
