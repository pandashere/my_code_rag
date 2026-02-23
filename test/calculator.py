"""
Main calculator module.
Demonstrates cross-file imports and function usage.
"""

from .math_utils import (
    add, subtract, multiply, divide,
    power, square_root, average, factorial,
    fibonacci, is_prime, gcd, lcm
)
from .geometry import (
    calculate_circle_area, calculate_circle_circumference,
    calculate_rectangle_area, calculate_rectangle_perimeter,
    calculate_triangle_area, calculate_hypotenuse,
    calculate_distance, calculate_sphere_volume,
    calculate_cube_volume
)


def calculate_circle_properties(radius: float) -> dict:
    """Calculate all properties of a circle."""
    return {
        "radius": radius,
        "area": calculate_circle_area(radius),
        "circumference": calculate_circle_circumference(radius)
    }


def calculate_rectangle_properties(length: float, width: float) -> dict:
    """Calculate all properties of a rectangle."""
    return {
        "length": length,
        "width": width,
        "area": calculate_rectangle_area(length, width),
        "perimeter": calculate_rectangle_perimeter(length, width),
        "diagonal": calculate_hypotenuse(length, width)
    }


def calculate_statistics(numbers: list) -> dict:
    """Calculate various statistics for a list of numbers."""
    if not numbers:
        return {"error": "Empty list"}
    
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "average": average(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }


def number_analysis(n: int) -> dict:
    """Analyze properties of a number."""
    return {
        "number": n,
        "is_prime": is_prime(n),
        "factorial": factorial(n) if n >= 0 else "undefined",
        "fibonacci": fibonacci(n),
        "square": power(n, 2),
        "square_root": square_root(n) if n >= 0 else "undefined"
    }


def calculate_triangle_properties(base: float, height: float, 
                                   side_a: float = None, side_b: float = None) -> dict:
    """Calculate properties of a triangle."""
    result = {
        "base": base,
        "height": height,
        "area": calculate_triangle_area(base, height)
    }
    
    if side_a and side_b:
        result["side_a"] = side_a
        result["side_b"] = side_b
        result["perimeter"] = add(base, add(side_a, side_b))
    
    return result


def calculate_3d_properties(shape: str, **kwargs) -> dict:
    """Calculate properties of 3D shapes."""
    if shape == "sphere":
        radius = kwargs.get("radius", 0)
        return {
            "shape": "sphere",
            "radius": radius,
            "volume": calculate_sphere_volume(radius)
        }
    elif shape == "cube":
        side = kwargs.get("side", 0)
        return {
            "shape": "cube",
            "side": side,
            "volume": calculate_cube_volume(side)
        }
    else:
        return {"error": f"Unknown shape: {shape}"}


def distance_calculator(points: list) -> dict:
    """Calculate distances between multiple points."""
    if len(points) < 2:
        return {"error": "Need at least 2 points"}
    
    distances = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dist = calculate_distance(x1, y1, x2, y2)
        distances.append({
            "from": points[i],
            "to": points[i + 1],
            "distance": dist
        })
    
    total_distance = sum(d["distance"] for d in distances)
    
    return {
        "segments": distances,
        "total_distance": total_distance
    }


def math_operations(a: float, b: float) -> dict:
    """Perform all basic math operations on two numbers."""
    return {
        "a": a,
        "b": b,
        "addition": add(a, b),
        "subtraction": subtract(a, b),
        "multiplication": multiply(a, b),
        "division": divide(a, b) if b != 0 else "undefined",
        "a_to_power_b": power(a, b),
        "gcd": gcd(int(a), int(b)),
        "lcm": lcm(int(a), int(b))
    }
