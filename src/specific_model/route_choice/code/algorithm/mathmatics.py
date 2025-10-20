import numpy as np


__all__ = ["heron", "heron_vertex"]


def heron(a: float, b: float, c: float) -> float:
    """
    Calculate the area of a triangle using Heron's formula.

    Args:
        a (float): Length of side a.
        b (float): Length of side b.
        c (float): Length of side c.

    Returns:
        float: Area of the triangle.
    """
    # a, b, c: length of triangle
    s = (a + b + c) / 2
    if s * (s - a) * (s - b) * (s - c) >= 0:
        return np.sqrt(s * (s - a) * (s - b) * (s - c))
    else:
        print("Invalid value for Heron's formula.", a, b, c)
        return 0
    

def heron_vertex(v1: tuple, v2: tuple, v3: tuple) -> float:
    """
    Calculate the area of a triangle given its vertices.

    Args:
        v1 (tuple): Vertex 1 (x, y).
        v2 (tuple): Vertex 2 (x, y).
        v3 (tuple): Vertex 3 (x, y).

    Returns:
        float: Area of the triangle.
    """
    a = np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    b = np.sqrt((v2[0] - v3[0])**2 + (v2[1] - v3[1])**2)
    c = np.sqrt((v3[0] - v1[0])**2 + (v3[1] - v1[1])**2)
    return heron(a, b, c)
