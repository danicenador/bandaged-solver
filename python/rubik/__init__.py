"""3x3x3 and 4x4x4 Rubik's cube engine with bandaging support and an algorithm search engine.

The heavy lifting is done in Rust (``rubik._rubik``); this package adds plotting and
convenience helpers for notebooks.
"""

from ._rubik import (
    Cube,
    Solver,
    alg_length,
    analyze,
    cubie_names,
    difficulty,
    human_difficulty,
    invert_alg,
    random_bandage,
    simplify_alg,
)
from .plot import plot_cube, plot_cubes

CORNERS = cubie_names()[:8]
EDGES = cubie_names()[8:20]
CENTERS = cubie_names()[20:]
# 4x4x4: uppercase letters for outer coordinates, lowercase for inner ones
WINGS4 = cubie_names(4)[8:32]
CENTERS4 = cubie_names(4)[32:]

__all__ = [
    "Cube",
    "Solver",
    "alg_length",
    "analyze",
    "cubie_names",
    "difficulty",
    "human_difficulty",
    "invert_alg",
    "random_bandage",
    "simplify_alg",
    "plot_cube",
    "plot_cubes",
    "to_dataframe",
    "CORNERS",
    "EDGES",
    "CENTERS",
    "WINGS4",
    "CENTERS4",
]


def to_dataframe(results):
    """Turn the list returned by ``Solver.find`` into a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(results, columns=["alg", "length", "qtm", "effect", "pieces", "pure", "kind"])
