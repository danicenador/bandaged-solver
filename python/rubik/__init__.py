"""3x3x3 Rubik's cube engine with bandaging support and an algorithm search engine.

The heavy lifting is done in Rust (``rubik._rubik``); this package adds plotting and
convenience helpers for notebooks.
"""

from ._rubik import Cube, Solver, alg_length, analyze, cubie_names, invert_alg, simplify_alg
from .plot import plot_cube, plot_cubes

CORNERS = cubie_names()[:8]
EDGES = cubie_names()[8:20]
CENTERS = cubie_names()[20:]

__all__ = [
    "Cube",
    "Solver",
    "alg_length",
    "analyze",
    "cubie_names",
    "invert_alg",
    "simplify_alg",
    "plot_cube",
    "plot_cubes",
    "to_dataframe",
    "CORNERS",
    "EDGES",
    "CENTERS",
]


def to_dataframe(results):
    """Turn the list returned by ``Solver.find`` into a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(results, columns=["alg", "length", "qtm", "effect", "pieces", "pure", "kind"])
