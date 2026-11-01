"""Matplotlib drawing of a cube as an unfolded net, with bandaged blocks shown merged."""

COLORS = {
    "U": "#ffffff",
    "R": "#c41e3a",
    "F": "#009e60",
    "D": "#ffd500",
    "L": "#ff5800",
    "B": "#0051ba",
}

# Top-left corner (column, row) of each face in the net, in face widths, in facelet order URFDLB.
_OFFSETS = {"U": (1, 0), "R": (2, 1), "F": (1, 1), "D": (1, 2), "L": (0, 1), "B": (3, 1)}
_FACES = "URFDLB"
_GAP = 0.08


def _facelet_xy(i, n=3):
    face = _FACES[i // (n * n)]
    r, c = divmod(i % (n * n), n)
    ox, oy = _OFFSETS[face]
    return n * ox + c, -(n * oy + r + 1)  # lower-left corner of the cell, y grows upwards


def plot_cube(cube, ax=None, title=None, highlight=None, size=3.0):
    """Draw ``cube`` (3x3x3 or 4x4x4) as a net.

    Args:
        cube: a ``rubik.Cube``.
        ax: matplotlib axes (a new figure is created if omitted).
        title: optional title.
        highlight: cubie names to mark with a dot (wherever those pieces currently are),
            e.g. ``analyze(alg)["affected"]``.
        size: figure height in inches when a new figure is created.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(size * 4 / 3, size))
    n = cube.size
    facelets = cube.facelets()
    owners = cube.facelet_cubies()
    marked = set(highlight or [])
    s = 1 - 2 * _GAP

    # dark cube body behind each face, so that glued (bridged) stickers are visible
    for ox, oy in _OFFSETS.values():
        ax.add_patch(Rectangle((n * ox, -n * (oy + 1)), n, n, facecolor="#1c1c1c", edgecolor="none", zorder=0))
    for i, col in enumerate(facelets):
        x, y = _facelet_xy(i, n)
        ax.add_patch(Rectangle((x + _GAP, y + _GAP), s, s, facecolor=COLORS[col], edgecolor="none", zorder=2))
        if owners[i] in marked:
            ax.add_patch(Circle((x + 0.5, y + 0.5), 0.17, facecolor="#222", edgecolor="#fff", lw=0.6, zorder=4))

    # bandaged neighbours: fill the gap between the two stickers
    pairs = cube.bonded_facelet_pairs()
    bonded = set(pairs)
    for i in range(6 * n * n):  # 2x2 glued square: also fill the gap crossing in its middle
        r, c = divmod(i % (n * n), n)
        if r < n - 1 and c < n - 1 and {(i, i + 1), (i + n, i + n + 1), (i, i + n), (i + 1, i + n + 1)} <= bonded:
            x, y = _facelet_xy(i, n)
            ax.add_patch(Rectangle((x + 1 - _GAP - 0.01, y - _GAP - 0.01), 2 * _GAP + 0.02, 2 * _GAP + 0.02,
                                   facecolor=COLORS[facelets[i]], edgecolor="none", zorder=3))
    for i, j in pairs:
        (x1, y1), (x2, y2) = _facelet_xy(i, n), _facelet_xy(j, n)
        color = COLORS[facelets[i]]
        if y1 == y2:  # horizontal neighbours
            x = max(x1, x2)
            ax.add_patch(Rectangle((x - _GAP - 0.01, y1 + _GAP), 2 * _GAP + 0.02, s,
                                   facecolor=color, edgecolor="none", zorder=3))
        else:
            y = max(y1, y2)
            ax.add_patch(Rectangle((x1 + _GAP, y - _GAP - 0.01), s, 2 * _GAP + 0.02,
                                   facecolor=color, edgecolor="none", zorder=3))

    ax.set_xlim(-0.2, 4 * n + 0.2)
    ax.set_ylim(-3 * n - 0.2, 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_cubes(cubes, titles=None, highlights=None, ncols=3, size=2.5):
    """Draw several cubes in a grid. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    cubes = list(cubes)
    n = len(cubes)
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size * 4 / 3, nrows * size), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k < n:
            plot_cube(cubes[k], ax=ax, title=titles[k] if titles else None,
                      highlight=highlights[k] if highlights else None)
        else:
            ax.axis("off")
    fig.tight_layout()
    return fig
