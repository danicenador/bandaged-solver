# rubik

A 3x3x3 Rubik's cube engine written in Rust, with **bandaged cube** support and a search engine
for short algorithms: edge/corner **3-cycles**, **2-edge flips** and **2-corner twists**.
The engine is exposed to Python (PyO3) so you can explore it from Jupyter notebooks.

## Setup

```bash
uv sync --all-extras
```

This creates `.venv`, installs the dev and notebook dependencies from `uv.lock` and builds the
extension (release mode). `uv sync` and `uv run` rebuild it automatically when `src/*.rs`,
`Cargo.toml` or `pyproject.toml` change. For a faster edit-compile loop you can also run
`uv run maturin develop --release --uv`. Then open `notebooks/rubik_tour.ipynb` with the
`.venv` interpreter.

On Windows with the GNU Rust toolchain, `.cargo/config.toml` and `build.rs` make the extension
link against the interpreter's `python3.lib` (pyo3's default `raw-dylib` linking needs a full
MinGW install).

## Python quick start

```python
import rubik

c = rubik.Cube(bandage="UF+UFR+UR+U | DF+D")   # two glued blocks
c.legal_moves()                                # ['L', 'L2', "L'", 'U', ...]
c.apply("U2 D2 B")                             # ValueError if the bandage blocks a move
rubik.plot_cube(c)                             # matplotlib net, glued stickers drawn merged

s = rubik.Solver(bandage="UF+UFR+UR+U | DF+D", depth=40, max_nodes=4_000_000)
s.find("edge3", pure=True, max_results=10)     # pure edge 3-cycles, shortest first
s.find("corner_twist", pieces=["DFR", "DBL"])  # two specific corners
rubik.to_dataframe(s.find("corner3"))
```

| `kind`         | effect                                                        |
|----------------|---------------------------------------------------------------|
| `edge3`        | 3-cycle of edges, nothing else changes                        |
| `corner3`      | 3-cycle of corners                                            |
| `edge_flip`    | two edges flipped in place                                    |
| `corner_twist` | two corners twisted in place                                  |
| `orient`       | up to 2 flips and/or up to 2 twists, everything in place      |
| `any`          | anything confined to `pieces=[...]`                           |

Other options of `find`: `pieces`, `pure`, `extra_depth`, `max_length`, `max_results`,
`unique_effects`, `time_limit`. Effects are written as cycles `(UF UR UB)` (the piece at UF goes
to UR...), `*` marks a flipped edge, `+`/`-` a corner twisted clockwise/counter-clockwise.

Move notation: `R U F L D B`, slices `M E S`, wide `r`/`Rw`, rotations `x y z`, suffixes `2` and
`'`, groups `(R U)3`, commutators `[R, U]`, conjugates `[F: R U R']`. Search move sets:
`moves="htm"` (default, 18 face turns), `"stm"` (+ slices), `"wide"`, or explicit lists like
`"U M"` / `"R U F2"`.

## How it works

* **Model** (`src/geometry.rs`, `src/cube.rs`): each of the 26 cubies stores one of the 24 cube
  rotations, which gives both its position and orientation. Layer turns are table lookups.
  Center spin is ignored, except for bandaged centers.
* **Bandaging**: a bandage is a set of blocks (bitmasks of cubies). A turn is legal when no block
  has cubies both inside and outside the turning layers. Blocks are attached to cubies, so they
  move with the puzzle.
* **Search** (`src/search.rs`): a breadth-first table holds every distinct state within `depth`
  legal moves of solved. If two sequences A and C reach states that agree on every piece outside
  the target set, then `A C'` only affects the target pieces. Grouping the table by a masked key
  finds all such pairs (algorithms up to `2*depth` moves). `extra_depth` extends the C side by
  streaming deeper sequences without storing them. Every candidate is replayed on the bandaged
  cube. It is kept only if every move is legal, the bandage shape is restored and the effect
  matches the query.

Costs: about 40 bytes per stored state. The unbandaged cube at depth 6 is 8.2M states (~2 s to
build). On bandaged cubes the branching factor is lower, so the table reaches much deeper.

## Tests

```bash
cargo test --release
```
```bash
uv run pytest tests
```

## Benchmarks

`benches/engine.rs` (Criterion) times move application, table builds and searches.

```bash
cargo bench -- --save-baseline before
```
```bash
cargo bench -- --baseline before
```

Run the first command before a change and the second after it to compare. Timings on a laptop
can vary by 2-3x between runs, so repeat a run before trusting a difference.

On the GNU toolchain, `Cargo.lock` pins `winapi-util` to 0.1.9: newer versions pull in
`windows-sys` 0.61, which needs MinGW's `dlltool` to link. A plain `cargo update` undoes the pin.
To restore it, run `cargo update -p winapi-util --precise 0.1.9`.
