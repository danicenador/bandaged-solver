# rubik

A 3x3x3 and 4x4x4 Rubik's cube engine written in Rust, with **bandaged cube** support and a search
engine for short algorithms: edge/corner/wing/center **3-cycles**, **2-edge flips** and **2-corner
twists**.
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

c4 = rubik.Cube(size=4, bandage="UFR+UFr+URf+Ufr | Dfr+Dfl")   # 4x4x4, same API
s4 = rubik.Solver(size=4, bandage="UFR+UFr+URf+Ufr")           # moves="stm", depth=4 by default
s4.find("edge3", max_results=5)                                # wing 3-cycles
s4.find("center3", max_results=5)
```

| `kind`         | effect                                                        |
|----------------|---------------------------------------------------------------|
| `edge3`        | 3-cycle of edges (wings on the 4x4x4), nothing else changes   |
| `corner3`      | 3-cycle of corners                                            |
| `edge_flip`    | two edges flipped in place (3x3x3 only)                       |
| `corner_twist` | two corners twisted in place                                  |
| `center3`      | 3-cycle of centers (4x4x4 only)                               |
| `orient`       | up to 2 flips and/or up to 2 twists, everything in place      |
| `any`          | anything confined to `pieces=[...]`                           |

Other options of `find`: `pieces`, `pure`, `extra_depth`, `max_length`, `max_results`,
`unique_effects`, `time_limit`, `move_blocks`.

**Moving glued blocks.** By default every glued cubie must end where it started. With
`move_blocks=True` (use `kind="any"` and list the blocks' cubies in `pieces`), a block may end on the
home of a congruent block as long as the puzzle gets its shape back, so the search finds block
swaps and block 3-cycles. Each result's `blocks` field reads like `(UBR+UR UFL+UF ULB+UB)`.
`Solver.home_shape_effects()` lists every home-shape position in the table with an optimal
algorithm (all block effects, when the table holds the whole puzzle), and `Cube.blocks_moved()`
describes a position. See `notebooks/glued_blocks.ipynb`. Effects are written as cycles `(UF UR UB)` (the piece at UF goes
to UR...), `*` marks a flipped edge, `+`/`-` a corner twisted clockwise/counter-clockwise.

Move notation (3x3x3): `R U F L D B`, slices `M E S`, wide `r`/`Rw`, rotations `x y z`, suffixes `2` and
`'`, groups `(R U)3`, commutators `[R, U]`, conjugates `[F: R U R']`. Search move sets:
`moves="htm"` (default, 18 face turns), `"stm"` (+ slices), `"wide"`, or explicit lists like
`"U M"` / `"R U F2"`.

### 4x4x4

`Cube(size=4)` and `Solver(size=4)` work like the 3x3x3 ones; `analyze`, `simplify_alg`,
`invert_alg`, `alg_length` and `cubie_names` take `size=4` too.

* **Cubie names** have one letter per axis, uppercase for an outer layer and lowercase for an inner
  one, in any order: corners `UFR`, wings `UFr` (on the UF edge, next to R) and centers `Ufr` (on U,
  in the quadrant towards F and R). Any set of cubies can be glued, e.g. `"UFR+UFr+URf+Ufr"` (a 2x2x1
  corner block) or `"Ufr+Ufl+Ubr+Ubl"` (a glued U center).
* **Notation**: outer `R`, inner slice `2R`, both right layers `r` = `Rw`, rotation `x` (same for
  every face). Search move sets: `"stm"` (default: every outer and inner layer, 36 moves), `"htm"`
  (outer faces), `"inner"`, `"wide"`, or a list such as `"R 2R U"`.
* **Identical centers**: the four centers of a colour look the same, so unless they are glued they
  are compared by colour only. Effects and the search see a state through that lens: the OLL parity
  `2R2 B2 U2 2L U2 2R' U2 2R U2 F2 2R F2 2L' B2 2R2` reads as a single wing swap. Glued centers keep
  their identity, since the block shows it.
* Wings cannot be flipped in place, so `edge_flip` does not apply. The difficulty study and human
  index below are 3x3x3 only.

## How it works

* **Model** (`src/geometry.rs`, `src/cube.rs`): each of the 26 cubies stores one of the 24 cube
  rotations, which gives both its position and orientation. Layer turns are table lookups.
  Center spin is ignored, except for bandaged centers. The 4x4x4 (`src/cube4.rs`) uses the same
  scheme for its 56 cubies on doubled coordinates (-3, -1, 1, 3); its packed key relabels free
  centers by colour (`Bandage4::canonical`).
* **Bandaging**: a bandage is a set of blocks (bitmasks of cubies). A turn is legal when no block
  has cubies both inside and outside the turning layers. Blocks are attached to cubies, so they
  move with the puzzle.
* **Search** (`src/search.rs`, generic over the `Puzzle` trait, implemented by `Bandage` and
  `Bandage4`): a breadth-first table holds every distinct state within `depth`
  legal moves of solved. If two sequences A and C reach states that agree on every piece outside
  the target set, then `A C'` only affects the target pieces. Grouping the table by a masked key
  finds all such pairs (algorithms up to `2*depth` moves). `extra_depth` extends the C side by
  streaming deeper sequences without storing them. Every candidate is replayed on the bandaged
  cube. It is kept only if every move is legal, the bandage shape is restored and the effect
  matches the query.

Costs: about 40 bytes per stored state (about 60 on the 4x4x4). The unbandaged cube at depth 6 is
8.2M states (~2 s to build). On bandaged cubes the branching factor is lower, so the table reaches
much deeper. The unbandaged 4x4x4 at depth 4 in `stm` holds 838k states, enough for the
8-move wing, center and corner 3-cycles.

## Difficulty study of bandaged cubes

`src/study.rs` measures how hard a bandaged cube is, using the 18 face turns, and
`src/bin/bandage_study.rs` runs it on N random bandaged cubes plus the plain 3x3x3 as a baseline:

```bash
cargo run --release --bin bandage_study -- --n 100 --seed 2026 --alg-time 10 --out study
```

It writes `study/results.csv` and `study/results.json`. A random bandage glues 1 to 12 random
pairs of touching cubies (corner-edge or edge-center); degenerate puzzles where only one face
ever turns are skipped. From Python: `rubik.difficulty(bandage="...")` and
`rubik.random_bandage(bonds, seed)`.

| KPI | how |
|---|---|
| `states`: reachable states | exact for every puzzle: `shapes * group_order` (below) |
| `gods_number`: worst-case optimal solution | exact when the whole puzzle fits in the table (`complete`) |
| `gods_lower`, `gods_counting` | proven lower bound from samples; counting-argument estimate |
| `mean_distance`: optimal length of a random state | exact if complete, else sampled (a lower bound when some samples are too deep) |
| `branching`: legal face turns in a random state | 18 on the 3x3x3 |
| `edge3`, `corner3`, `edge_flip`, `corner_twist` | shortest algorithm (HTM) that does it and restores the bandage, `possible` (how many distinct such effects exist) and `coverage` (share of the free pieces that can be involved) |

How it is computed:

* **Exact state count** (`src/group.rs`). Legality depends only on where the bandaged cubies
  are (the *shape*), and every shape is reached by the same number of states, so
  `#states = #shapes * |G0|`, where G0 is the group of the loops through the solved shape. The
  shapes are enumerated breadth first. Closing each non-tree edge of the shape graph gives
  generators of G0, and Schreier-Sims gives its order. On the 3x3x3 this returns
  43,252,003,274,489,856,000. Membership in G0 also tells exactly which 3-cycles, flips and
  twists are possible at all.
* **Distances.** The breadth-first table (10M states by default) holds every state within `d`
  moves. Small puzzles fit entirely, which gives exact God's number and distance distribution.
  For larger ones, uniform random states are drawn with a lazy random walk (propose one of the
  18 turns, stay put if blocked: the proposal is symmetric, so the walk is uniform). A
  breadth-first ball is grown around each sample until it touches the table. The first contact
  at ball depth `k` proves the distance is exactly `k + d`.
* **Algorithms.** On complete puzzles every state is scanned, so lengths are optimal. Otherwise
  the meet-in-the-middle search is used, and a length is marked `exact` when it lies within the
  exhaustively searched range (`searched_to`).
* **Baseline.** The 3x3x3 is too deep for the sampler, so its God's number (20) and mean
  distance (about 17.7) are the published values (Rokicki, Kociemba, Davidson, Dethridge, 2010).
  Its state count and algorithm KPIs are computed like the other puzzles.

### Human difficulty index

`src/human.rs` (`rubik.human_difficulty(bandage)`) estimates the moves of a *commutator solve*
rather than an optimal one: navigation back to the home shape, plus, for every class of pieces
(free corners, free edges, families of congruent blocks), the tool applications a random position
needs times their cost, plus a discovery cost (×5) for every tool that no standard 3x3x3 algorithm
provides in any grip, measured as |X| of the shortest commutator `[X, Y]`. The index is the total
divided by the 3x3x3's. `notebooks/human_index.ipynb` applies it to 20 puzzles.

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
