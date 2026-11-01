//! Python bindings (module `rubik._rubik`, re-exported by the `rubik` package).
//!
//! `Cube` and `Solver` take a `size` (3 or 4); the difficulty study is 3x3x3 only.

use crate::cube::{Bandage, Cube, Effect, State};
use crate::cube4::{self, geo4, Bandage4, Cube4, State4};
use crate::geometry::{piece_type, PieceType, NAMES};
use crate::moves::{self, Alg, Move};
use crate::search::{Kind, Puzzle, Query, SearchStats, Solver};
use crate::study::{profile, Rng, StudyConfig, ALG_KINDS};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Accepts None, a string ("UF+UFR+UR | D DF"), a list of strings (one block each) or a list
/// of lists of cubie names, and returns it as a bandage string.
fn bandage_spec(spec: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(String::new());
    };
    if let Ok(s) = spec.extract::<String>() {
        return Ok(s);
    }
    if let Ok(groups) = spec.extract::<Vec<Vec<String>>>() {
        return Ok(groups.iter().map(|g| g.join("+")).collect::<Vec<_>>().join("|"));
    }
    if let Ok(blocks) = spec.extract::<Vec<String>>() {
        return Ok(blocks.join("|"));
    }
    Err(err("bandage must be a string, a list of strings or a list of lists of cubie names".into()))
}

/// A 3x3x3 bandage (for the 3x3x3-only functions).
fn bandage_from(spec: Option<&Bound<'_, PyAny>>) -> PyResult<Bandage> {
    Bandage::parse(&bandage_spec(spec)?).map_err(err)
}

fn check_size(size: usize) -> PyResult<()> {
    if size == 3 || size == 4 { Ok(()) } else { Err(err(format!("size must be 3 or 4, not {}", size))) }
}

/// What the bindings need from each cube size.
trait Sized3or4: Puzzle {
    const DEFAULT_MOVES: &'static str;
    const DEFAULT_DEPTH: usize;
    fn names() -> &'static [&'static str];
    fn parse_bandage(spec: &str) -> Result<Self, String>;
    fn parse_moveset(s: &str) -> Result<Vec<Move>, String>;
    fn parse_cubie(name: &str) -> Option<usize>;
    fn move_name(m: Move) -> String;
    fn effect_of(&self, s: &Self::State) -> Effect;
    fn describe_blocks(&self, s: &Self::State) -> String;
    fn n_blocks(&self) -> usize;
    /// Every unglued corner and edge (wing) is home and oriented.
    fn free_solved(&self, s: &Self::State) -> bool;

    fn parse_pieces(names: Vec<String>) -> PyResult<u64> {
        let mut m = 0u64;
        for name in names {
            m |= 1 << Self::parse_cubie(&name).ok_or_else(|| err(format!("unknown cubie '{}'", name)))?;
        }
        Ok(m)
    }

    /// State after `alg` from solved, if the bandage allows it.
    fn replay(&self, alg: &[Move]) -> Result<Self::State, String> {
        let mut s = self.solved();
        for (i, &m) in alg.iter().enumerate() {
            if !self.allows(&s, moves::base_of(m)) {
                return Err(format!("move {} ({}) is blocked by the bandage", i + 1, Self::move_name(m)));
            }
            Self::apply_move(&mut s, m);
        }
        Ok(s)
    }
}

impl Sized3or4 for Bandage {
    const DEFAULT_MOVES: &'static str = "htm";
    const DEFAULT_DEPTH: usize = 6;
    fn names() -> &'static [&'static str] {
        &NAMES
    }
    fn parse_bandage(spec: &str) -> Result<Self, String> {
        Bandage::parse(spec)
    }
    fn parse_moveset(s: &str) -> Result<Vec<Move>, String> {
        moves::parse_moveset(s)
    }
    fn parse_cubie(name: &str) -> Option<usize> {
        crate::geometry::parse_cubie(name)
    }
    fn move_name(m: Move) -> String {
        moves::move_name(m)
    }
    fn effect_of(&self, s: &State) -> Effect {
        s.effect()
    }
    fn describe_blocks(&self, s: &State) -> String {
        Bandage::describe_blocks(self, s)
    }
    fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
    fn free_solved(&self, s: &State) -> bool {
        let glued = self.bandaged_cubies();
        (0..NAMES.len())
            .filter(|&c| glued & (1 << c) == 0 && piece_type(c) != PieceType::Center)
            .all(|c| s.slot(c) == c && s.ori(c) == 0)
    }
}

impl Sized3or4 for Bandage4 {
    const DEFAULT_MOVES: &'static str = "stm";
    const DEFAULT_DEPTH: usize = 4;
    fn names() -> &'static [&'static str] {
        &geo4().names
    }
    fn parse_bandage(spec: &str) -> Result<Self, String> {
        Bandage4::parse(spec)
    }
    fn parse_moveset(s: &str) -> Result<Vec<Move>, String> {
        cube4::parse_moveset4(s)
    }
    fn parse_cubie(name: &str) -> Option<usize> {
        cube4::parse_cubie4(name)
    }
    fn move_name(m: Move) -> String {
        cube4::move_name4(m)
    }
    fn effect_of(&self, s: &State4) -> Effect {
        self.effect(s)
    }
    fn describe_blocks(&self, s: &State4) -> String {
        Bandage4::describe_blocks(self, s)
    }
    fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
    fn free_solved(&self, s: &State4) -> bool {
        let glued = self.bandaged_cubies();
        (0..cube4::CENTER0_4).filter(|&c| glued & (1 << c) == 0).all(|c| s.slot(c) == c && s.ori(c) == 0)
    }
}

fn effect_dict<'py>(py: Python<'py>, e: &Effect, names: &[&'static str]) -> PyResult<Bound<'py, PyDict>> {
    let cycle = |cyc: &Vec<(usize, u8)>| cyc.iter().map(|x| names[x.0]).collect::<Vec<_>>();
    let d = PyDict::new(py);
    d.set_item("description", e.describe_with(names))?;
    d.set_item("edge_cycles", e.edge_cycles.iter().map(cycle).collect::<Vec<_>>())?;
    d.set_item("corner_cycles", e.corner_cycles.iter().map(cycle).collect::<Vec<_>>())?;
    d.set_item(
        "center_cycles",
        e.center_cycles.iter().map(|c| c.iter().map(|&x| names[x]).collect::<Vec<_>>()).collect::<Vec<_>>(),
    )?;
    d.set_item("flipped", e.flipped.iter().map(|&c| names[c]).collect::<Vec<_>>())?;
    d.set_item(
        "twisted",
        e.twisted.iter().map(|&(c, o)| (names[c], if o == 1 { "cw" } else { "ccw" })).collect::<Vec<_>>(),
    )?;
    d.set_item("affected", e.affected().iter().map(|&c| names[c]).collect::<Vec<_>>())?;
    d.set_item("pure", e.is_pure())?;
    d.set_item("n_edges", e.n_edges())?;
    d.set_item("n_corners", e.n_corners())?;
    Ok(d)
}

#[derive(Clone)]
enum AnyCube {
    C3(Cube),
    C4(Cube4),
}

/// A 3x3x3 or 4x4x4 cube, optionally bandaged.
///
/// Args:
///     bandage: blocks of cubies that always move together, as a string ("UF+UFR+UR | D+DF"),
///         a list of strings (["UF UFR UR", "D DF"]) or a list of lists of cubie names.
///         Overlapping blocks are merged. 4x4x4 names use uppercase for outer and lowercase for
///         inner coordinates: corner "UFR", wing "UFr" (UF edge, next to R), center "Ufr".
///     alg: optional algorithm applied after construction.
///     size: 3 or 4.
#[pyclass(name = "Cube", module = "rubik")]
#[derive(Clone)]
pub struct PyCube {
    inner: AnyCube,
}

impl PyCube {
    fn parse(&self, alg: &str) -> PyResult<Alg> {
        match &self.inner {
            AnyCube::C3(_) => moves::parse_alg(alg),
            AnyCube::C4(_) => cube4::parse_alg4(alg),
        }
        .map_err(err)
    }

    fn names(&self) -> &'static [&'static str] {
        match &self.inner {
            AnyCube::C3(_) => &NAMES,
            AnyCube::C4(_) => &geo4().names,
        }
    }

    fn apply_alg(&mut self, alg: &[Move]) -> Result<(), String> {
        match &mut self.inner {
            AnyCube::C3(c) => c.apply(alg),
            AnyCube::C4(c) => c.apply(alg),
        }
    }
}

#[pymethods]
impl PyCube {
    #[new]
    #[pyo3(signature = (bandage = None, alg = None, size = 3))]
    fn new(bandage: Option<&Bound<'_, PyAny>>, alg: Option<&str>, size: usize) -> PyResult<Self> {
        check_size(size)?;
        let spec = bandage_spec(bandage)?;
        let inner = if size == 3 {
            AnyCube::C3(Cube::new(Bandage::parse(&spec).map_err(err)?))
        } else {
            AnyCube::C4(Cube4::new(Bandage4::parse(&spec).map_err(err)?))
        };
        let mut c = PyCube { inner };
        if let Some(a) = alg {
            let a = c.parse(a)?;
            c.apply_alg(&a).map_err(err)?;
        }
        Ok(c)
    }

    /// 3 or 4.
    #[getter]
    fn size(&self) -> usize {
        match &self.inner {
            AnyCube::C3(_) => 3,
            AnyCube::C4(_) => 4,
        }
    }

    /// Apply an algorithm. Raises ValueError (and leaves the cube unchanged) if a move is
    /// blocked by the bandage. Returns the cube, so calls can be chained.
    fn apply<'py>(mut slf: PyRefMut<'py, Self>, alg: &str) -> PyResult<PyRefMut<'py, Self>> {
        let a = slf.parse(alg)?;
        slf.apply_alg(&a).map_err(err)?;
        Ok(slf)
    }

    /// True if the whole algorithm can be executed from the current state.
    fn can_apply(&self, alg: &str) -> PyResult<bool> {
        let a = self.parse(alg)?;
        Ok(self.clone().apply_alg(&a).is_ok())
    }

    /// Moves of the move set that the bandage allows now. The default move set is the 18 face
    /// turns on the 3x3x3 and every single-layer turn (outer and inner) on the 4x4x4.
    #[pyo3(signature = (moves = None))]
    fn legal_moves(&self, moves: Option<&str>) -> PyResult<Vec<String>> {
        Ok(match &self.inner {
            AnyCube::C3(c) => {
                let ms = moves::parse_moveset(moves.unwrap_or(Bandage::DEFAULT_MOVES)).map_err(err)?;
                c.legal_moves(&ms).iter().map(|&m| moves::move_name(m)).collect()
            }
            AnyCube::C4(c) => {
                let ms = cube4::parse_moveset4(moves.unwrap_or(Bandage4::DEFAULT_MOVES)).map_err(err)?;
                c.legal_moves(&ms).iter().map(|&m| cube4::move_name4(m)).collect()
            }
        })
    }

    /// Solved, including the bandage shape (bandaged centers must not be spun or moved).
    fn is_solved(&self) -> bool {
        match &self.inner {
            AnyCube::C3(c) => c.is_solved(),
            AnyCube::C4(c) => c.is_solved(),
        }
    }

    /// Return to the solved state (the bandage is kept).
    fn reset(&mut self) {
        match &mut self.inner {
            AnyCube::C3(c) => c.state = State::solved(),
            AnyCube::C4(c) => c.state = State4::solved(),
        }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Facelet colours as a string in URFDLB order, each face row by row (Kociemba layout):
    /// 54 characters on the 3x3x3, 96 on the 4x4x4.
    fn facelets(&self) -> String {
        match &self.inner {
            AnyCube::C3(c) => c.state.facelets(),
            AnyCube::C4(c) => c.state.facelets(),
        }
    }

    /// What the current state does relative to solved: cycles, flips, twists. On the 4x4x4,
    /// wings are reported as edges and centers of one colour are interchangeable unless glued.
    fn effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let e = match &self.inner {
            AnyCube::C3(c) => c.state.effect(),
            AnyCube::C4(c) => c.effect(),
        };
        effect_dict(py, &e, self.names())
    }

    /// Bandage blocks as lists of cubie names.
    #[getter]
    fn bandage(&self) -> Vec<Vec<String>> {
        match &self.inner {
            AnyCube::C3(c) => c.bandage.block_names(),
            AnyCube::C4(c) => c.bandage.block_names(),
        }
    }

    /// Moved blocks as cycles, e.g. "(UFL+UF ULB+UB)" (the block at the first home is now at the
    /// second); "" if every block is home, "?" if the puzzle does not have its home shape.
    fn blocks_moved(&self) -> String {
        match &self.inner {
            AnyCube::C3(c) => c.bandage.describe_blocks(&c.state),
            AnyCube::C4(c) => c.bandage.describe_blocks(&c.state),
        }
    }

    /// Pairs of facelet indices (same face, adjacent) whose cubies are glued together.
    fn bonded_facelet_pairs(&self) -> Vec<(usize, usize)> {
        match &self.inner {
            AnyCube::C3(c) => c.bandage.bonded_facelet_pairs(&c.state),
            AnyCube::C4(c) => c.bandage.bonded_facelet_pairs(&c.state),
        }
    }

    /// Name of the cubie currently showing each facelet.
    fn facelet_cubies(&self) -> Vec<&'static str> {
        let names = self.names();
        match &self.inner {
            AnyCube::C3(c) => crate::cube::facelet_cubies(&c.state).iter().map(|&x| names[x]).collect(),
            AnyCube::C4(c) => cube4::facelet_cubies4(&c.state).iter().map(|&x| names[x]).collect(),
        }
    }

    /// For every cubie: (name, name of the slot it occupies, orientation).
    fn pieces(&self) -> Vec<(&'static str, &'static str, u8)> {
        let names = self.names();
        match &self.inner {
            AnyCube::C3(c) => (0..names.len()).map(|x| (names[x], names[c.state.slot(x)], c.state.ori(x))).collect(),
            AnyCube::C4(c) => (0..names.len()).map(|x| (names[x], names[c.state.slot(x)], c.state.ori(x))).collect(),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (AnyCube::C3(a), AnyCube::C3(b)) => a.state.pack() == b.state.pack() && a.bandage == b.bandage,
            (AnyCube::C4(a), AnyCube::C4(b)) => {
                a.bandage == b.bandage && a.bandage.canonical(&a.state) == b.bandage.canonical(&b.state)
            }
            _ => false,
        }
    }

    fn __repr__(&self) -> String {
        let n = self.size();
        let f: Vec<char> = self.facelets().chars().collect();
        let row = |face: usize, r: usize| -> String {
            (0..n).map(|c| f[face * n * n + r * n + c].to_string()).collect::<Vec<_>>().join(" ")
        };
        let pad = " ".repeat(2 * n);
        let mut s = String::new();
        for r in 0..n {
            s += &format!("{}{}\n", pad, row(0, r));
        }
        for r in 0..n {
            s += &format!("{} {} {} {}\n", row(4, r), row(2, r), row(1, r), row(5, r));
        }
        for r in 0..n {
            s += &format!("{}{}\n", pad, row(3, r));
        }
        s
    }
}

enum AnySolver {
    S3(Solver),
    S4(Solver<Bandage4>),
}

/// Meet-in-the-middle algorithm search for a (bandaged) 3x3x3 or 4x4x4.
///
/// Building the solver enumerates every state within `depth` moves of solved (stopping early,
/// at a complete layer, if `max_nodes` would be exceeded). Queries then find algorithms of up to
/// `2*depth + extra_depth` moves.
///
/// Args:
///     bandage: blocks of cubies, same formats as for Cube.
///     moves: move set. 3x3x3: "htm" (18 face turns, the default), "stm" (faces + slices),
///         "R U F", "U M", "htm slices wide". 4x4x4: "stm" (every outer and inner layer, the
///         default), "htm" (outer faces), "inner", "wide", "R 2R U". A base without suffix
///         adds its three turns.
///     depth: maximum table depth (default 6 on the 3x3x3, 4 on the 4x4x4).
///     max_nodes: maximum number of stored states (about 40 bytes each on the 3x3x3, 60 on the
///         4x4x4).
///     size: 3 or 4.
#[pyclass(name = "Solver", module = "rubik")]
pub struct PySolver {
    inner: AnySolver,
    last_stats: SearchStats,
}

fn build_solver<P: Sized3or4>(
    py: Python<'_>,
    spec: &str,
    moves: Option<&str>,
    depth: Option<usize>,
    max_nodes: usize,
) -> PyResult<Solver<P>> {
    let b = P::parse_bandage(spec).map_err(err)?;
    let ms = P::parse_moveset(moves.unwrap_or(P::DEFAULT_MOVES)).map_err(err)?;
    let depth = depth.unwrap_or(P::DEFAULT_DEPTH);
    Ok(py.detach(|| Solver::new(b, ms, depth, max_nodes)))
}

fn find_in<'py, P: Sized3or4>(
    py: Python<'py>,
    solver: &Solver<P>,
    q: Query,
) -> PyResult<(Vec<Bound<'py, PyDict>>, SearchStats)> {
    let (found, stats) = py.detach(|| solver.search(&q)).map_err(err)?;
    let names = P::names();
    let dicts = found
        .iter()
        .map(|f| {
            let d = PyDict::new(py);
            d.set_item("alg", P::format_alg(&f.alg))?;
            d.set_item("length", f.length)?;
            d.set_item("qtm", f.qtm)?;
            d.set_item("effect", f.effect.describe_with(names))?;
            d.set_item("pieces", f.effect.affected().iter().map(|&c| names[c]).collect::<Vec<_>>())?;
            d.set_item("pure", f.effect.is_pure())?;
            d.set_item("kind", q.kind.name())?;
            let st = solver.bandage.replay(&f.alg).map_err(err)?;
            d.set_item("blocks", solver.bandage.describe_blocks(&st))?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((dicts, stats))
}

fn home_shapes_in<'py, P: Sized3or4>(
    py: Python<'py>,
    solver: &Solver<P>,
    max_results: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let found = py.detach(|| crate::study::home_shape_states(solver));
    let names = P::names();
    found
        .iter()
        .take(max_results)
        .map(|&(d, k)| {
            let st = solver.bandage.unpack(k);
            let alg = solver.path_to(&st).unwrap();
            let dict = PyDict::new(py);
            dict.set_item("alg", P::format_alg(&alg))?;
            dict.set_item("length", d)?;
            dict.set_item("blocks", solver.bandage.describe_blocks(&st))?;
            dict.set_item("effect", solver.bandage.effect_of(&st).describe_with(names))?;
            dict.set_item("free_solved", solver.bandage.free_solved(&st))?;
            Ok(dict)
        })
        .collect()
}

fn moves_of<P: Sized3or4>(solver: &Solver<P>) -> Vec<String> {
    solver.moves.iter().map(|&m| P::move_name(m)).collect()
}

/// Run `$body` with `$s` bound to the solver of either size.
macro_rules! with_solver {
    ($self:expr, $s:ident => $body:expr) => {
        match &$self.inner {
            AnySolver::S3($s) => $body,
            AnySolver::S4($s) => $body,
        }
    };
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (bandage = None, moves = None, depth = None, max_nodes = 5_000_000, size = 3))]
    fn new(
        py: Python<'_>,
        bandage: Option<&Bound<'_, PyAny>>,
        moves: Option<&str>,
        depth: Option<usize>,
        max_nodes: usize,
        size: usize,
    ) -> PyResult<Self> {
        check_size(size)?;
        let spec = bandage_spec(bandage)?;
        let inner = if size == 3 {
            AnySolver::S3(build_solver::<Bandage>(py, &spec, moves, depth, max_nodes)?)
        } else {
            AnySolver::S4(build_solver::<Bandage4>(py, &spec, moves, depth, max_nodes)?)
        };
        Ok(PySolver { inner, last_stats: SearchStats::default() })
    }

    /// Find algorithms.
    ///
    /// Args:
    ///     kind: "edge3" (edge 3-cycle; wing 3-cycle on the 4x4x4), "corner3" (corner 3-cycle),
    ///         "center3" (center 3-cycle, 4x4x4 only), "edge_flip" (2 edges flipped, 3x3x3
    ///         only), "corner_twist" (2 corners twisted), "orient" (up to 2 flips and/or
    ///         2 twists, all in place), or "any" (anything restricted to `pieces`).
    ///     pieces: restrict the effect to these cubies, e.g. ["UF", "UR", "UB"].
    ///     pure: 3-cycles must keep the orientation of the cycled pieces.
    ///     extra_depth: stream this many extra moves on one side beyond the table.
    ///     max_length: discard longer algorithms.
    ///     max_results: number of algorithms returned (shortest first).
    ///     unique_effects: keep only the shortest algorithm for each distinct effect.
    ///     time_limit: seconds allowed for the extra_depth stream (results so far are kept).
    ///
    /// Returns a list of dicts with keys alg, length (STM/HTM), qtm, effect, pieces, pure, kind.
    #[pyo3(signature = (kind, pieces = None, pure = false, extra_depth = 0, max_length = None,
                        max_results = 100, unique_effects = false, time_limit = None, move_blocks = false))]
    #[allow(clippy::too_many_arguments)]
    fn find<'py>(
        &mut self,
        py: Python<'py>,
        kind: &str,
        pieces: Option<Vec<String>>,
        pure: bool,
        extra_depth: usize,
        max_length: Option<usize>,
        max_results: usize,
        unique_effects: bool,
        time_limit: Option<f64>,
        move_blocks: bool,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let kind = Kind::parse(kind).map_err(err)?;
        let q = Query {
            pure,
            extra_depth,
            max_length,
            max_results,
            unique_effects,
            time_limit,
            move_blocks,
            ..Query::new(kind)
        };
        let (found, stats) = match &self.inner {
            AnySolver::S3(s) => {
                let pieces = pieces.map(Bandage::parse_pieces).transpose()?;
                find_in(py, s, Query { pieces, ..q })?
            }
            AnySolver::S4(s) => {
                let pieces = pieces.map(Bandage4::parse_pieces).transpose()?;
                find_in(py, s, Query { pieces, ..q })?
            }
        };
        self.last_stats = stats;
        Ok(found)
    }

    /// Every position in the table that has the home shape (blocks possibly exchanged for
    /// congruent ones), nearest first, with an optimal algorithm. On a puzzle the table holds
    /// completely, these are all the block-level effects of the puzzle with optimal lengths.
    ///
    /// Each dict: alg, length, blocks (block cycles, e.g. "(UFL+UF ULB+UB)"), effect (cubie level),
    /// free_solved (every unglued corner and edge is home).
    #[pyo3(signature = (max_results = 1000))]
    fn home_shape_effects<'py>(&self, py: Python<'py>, max_results: usize) -> PyResult<Vec<Bound<'py, PyDict>>> {
        with_solver!(self, s => home_shapes_in(py, s, max_results))
    }

    /// Distance from solved of a cube state, if it lies within the table.
    fn distance(&self, cube: &PyCube) -> PyResult<Option<usize>> {
        match (&self.inner, &cube.inner) {
            (AnySolver::S3(s), AnyCube::C3(c)) => Ok(s.distance(&c.state)),
            (AnySolver::S4(s), AnyCube::C4(c)) => Ok(s.distance(&c.state)),
            _ => Err(err("the cube and the solver have different sizes".into())),
        }
    }

    /// An optimal algorithm that *creates* the given state from solved (invert it to solve).
    fn path_to(&self, cube: &PyCube) -> PyResult<Option<String>> {
        match (&self.inner, &cube.inner) {
            (AnySolver::S3(s), AnyCube::C3(c)) => Ok(s.path_to(&c.state).map(|a| moves::format_alg(&a))),
            (AnySolver::S4(s), AnyCube::C4(c)) => Ok(s.path_to(&c.state).map(|a| cube4::format_alg4(&a))),
            _ => Err(err("the cube and the solver have different sizes".into())),
        }
    }

    /// 3 or 4.
    #[getter]
    fn size(&self) -> usize {
        match &self.inner {
            AnySolver::S3(_) => 3,
            AnySolver::S4(_) => 4,
        }
    }

    #[getter]
    fn depth(&self) -> usize {
        with_solver!(self, s => s.depth)
    }

    /// False if `max_nodes` stopped the table before the requested depth.
    #[getter]
    fn reached_max_depth(&self) -> bool {
        with_solver!(self, s => s.reached_max_depth)
    }

    #[getter]
    fn num_states(&self) -> usize {
        with_solver!(self, s => s.num_states())
    }

    /// Number of distinct states at each distance from solved.
    #[getter]
    fn layer_sizes(&self) -> Vec<usize> {
        with_solver!(self, s => s.layer_sizes())
    }

    #[getter]
    fn moves(&self) -> Vec<String> {
        with_solver!(self, s => moves_of(s))
    }

    #[getter]
    fn bandage(&self) -> Vec<Vec<String>> {
        with_solver!(self, s => s.bandage.block_names())
    }

    /// Statistics of the last `find` call.
    #[getter]
    fn last_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("candidates", self.last_stats.candidates)?;
        d.set_item("illegal", self.last_stats.illegal)?;
        d.set_item("matched", self.last_stats.matched)?;
        d.set_item("truncated", self.last_stats.truncated)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let (depth, states, moves, blocks) =
            with_solver!(self, s => (s.depth, s.num_states(), s.moves.len(), s.bandage.n_blocks()));
        format!("Solver(size={}, depth={}, states={}, moves={}, blocks={})", self.size(), depth, states, moves, blocks)
    }
}

fn parse_alg_sized(alg: &str, size: usize) -> PyResult<Alg> {
    check_size(size)?;
    if size == 3 { moves::parse_alg(alg) } else { cube4::parse_alg4(alg) }.map_err(err)
}

fn format_alg_sized(alg: &[Move], size: usize) -> String {
    if size == 3 { moves::format_alg(alg) } else { cube4::format_alg4(alg) }
}

/// Normalise an algorithm: expand groups/commutators, cancel and merge moves.
#[pyfunction]
#[pyo3(signature = (alg, size = 3))]
fn simplify_alg(alg: &str, size: usize) -> PyResult<String> {
    let a = parse_alg_sized(alg, size)?;
    let s = if size == 3 { moves::simplify(&a) } else { cube4::simplify4(&a) };
    Ok(format_alg_sized(&s, size))
}

/// Inverse of an algorithm.
#[pyfunction]
#[pyo3(signature = (alg, size = 3))]
fn invert_alg(alg: &str, size: usize) -> PyResult<String> {
    Ok(format_alg_sized(&moves::invert(&parse_alg_sized(alg, size)?), size))
}

/// (length, qtm) of an algorithm; whole-cube rotations are free.
#[pyfunction]
#[pyo3(signature = (alg, size = 3))]
fn alg_length(alg: &str, size: usize) -> PyResult<(usize, usize)> {
    let a = parse_alg_sized(alg, size)?;
    Ok(if size == 3 { Bandage::lengths(&a) } else { Bandage4::lengths(&a) })
}

/// Effect of an algorithm applied to a solved (optionally bandaged) cube.
#[pyfunction]
#[pyo3(signature = (alg, bandage = None, size = 3))]
fn analyze<'py>(
    py: Python<'py>,
    alg: &str,
    bandage: Option<&Bound<'_, PyAny>>,
    size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mut c = PyCube::new(bandage, None, size)?;
    let a = c.parse(alg)?;
    c.apply_alg(&a).map_err(err)?;
    c.effect(py)
}

/// Difficulty KPIs of a bandaged cube with face turns (see the README's difficulty study).
///
/// Returns a dict: states (exact unless states_exact is False), shapes, group_order, complete,
/// gods_number (exact, complete puzzles only), gods_lower, gods_counting, mean_distance
/// (a lower bound unless mean_distance_exact), branching, layer_sizes and, per algorithm kind
/// (edge3, corner3, edge_flip, corner_twist), min_len / exact / searched_to / possible /
/// coverage / example.
#[pyfunction]
#[pyo3(signature = (bandage = None, max_nodes = 10_000_000, samples = 112, ball_nodes = 300_000,
                    alg_time = 10.0, seed = 1))]
#[allow(clippy::too_many_arguments)]
fn difficulty<'py>(
    py: Python<'py>,
    bandage: Option<&Bound<'_, PyAny>>,
    max_nodes: usize,
    samples: usize,
    ball_nodes: usize,
    alg_time: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let b = bandage_from(bandage)?;
    let cfg = StudyConfig { max_nodes, samples, ball_nodes, alg_time_limit: alg_time, seed, ..StudyConfig::default() };
    let p = py.detach(|| profile(&b, &cfg));
    let d = PyDict::new(py);
    d.set_item("bandage", &p.spec)?;
    d.set_item("states", p.states)?;
    d.set_item("states_exact", p.states_exact)?;
    d.set_item("shapes", p.shapes)?;
    d.set_item("group_order", p.group_order)?;
    d.set_item("complete", p.complete)?;
    d.set_item("gods_number", p.gods_number)?;
    d.set_item("gods_lower", p.gods_lower)?;
    d.set_item("gods_counting", p.gods_counting)?;
    d.set_item("mean_distance", p.mean_distance)?;
    d.set_item("mean_distance_exact", p.mean_distance_exact)?;
    d.set_item("samples", p.samples)?;
    d.set_item("samples_resolved", p.samples_resolved)?;
    d.set_item("antipode_distance", p.antipode_distance)?;
    d.set_item("branching", p.branching)?;
    d.set_item("legal_at_solved", p.legal_at_solved)?;
    d.set_item("free_corners", p.free_corners)?;
    d.set_item("free_edges", p.free_edges)?;
    d.set_item("layer_sizes", p.layer_sizes.clone())?;
    d.set_item("antipode", &p.antipode)?;
    for (kind, a) in ALG_KINDS.iter().zip(&p.algs) {
        let k = PyDict::new(py);
        k.set_item("min_len", a.min_len)?;
        k.set_item("exact", a.exact)?;
        k.set_item("searched_to", a.searched_to)?;
        k.set_item("possible", a.possible)?;
        k.set_item("coverage", a.coverage)?;
        k.set_item("example", &a.example)?;
        d.set_item(kind.name(), k)?;
    }
    d.set_item("seconds", p.seconds)?;
    Ok(d)
}

/// Human difficulty index of a bandaged cube: an estimate, in moves, of a commutator solve
/// (see `rubik::human`). Divide `total` by the 3x3x3's `total` to get the index.
///
/// Returns a dict with shapes, home_positions, nav_mean, nav_max, branching, navigation, load,
/// align, discovery, penalty, total, and classes: one dict per piece class (name, units,
/// configs, tool_configs, perm_tools, orient_tools, load, align, discovery, penalty, tools),
/// each tool with kind, known (a standard algorithm that works here), possible, comm_depth,
/// commutator, cost. Also transfer: the share of the standard algorithm library usable here. None if there are too many shapes.
#[pyfunction]
#[pyo3(signature = (bandage = None, comm_depth = 6, setup = 2.0, discovery_weight = 5.0, samples = 400, seed = 1))]
fn human_difficulty<'py>(
    py: Python<'py>,
    bandage: Option<&Bound<'_, PyAny>>,
    comm_depth: usize,
    setup: f64,
    discovery_weight: f64,
    samples: usize,
    seed: u64,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let b = bandage_from(bandage)?;
    let cfg = crate::human::HumanConfig { comm_depth, setup, discovery_weight, samples, seed, ..Default::default() };
    let Some(p) = py.detach(|| crate::human::human_profile(&b, &cfg)) else { return Ok(None) };
    let d = PyDict::new(py);
    d.set_item("bandage", &p.spec)?;
    d.set_item("shapes", p.shapes)?;
    d.set_item("home_positions", p.home_positions)?;
    d.set_item("nav_mean", p.nav_mean)?;
    d.set_item("nav_max", p.nav_max)?;
    d.set_item("branching", p.branching)?;
    d.set_item("transfer", p.transfer)?;
    for (k, v) in [("navigation", p.navigation), ("load", p.load), ("align", p.align), ("discovery", p.discovery), ("penalty", p.penalty), ("total", p.total)] {
        d.set_item(k, v)?;
    }
    let mut classes = Vec::new();
    for c in &p.classes {
        let cd = PyDict::new(py);
        cd.set_item("name", &c.name)?;
        cd.set_item("units", c.units)?;
        cd.set_item("configs", c.configs)?;
        cd.set_item("tool_configs", c.tool_configs)?;
        cd.set_item("perm_tools", c.perm_tools)?;
        cd.set_item("orient_tools", c.orient_tools)?;
        cd.set_item("load", c.load)?;
        cd.set_item("align", c.align)?;
        cd.set_item("discovery", c.discovery)?;
        cd.set_item("penalty", c.penalty)?;
        let mut tools = Vec::new();
        for t in &c.tools {
            let td = PyDict::new(py);
            td.set_item("kind", t.kind.name())?;
            td.set_item("known", &t.known)?;
            td.set_item("possible", t.possible)?;
            td.set_item("comm_depth", t.comm_depth)?;
            td.set_item("commutator", &t.commutator)?;
            td.set_item("cost", t.cost)?;
            tools.push(td);
        }
        cd.set_item("tools", tools)?;
        classes.push(cd);
    }
    d.set_item("classes", classes)?;
    Ok(Some(d))
}

/// A random bandage made of `bonds` glued pairs of adjacent cubies, as a bandage string.
#[pyfunction]
#[pyo3(signature = (bonds, seed = 1))]
fn random_bandage(bonds: usize, seed: u64) -> String {
    crate::study::random_bandage(&mut Rng::new(seed), bonds).spec()
}

/// Cubie names. 3x3x3: 8 corners, 12 edges, 6 centers. 4x4x4: 8 corners, 24 wings, 24 centers.
#[pyfunction]
#[pyo3(signature = (size = 3))]
fn cubie_names(size: usize) -> PyResult<Vec<&'static str>> {
    check_size(size)?;
    Ok(if size == 3 { NAMES.to_vec() } else { geo4().names.clone() })
}

#[pymodule]
fn _rubik(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCube>()?;
    m.add_class::<PySolver>()?;
    m.add_function(wrap_pyfunction!(simplify_alg, m)?)?;
    m.add_function(wrap_pyfunction!(invert_alg, m)?)?;
    m.add_function(wrap_pyfunction!(alg_length, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(cubie_names, m)?)?;
    m.add_function(wrap_pyfunction!(difficulty, m)?)?;
    m.add_function(wrap_pyfunction!(random_bandage, m)?)?;
    m.add_function(wrap_pyfunction!(human_difficulty, m)?)?;
    Ok(())
}
