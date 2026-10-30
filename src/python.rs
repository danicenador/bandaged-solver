//! Python bindings (module `rubik._rubik`, re-exported by the `rubik` package).

use crate::cube::{Bandage, Cube, Effect};
use crate::geometry::{parse_cubie, NAMES, N_CUBIES};
use crate::moves::{format_alg, invert, length, parse_alg, parse_moveset, qtm_length, simplify};
use crate::search::{Kind, Query, SearchStats, Solver};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Accepts None, a string ("UF+UFR+UR | D DF"), a list of strings (one block each) or a list
/// of lists of cubie names.
fn bandage_from(spec: Option<&Bound<'_, PyAny>>) -> PyResult<Bandage> {
    let Some(spec) = spec.filter(|s| !s.is_none()) else {
        return Ok(Bandage::none());
    };
    if let Ok(s) = spec.extract::<String>() {
        return Bandage::parse(&s).map_err(err);
    }
    if let Ok(groups) = spec.extract::<Vec<Vec<String>>>() {
        return Bandage::from_names(&groups).map_err(err);
    }
    if let Ok(blocks) = spec.extract::<Vec<String>>() {
        return Bandage::parse(&blocks.join("|")).map_err(err);
    }
    Err(err("bandage must be a string, a list of strings or a list of lists of cubie names".into()))
}

fn effect_dict<'py>(py: Python<'py>, e: &Effect) -> PyResult<Bound<'py, PyDict>> {
    let names = |cyc: &Vec<(usize, u8)>| cyc.iter().map(|x| NAMES[x.0]).collect::<Vec<_>>();
    let d = PyDict::new(py);
    d.set_item("description", e.describe())?;
    d.set_item("edge_cycles", e.edge_cycles.iter().map(names).collect::<Vec<_>>())?;
    d.set_item("corner_cycles", e.corner_cycles.iter().map(names).collect::<Vec<_>>())?;
    d.set_item(
        "center_cycles",
        e.center_cycles.iter().map(|c| c.iter().map(|&x| NAMES[x]).collect::<Vec<_>>()).collect::<Vec<_>>(),
    )?;
    d.set_item("flipped", e.flipped.iter().map(|&c| NAMES[c]).collect::<Vec<_>>())?;
    d.set_item(
        "twisted",
        e.twisted.iter().map(|&(c, o)| (NAMES[c], if o == 1 { "cw" } else { "ccw" })).collect::<Vec<_>>(),
    )?;
    d.set_item("affected", e.affected().iter().map(|&c| NAMES[c]).collect::<Vec<_>>())?;
    d.set_item("pure", e.is_pure())?;
    d.set_item("n_edges", e.n_edges())?;
    d.set_item("n_corners", e.n_corners())?;
    Ok(d)
}

/// A 3x3x3 cube, optionally bandaged.
///
/// Args:
///     bandage: blocks of cubies that always move together, as a string ("UF+UFR+UR | D+DF"),
///         a list of strings (["UF UFR UR", "D DF"]) or a list of lists of cubie names.
///         Overlapping blocks are merged.
///     alg: optional algorithm applied after construction.
#[pyclass(name = "Cube", module = "rubik")]
#[derive(Clone)]
pub struct PyCube {
    inner: Cube,
}

#[pymethods]
impl PyCube {
    #[new]
    #[pyo3(signature = (bandage = None, alg = None))]
    fn new(bandage: Option<&Bound<'_, PyAny>>, alg: Option<&str>) -> PyResult<Self> {
        let mut c = PyCube { inner: Cube::new(bandage_from(bandage)?) };
        if let Some(a) = alg {
            c.inner.apply(&parse_alg(a).map_err(err)?).map_err(err)?;
        }
        Ok(c)
    }

    /// Apply an algorithm. Raises ValueError (and leaves the cube unchanged) if a move is
    /// blocked by the bandage. Returns the cube, so calls can be chained.
    fn apply<'py>(mut slf: PyRefMut<'py, Self>, alg: &str) -> PyResult<PyRefMut<'py, Self>> {
        let a = parse_alg(alg).map_err(err)?;
        slf.inner.apply(&a).map_err(err)?;
        Ok(slf)
    }

    /// True if the whole algorithm can be executed from the current state.
    fn can_apply(&self, alg: &str) -> PyResult<bool> {
        let a = parse_alg(alg).map_err(err)?;
        Ok(self.inner.clone().apply(&a).is_ok())
    }

    /// Moves of the move set (default: the 18 face turns) that the bandage allows now.
    #[pyo3(signature = (moves = "htm"))]
    fn legal_moves(&self, moves: &str) -> PyResult<Vec<String>> {
        let ms = parse_moveset(moves).map_err(err)?;
        Ok(self.inner.legal_moves(&ms).iter().map(|&m| crate::moves::move_name(m)).collect())
    }

    /// Solved, including the bandage shape (bandaged centers must not be spun).
    fn is_solved(&self) -> bool {
        self.inner.is_solved()
    }

    /// Return to the solved state (the bandage is kept).
    fn reset(&mut self) {
        self.inner.state = crate::cube::State::solved();
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// 54 facelet colours as a string in URFDLB order (each face row by row, Kociemba layout).
    fn facelets(&self) -> String {
        self.inner.state.facelets()
    }

    /// What the current state does relative to solved: cycles, flips, twists.
    fn effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        effect_dict(py, &self.inner.state.effect())
    }

    /// Bandage blocks as lists of cubie names.
    #[getter]
    fn bandage(&self) -> Vec<Vec<String>> {
        self.inner.bandage.block_names()
    }

    /// Pairs of facelet indices (same face, adjacent) whose cubies are glued together.
    fn bonded_facelet_pairs(&self) -> Vec<(usize, usize)> {
        self.inner.bandage.bonded_facelet_pairs(&self.inner.state)
    }

    /// Name of the cubie currently showing each of the 54 facelets.
    fn facelet_cubies(&self) -> Vec<&'static str> {
        crate::cube::facelet_cubies(&self.inner.state).iter().map(|&c| NAMES[c]).collect()
    }

    /// For every cubie: (name, name of the slot it occupies, orientation).
    fn pieces(&self) -> Vec<(&'static str, &'static str, u8)> {
        (0..N_CUBIES).map(|c| (NAMES[c], NAMES[self.inner.state.slot(c)], self.inner.state.ori(c))).collect()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner.state.pack() == other.inner.state.pack() && self.inner.bandage == other.inner.bandage
    }

    fn __repr__(&self) -> String {
        let f: Vec<char> = self.inner.state.facelets().chars().collect();
        let row = |face: usize, r: usize| -> String {
            (0..3).map(|c| f[face * 9 + r * 3 + c].to_string()).collect::<Vec<_>>().join(" ")
        };
        let mut s = String::new();
        for r in 0..3 {
            s += &format!("      {}\n", row(0, r));
        }
        for r in 0..3 {
            s += &format!("{} {} {} {}\n", row(4, r), row(2, r), row(1, r), row(5, r));
        }
        for r in 0..3 {
            s += &format!("      {}\n", row(3, r));
        }
        s
    }
}

/// Meet-in-the-middle algorithm search for a (bandaged) cube.
///
/// Building the solver enumerates every state within `depth` moves of solved (stopping early,
/// at a complete layer, if `max_nodes` would be exceeded). Queries then find algorithms of up to
/// `2*depth + extra_depth` moves.
///
/// Args:
///     bandage: blocks of cubies, same formats as for Cube.
///     moves: move set, e.g. "htm" (18 face turns), "stm" (faces + slices), "R U F",
///         "U M", "htm slices wide". A base without suffix adds its three turns.
///     depth: maximum table depth.
///     max_nodes: maximum number of stored states (about 40 bytes each).
#[pyclass(name = "Solver", module = "rubik")]
pub struct PySolver {
    inner: Solver,
    last_stats: SearchStats,
}

#[pymethods]
impl PySolver {
    #[new]
    #[pyo3(signature = (bandage = None, moves = "htm", depth = 6, max_nodes = 5_000_000))]
    fn new(
        py: Python<'_>,
        bandage: Option<&Bound<'_, PyAny>>,
        moves: &str,
        depth: usize,
        max_nodes: usize,
    ) -> PyResult<Self> {
        let b = bandage_from(bandage)?;
        let ms = parse_moveset(moves).map_err(err)?;
        let inner = py.detach(|| Solver::new(b, ms, depth, max_nodes));
        Ok(PySolver { inner, last_stats: SearchStats::default() })
    }

    /// Find algorithms.
    ///
    /// Args:
    ///     kind: "edge3" (edge 3-cycle), "corner3" (corner 3-cycle), "edge_flip" (2 edges
    ///         flipped), "corner_twist" (2 corners twisted), "orient" (up to 2 flips and/or
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
                        max_results = 100, unique_effects = false, time_limit = None))]
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
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let kind = Kind::parse(kind).map_err(err)?;
        let pieces = match pieces {
            None => None,
            Some(p) => {
                let mut m = 0u32;
                for name in p {
                    m |= 1 << parse_cubie(&name).ok_or_else(|| err(format!("unknown cubie '{}'", name)))?;
                }
                Some(m)
            }
        };
        let q = Query {
            pieces,
            pure,
            extra_depth,
            max_length,
            max_results,
            unique_effects,
            time_limit,
            ..Query::new(kind)
        };
        let solver = &self.inner;
        let (found, stats) = py.detach(|| solver.search(&q)).map_err(err)?;
        self.last_stats = stats;
        found
            .iter()
            .map(|f| {
                let d = PyDict::new(py);
                d.set_item("alg", format_alg(&f.alg))?;
                d.set_item("length", f.length)?;
                d.set_item("qtm", f.qtm)?;
                d.set_item("effect", f.effect.describe())?;
                d.set_item("pieces", f.effect.affected().iter().map(|&c| NAMES[c]).collect::<Vec<_>>())?;
                d.set_item("pure", f.effect.is_pure())?;
                d.set_item("kind", kind.name())?;
                Ok(d)
            })
            .collect()
    }

    /// Distance from solved of a cube state, if it lies within the table.
    fn distance(&self, cube: &PyCube) -> Option<usize> {
        self.inner.distance(&cube.inner.state)
    }

    /// An optimal algorithm that *creates* the given state from solved (invert it to solve).
    fn path_to(&self, cube: &PyCube) -> Option<String> {
        self.inner.path_to(&cube.inner.state).map(|a| format_alg(&a))
    }

    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth
    }

    /// False if `max_nodes` stopped the table before the requested depth.
    #[getter]
    fn reached_max_depth(&self) -> bool {
        self.inner.reached_max_depth
    }

    #[getter]
    fn num_states(&self) -> usize {
        self.inner.num_states()
    }

    /// Number of distinct states at each distance from solved.
    #[getter]
    fn layer_sizes(&self) -> Vec<usize> {
        self.inner.layer_sizes()
    }

    #[getter]
    fn moves(&self) -> Vec<String> {
        self.inner.moves.iter().map(|&m| crate::moves::move_name(m)).collect()
    }

    #[getter]
    fn bandage(&self) -> Vec<Vec<String>> {
        self.inner.bandage.block_names()
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
        format!(
            "Solver(depth={}, states={}, moves={}, blocks={})",
            self.inner.depth,
            self.inner.num_states(),
            self.inner.moves.len(),
            self.inner.bandage.blocks.len()
        )
    }
}

/// Normalise an algorithm: expand groups/commutators, cancel and merge moves.
#[pyfunction]
fn simplify_alg(alg: &str) -> PyResult<String> {
    Ok(format_alg(&simplify(&parse_alg(alg).map_err(err)?)))
}

/// Inverse of an algorithm.
#[pyfunction]
fn invert_alg(alg: &str) -> PyResult<String> {
    Ok(format_alg(&invert(&parse_alg(alg).map_err(err)?)))
}

/// (length, qtm) of an algorithm; whole-cube rotations are free.
#[pyfunction]
fn alg_length(alg: &str) -> PyResult<(usize, usize)> {
    let a = parse_alg(alg).map_err(err)?;
    Ok((length(&a), qtm_length(&a)))
}

/// Effect of an algorithm applied to a solved (optionally bandaged) cube.
#[pyfunction]
#[pyo3(signature = (alg, bandage = None))]
fn analyze<'py>(py: Python<'py>, alg: &str, bandage: Option<&Bound<'_, PyAny>>) -> PyResult<Bound<'py, PyDict>> {
    let mut c = Cube::new(bandage_from(bandage)?);
    c.apply(&parse_alg(alg).map_err(err)?).map_err(err)?;
    effect_dict(py, &c.state.effect())
}

/// Names of the 26 cubies: 8 corners, 12 edges, 6 centers.
#[pyfunction]
fn cubie_names() -> Vec<&'static str> {
    NAMES.to_vec()
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
    Ok(())
}
