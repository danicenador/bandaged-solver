//! Human difficulty index of a bandaged cube: an estimate of the work of a commutator solve.
//!
//! The solve is modelled as: bring the puzzle back to its home shape, then fix each *class* of
//! pieces (the free corners, the free edges, each family of congruent glued blocks) with tools:
//! 3-cycles, 2-flips, 2-twists and block 3-cycles. Everything is in moves:
//!
//! * **navigation**: mean number of moves from a random position to the nearest home-shape
//!   position (a breadth-first search of the shape graph; every shape holds the same number of
//!   positions, so the plain mean over shapes is the mean over positions).
//! * **load**: per class, the number of tool applications a random position needs (from the
//!   subgroup K the tools generate) times the cost of one application: a commutator `[X, Y]` of
//!   length `2|X| + 2` plus `setup` moves.
//! * **alignment**: the part of a class the tools cannot reach (`|G_c| / |K_c|`, where G_c is
//!   everything the puzzle can do to the class) is fixed with plain moves: `ln(ratio) / ln(b)`.
//!   On the 3x3x3 that is the corner/edge parity (ratio 2). Above `align_limit` configurations it
//!   also costs a penalty: the solver needs a method the tools do not give.
//! * **discovery**: `|X|` of the shortest commutator `[X, Y]` (Y one face turn) for each tool a
//!   class needs, weighted by `discovery_weight`. Tools without such a commutator up to
//!   `comm_depth` count as `comm_depth + 1`.
//!
//! * **transfer**: a tool counts as *known* (no discovery, its own length as cost) when one of the
//!   standard 3x3x3 algorithms in [`LIBRARY`] does that job on this puzzle in some grip (one of the
//!   24 whole-cube orientations): legal, shape restored, effect of the right kind.
//!
//! `total = navigation + load + alignment + discovery_weight * discovery + penalty`, and the index
//! is `total / total(3x3x3)`.
//!
//! The groups use the shape graph where blocks may trade places with congruent blocks (a block
//! may end on another block's home), so G0 here includes block permutations.

use crate::cube::{Bandage, Cube, State};
use crate::geometry::{cubie_at_position, geom, piece_type, sticker_normals, PieceType, Vec3, EDGE0, NAMES, N_CUBIES};
use crate::group::{candidate_effects, compose, inverse, perm_of, point0, Chain, Perm, IDENTITY, N_POINTS};
use crate::moves::{amount_of, base_of, format_alg, invert, length, mv, parse_alg, parse_moveset, Move, N_BASES};
use crate::search::{Kind, Solver};
use crate::study::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub struct HumanConfig {
    /// Shape limit of the shape graph.
    pub max_shapes: usize,
    /// Longest X tried in the commutator search.
    pub comm_depth: usize,
    /// Node limit of the commutator search table.
    pub comm_nodes: usize,
    /// Random positions used to count tool applications.
    pub samples: usize,
    /// Setup moves added to every tool application (a conjugate S [X, Y] S').
    pub setup: f64,
    pub discovery_weight: f64,
    /// Configurations the tools cannot reach that still count as simple alignment.
    pub align_limit: f64,
    pub seed: u64,
}

impl Default for HumanConfig {
    fn default() -> Self {
        HumanConfig {
            max_shapes: 2_000_000,
            comm_depth: 6,
            comm_nodes: 12_000_000,
            samples: 400,
            setup: 2.0,
            discovery_weight: 5.0,
            align_limit: 24.0,
            seed: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolKind {
    Cycle3,
    Flip,
    Twist,
    Block3,
}

impl ToolKind {
    pub fn name(&self) -> &'static str {
        match self {
            ToolKind::Cycle3 => "3-cycle",
            ToolKind::Flip => "2-flip",
            ToolKind::Twist => "2-twist",
            ToolKind::Block3 => "block 3-cycle",
        }
    }
}

/// Standard 3x3x3 algorithms a solver is likely to know, by the effect they have.
pub const LIBRARY: &[(Kind, &str)] = &[
    (Kind::Edge3, "R U' R U R U R U' R' U' R2"),
    (Kind::Edge3, "R2 U R U R' U' R' U' R' U R'"),
    (Kind::Edge3, "M2 U M U2 M' U M2"),
    (Kind::Edge3, "M2 U' M U2 M' U' M2"),
    (Kind::Corner3, "R U R' D R U' R' D'"),
    (Kind::Corner3, "x R' U R' D2 R U' R' D2 R2 x'"),
    (Kind::Corner3, "x R2 D2 R U R' D2 R U' R x'"),
    (Kind::Corner3, "R' F R' B2 R F' R' B2 R2"),
    (Kind::EdgeFlip, "M' U M' U M' U2 M U M U M U2"),
    (Kind::EdgeFlip, "M' U M' U M' U M' U2 M' U M' U M' U M'"),
    (Kind::Edge3, "R U R' U' M' U R U' r'"),
    (Kind::CornerTwist, "R' D' R D R' D' R D U D' R' D R D' R' D R U'"),
    (Kind::CornerTwist, "R' D' R D R' D' R D U2 D' R' D R D' R' D R U2"),
];

const GRIPS: [&str; 24] = [
    "", "y", "y2", "y'", "x", "x y", "x y2", "x y'", "x2", "x2 y", "x2 y2", "x2 y'", "x'", "x' y", "x' y2", "x' y'",
    "z", "z y", "z y2", "z y'", "z'", "z' y", "z' y2", "z' y'",
];

/// For each kind, the shortest library algorithm that does it on this puzzle in some grip,
/// and the share of library entries usable at all.
pub fn known_tools(b: &Bandage) -> (HashMap<Kind, String>, f64) {
    let mut best: HashMap<Kind, (usize, String)> = HashMap::new();
    let mut usable = 0;
    for &(kind, alg) in LIBRARY {
        let mut ok = false;
        for grip in GRIPS {
            let g = parse_alg(grip).unwrap();
            let mut a = g.clone();
            a.extend(parse_alg(alg).unwrap());
            a.extend(invert(&g));
            let mut c = Cube::new(b.clone());
            if c.apply(&a).is_err() || !c.bandage_restored() || !kind.matches(&c.state.effect(), false) {
                continue;
            }
            ok = true;
            let text = if grip.is_empty() { alg.to_string() } else { format!("[{}: {}]", grip, alg) };
            let e = best.entry(kind).or_insert((usize::MAX, String::new()));
            if length(&a) < e.0 {
                *e = (length(&a), text);
            }
        }
        usable += ok as usize;
    }
    (best.into_iter().map(|(k, v)| (k, v.1)).collect(), usable as f64 / LIBRARY.len() as f64)
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub kind: ToolKind,
    /// A standard 3x3x3 algorithm that does this job here (in some grip), if any.
    pub known: String,
    /// Distinct effects of this kind the puzzle can do.
    pub possible: usize,
    /// |X| of the shortest commutator [X, Y] found, if any up to `comm_depth`.
    pub comm_depth: Option<usize>,
    pub commutator: String,
    /// Moves per application: 2|X| + 2 + setup.
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct PieceClass {
    pub name: String,
    pub units: usize,
    /// |G_c|: configurations of the class the puzzle can produce.
    pub configs: f64,
    /// |K_c|: configurations the tools can fix.
    pub tool_configs: f64,
    pub tools: Vec<Tool>,
    /// Mean tool applications for permutation and for orientation in a random position.
    pub perm_tools: f64,
    pub orient_tools: f64,
    pub load: f64,
    pub align: f64,
    pub discovery: f64,
    pub penalty: f64,
}

#[derive(Clone, Debug, Default)]
pub struct HumanProfile {
    pub spec: String,
    /// Shapes, counting blocks that traded places as the same shape.
    pub shapes: usize,
    /// Positions with the home shape (order of G0 with block permutations).
    pub home_positions: f64,
    pub nav_mean: f64,
    pub nav_max: usize,
    /// Mean legal face turns over all positions.
    pub branching: f64,
    /// Share of the standard algorithm library usable on this puzzle.
    pub transfer: f64,
    pub classes: Vec<PieceClass>,
    pub navigation: f64,
    pub load: f64,
    pub align: f64,
    pub discovery: f64,
    pub penalty: f64,
    pub total: f64,
}

// ---- shape graph with interchangeable blocks ----

fn shape_key(b: &Bandage, s: &State) -> Vec<u32> {
    let mut v: Vec<u32> = b
        .blocks
        .iter()
        .map(|&blk| (0..N_CUBIES).filter(|&c| blk & (1 << c) != 0).fold(0, |m, c| m | 1 << s.slot(c)))
        .collect();
    v.sort_unstable();
    v
}

struct ShapeGraph {
    dist: Vec<usize>,
    legal: Vec<usize>,
    group: Chain,
}

/// Breadth-first search of the shapes; closing every non-tree edge gives generators of G0.
fn shape_graph(b: &Bandage, moves: &[Move], max_shapes: usize) -> Option<ShapeGraph> {
    let mut index: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut reps = vec![State::solved()];
    let mut rep_inv = vec![IDENTITY];
    let mut dist = vec![0usize];
    let mut legal = Vec::new();
    index.insert(shape_key(b, &State::solved()), 0);
    let mut group = Chain::new();
    let mut i = 0;
    while i < reps.len() {
        let s = reps[i];
        let mut allowed = [false; N_BASES];
        for &m in moves {
            allowed[base_of(m)] = b.allows(&s, base_of(m));
        }
        legal.push(moves.iter().filter(|&&m| allowed[base_of(m)]).count());
        for &m in moves {
            if !allowed[base_of(m)] {
                continue;
            }
            let mut t = s;
            t.apply_move(m);
            let key = shape_key(b, &t);
            match index.get(&key) {
                Some(&j) => {
                    group.add(&compose(&rep_inv[j as usize], &perm_of(&t)));
                }
                None => {
                    if reps.len() >= max_shapes {
                        return None;
                    }
                    index.insert(key, reps.len() as u32);
                    reps.push(t);
                    rep_inv.push(inverse(&perm_of(&t)));
                    dist.push(dist[i] + 1);
                }
            }
        }
        i += 1;
    }
    Some(ShapeGraph { dist, legal, group })
}

// ---- piece classes and their tools ----

struct ClassDef {
    name: String,
    /// Library kind of each tool kind (free pieces only).
    lib: Vec<(ToolKind, Kind)>,
    /// Cubies of each unit (a free piece or a block), centers left out.
    units: Vec<Vec<usize>>,
    oriented: bool,
    perm_kind: ToolKind,
    orient_kind: Option<ToolKind>,
    candidates: Vec<(ToolKind, Vec<Perm>)>,
}

fn cubies_of(mask: u32) -> Vec<usize> {
    (0..N_CUBIES).filter(|&c| mask & (1 << c) != 0).collect()
}

fn sorted_positions(cubies: &[usize], g: u8) -> Vec<Vec3> {
    let geo = geom();
    let mut v: Vec<Vec3> = cubies.iter().map(|&c| geo.apply(g, geo.home[c])).collect();
    v.sort();
    v
}

/// Rotations carrying block `a` (at home) onto the home of block `b`.
fn block_maps(a: &[usize], b: &[usize]) -> Vec<u8> {
    let target = sorted_positions(b, 0);
    (0..24u8).filter(|&g| sorted_positions(a, g) == target).collect()
}

/// Sticker images of the cubies of `a` carried by rotation `g`.
fn block_move(a: &[usize], g: u8, p: &mut Perm) {
    let geo = geom();
    for &c in a {
        let s = cubie_at_position(geo.apply(g, geo.home[c])).unwrap();
        let theirs = sticker_normals(s);
        for (k, &n) in sticker_normals(c).iter().enumerate() {
            let moved = geo.apply(g, n);
            p[point0(c) + k] = (point0(s) + theirs.iter().position(|&x| x == moved).unwrap()) as u8;
        }
    }
}

fn block_3cycles(blocks: &[Vec<usize>]) -> Vec<Perm> {
    let mut out = Vec::new();
    let n = blocks.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                for (a, b, c) in [(i, j, k), (i, k, j)] {
                    for &ga in &block_maps(&blocks[a], &blocks[b]) {
                        for &gb in &block_maps(&blocks[b], &blocks[c]) {
                            for &gc in &block_maps(&blocks[c], &blocks[a]) {
                                let mut p = IDENTITY;
                                block_move(&blocks[a], ga, &mut p);
                                block_move(&blocks[b], gb, &mut p);
                                block_move(&blocks[c], gc, &mut p);
                                out.push(p);
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn class_defs(b: &Bandage) -> Vec<ClassDef> {
    let bandaged = b.bandaged_cubies();
    let free = !bandaged;
    let mut defs = Vec::new();
    for (t, name, c3, orient, ok) in [
        (PieceType::Corner, "free corners", Kind::Corner3, Kind::CornerTwist, ToolKind::Twist),
        (PieceType::Edge, "free edges", Kind::Edge3, Kind::EdgeFlip, ToolKind::Flip),
    ] {
        let units: Vec<Vec<usize>> =
            (0..N_CUBIES).filter(|&c| piece_type(c) == t && free & (1 << c) != 0).map(|c| vec![c]).collect();
        if units.is_empty() {
            continue;
        }
        defs.push(ClassDef {
            name: name.to_string(),
            units,
            oriented: true,
            lib: vec![(ToolKind::Cycle3, c3), (ok, orient)],
            perm_kind: ToolKind::Cycle3,
            orient_kind: Some(ok),
            candidates: vec![(ToolKind::Cycle3, candidate_effects(c3, free)), (ok, candidate_effects(orient, free))],
        });
    }
    // blocks without a center can move; group them by congruence
    let movable: Vec<Vec<usize>> = b
        .blocks
        .iter()
        .filter(|&&m| cubies_of(m).iter().all(|&c| piece_type(c) != PieceType::Center))
        .map(|&m| cubies_of(m))
        .collect();
    let mut used = vec![false; movable.len()];
    for i in 0..movable.len() {
        if used[i] {
            continue;
        }
        let class: Vec<usize> =
            (i..movable.len()).filter(|&j| !used[j] && !block_maps(&movable[i], &movable[j]).is_empty()).collect();
        for &j in &class {
            used[j] = true;
        }
        if class.len() < 2 {
            continue;
        }
        let blocks: Vec<Vec<usize>> = class.iter().map(|&j| movable[j].clone()).collect();
        let shape: Vec<&str> = blocks[0].iter().map(|&c| NAMES[c]).collect();
        defs.push(ClassDef {
            name: format!("blocks like {}", shape.join("+")),
            units: blocks.clone(),
            oriented: false,
            lib: Vec::new(),
            perm_kind: ToolKind::Block3,
            orient_kind: None,
            candidates: vec![(ToolKind::Block3, block_3cycles(&blocks))],
        });
    }
    defs
}

fn project(p: &Perm, pts: &[bool; N_POINTS]) -> Perm {
    let mut q = IDENTITY;
    for x in 0..N_POINTS {
        if pts[x] {
            q[x] = p[x];
        }
    }
    q
}

/// (tool applications for permutation, for orientation) needed to undo `p` on the class.
fn tools_needed(p: &Perm, d: &ClassDef, unit_of: &[Option<usize>; N_POINTS], cubie_of: &[usize; N_POINTS]) -> (f64, f64) {
    let n = d.units.len();
    let dest: Vec<usize> = d.units.iter().map(|u| unit_of[p[point0(u[0])] as usize].unwrap()).collect();
    let moved = (0..n).filter(|&u| dest[u] != u).count();
    let mut seen = vec![false; n];
    let mut cycles = 0;
    for u in 0..n {
        if seen[u] || dest[u] == u {
            continue;
        }
        cycles += 1;
        let mut v = u;
        while !seen[v] {
            seen[v] = true;
            v = dest[v];
        }
    }
    let mis = if d.oriented {
        d.units
            .iter()
            .filter(|u| {
                let x = p[point0(u[0])] as usize;
                x != point0(cubie_of[x])
            })
            .count()
    } else {
        0
    };
    ((moved - cycles) as f64 / 2.0, mis as f64 / 2.0)
}

// ---- commutator search ----

type Found = HashMap<(usize, ToolKind), (usize, String)>;

/// Shortest commutators [X, Y] (Y one face turn, |X| <= depth) whose effect is one of `targets`.
fn commutators(b: &Bandage, moves: &[Move], depth: usize, nodes: usize, targets: &HashMap<Perm, (usize, ToolKind)>) -> Found {
    let solver = Solver::new(b.clone(), moves.to_vec(), depth, nodes);
    let ys: Vec<(Move, Perm, Perm)> = moves
        .iter()
        .map(|&m| {
            let mut s = State::solved();
            s.apply_move(m);
            let p = perm_of(&s);
            (m, p, inverse(&p))
        })
        .collect();
    let wanted: std::collections::HashSet<(usize, ToolKind)> = targets.values().copied().collect();
    let found: Mutex<Found> = Mutex::new(HashMap::new());
    for d in 1..=solver.depth {
        solver.layer(d).par_iter().for_each(|&k| {
            let x = State::unpack(k);
            let px = perm_of(&x);
            let pxi = inverse(&px);
            for (m, py, pyi) in &ys {
                let p = compose(pyi, &compose(&pxi, &compose(py, &px)));
                let Some(&key) = targets.get(&p) else { continue };
                if found.lock().unwrap().contains_key(&key) {
                    continue;
                }
                let xa = solver.path_to(&x).unwrap();
                let mut alg = xa.clone();
                alg.push(*m);
                alg.extend(invert(&xa));
                alg.push(mv(base_of(*m), 4 - amount_of(*m)));
                if Cube::new(b.clone()).apply(&alg).is_ok() {
                    let mut f = found.lock().unwrap();
                    let e = f.entry(key).or_insert((d, String::new()));
                    if e.1.is_empty() || e.0 > d {
                        *e = (d, format!("[{}, {}]", format_alg(&xa), format_alg(&[*m])));
                    }
                }
            }
        });
        if wanted.iter().all(|k| found.lock().unwrap().contains_key(k)) {
            break;
        }
    }
    found.into_inner().unwrap()
}

/// Moves of a library entry as written by `known_tools` (`[grip: alg]` or `alg`); rotations are free.
fn known_len(text: &str) -> usize {
    let alg = text.trim_start_matches('[').trim_end_matches(']');
    let alg = alg.split_once(':').map_or(alg, |x| x.1);
    length(&parse_alg(alg).unwrap())
}

// ---- the index ----

pub fn human_profile(b: &Bandage, cfg: &HumanConfig) -> Option<HumanProfile> {
    let moves = parse_moveset("htm").unwrap();
    let sg = shape_graph(b, &moves, cfg.max_shapes)?;
    let mut p = HumanProfile {
        spec: b.spec(),
        shapes: sg.dist.len(),
        home_positions: sg.group.order() as f64,
        nav_mean: sg.dist.iter().sum::<usize>() as f64 / sg.dist.len() as f64,
        nav_max: *sg.dist.iter().max().unwrap(),
        branching: sg.legal.iter().sum::<usize>() as f64 / sg.legal.len() as f64,
        ..HumanProfile::default()
    };
    let ln_b = p.branching.max(2.0).ln();
    let defs = class_defs(b);

    // possible tools, the groups G_c and K_c, and the tools a random position needs
    let mut targets: HashMap<Perm, (usize, ToolKind)> = HashMap::new();
    let mut partial = Vec::new();
    for (ci, d) in defs.iter().enumerate() {
        let mut pts = [false; N_POINTS];
        let mut unit_of = [None; N_POINTS];
        let mut cubie_of = [0usize; N_POINTS];
        for (u, cubies) in d.units.iter().enumerate() {
            for &c in cubies {
                let n = if c < EDGE0 { 3 } else { 2 };
                for k in 0..n {
                    pts[point0(c) + k] = true;
                    unit_of[point0(c) + k] = Some(u);
                    cubie_of[point0(c) + k] = c;
                }
            }
        }
        let mut gc = Chain::new();
        for g in sg.group.generators() {
            gc.add(&project(g, &pts));
        }
        let mut kc = Chain::new();
        let mut all_tools = Vec::new();
        let mut possible = Vec::new();
        for (kind, cands) in &d.candidates {
            let ok: Vec<Perm> = cands.iter().filter(|q| sg.group.contains(q)).copied().collect();
            for q in &ok {
                kc.add(q);
                targets.insert(*q, (ci, *kind));
            }
            possible.push((*kind, ok.len()));
            all_tools.extend(ok);
        }
        let (mut tp, mut to) = (0.0, 0.0);
        if !all_tools.is_empty() {
            let mut rng = Rng::new(cfg.seed ^ (ci as u64 + 1) * 0x9e37_79b9);
            for _ in 0..cfg.samples {
                let mut e = IDENTITY;
                for _ in 0..200 {
                    e = compose(&all_tools[rng.below(all_tools.len())], &e);
                }
                let (a, o) = tools_needed(&e, d, &unit_of, &cubie_of);
                tp += a;
                to += o;
            }
            tp /= cfg.samples as f64;
            to /= cfg.samples as f64;
        }
        partial.push((gc.order() as f64, kc.order() as f64, possible, tp, to));
    }

    let found = commutators(b, &moves, cfg.comm_depth, cfg.comm_nodes, &targets);
    let (known, transfer) = known_tools(b);
    p.transfer = transfer;
    for (ci, (d, (g_ord, k_ord, possible, tp, to))) in defs.iter().zip(partial).enumerate() {
        let tools: Vec<Tool> = possible
            .iter()
            .filter(|(_, n)| *n > 0)
            .map(|&(kind, n)| {
                let f = found.get(&(ci, kind));
                let k = f.map_or(cfg.comm_depth + 1, |x| x.0);
                let lib = d.lib.iter().find(|x| x.0 == kind).and_then(|x| known.get(&x.1)).cloned().unwrap_or_default();
                let lib_len = (!lib.is_empty()).then(|| known_len(&lib));
                Tool {
                    kind,
                    known: lib,
                    possible: n,
                    comm_depth: f.map(|x| x.0),
                    commutator: f.map(|x| x.1.clone()).unwrap_or_default(),
                    cost: lib_len.map_or((2 * k + 2) as f64, |l| l as f64) + cfg.setup,
                }
            })
            .collect();
        let tool = |kind: Option<ToolKind>| kind.and_then(|k| tools.iter().find(|t| t.kind == k));
        // a known algorithm needs no discovery
        let depth = |t: &Tool| if t.known.is_empty() { t.comm_depth.unwrap_or(cfg.comm_depth + 1) as f64 } else { 0.0 };
        let perm = tool(Some(d.perm_kind));
        let orient = tool(d.orient_kind);
        let mut load = 0.0;
        let mut discovery = 0.0;
        if tp > 0.0
            && let Some(t) = perm
        {
            load += tp * t.cost;
            discovery += depth(t);
        }
        if to > 0.0 {
            // without a flip or twist, two 3-cycles fix a pair of orientations
            match (orient, perm) {
                (Some(t), _) => {
                    load += to * t.cost;
                    discovery += depth(t);
                }
                (None, Some(t)) => load += to * 2.0 * t.cost,
                _ => {}
            }
        }
        let ratio = g_ord / k_ord;
        let align = if ratio > 1.0 { ratio.ln() / ln_b } else { 0.0 };
        let penalty = if ratio > cfg.align_limit { cfg.discovery_weight * (cfg.comm_depth + 1) as f64 } else { 0.0 };
        p.classes.push(PieceClass {
            name: d.name.clone(),
            units: d.units.len(),
            configs: g_ord,
            tool_configs: k_ord,
            tools,
            perm_tools: tp,
            orient_tools: to,
            load,
            align,
            discovery,
            penalty,
        });
    }
    p.navigation = p.nav_mean;
    p.load = p.classes.iter().map(|c| c.load).sum();
    p.align = p.classes.iter().map(|c| c.align).sum();
    p.discovery = p.classes.iter().map(|c| c.discovery).sum();
    p.penalty = p.classes.iter().map(|c| c.penalty).sum();
    p.total = p.navigation + p.load + p.align + cfg.discovery_weight * p.discovery + p.penalty;
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_cube() {
        let p = human_profile(&Bandage::none(), &HumanConfig { comm_depth: 4, ..HumanConfig::default() }).unwrap();
        assert_eq!(p.shapes, 1);
        assert_eq!(p.home_positions, 43_252_003_274_489_856_000.0);
        assert_eq!(p.classes.len(), 2);
        let corners = &p.classes[0];
        assert_eq!(corners.configs / corners.tool_configs, 2.0); // parity
        let c3 = corners.tools.iter().find(|t| t.kind == ToolKind::Cycle3).unwrap();
        assert_eq!(c3.comm_depth, Some(3)); // [R U R', D]
    }

    #[test]
    fn library_is_valid() {
        // every entry does what it claims on a plain cube
        let (known, transfer) = known_tools(&Bandage::none());
        assert_eq!(transfer, 1.0);
        assert_eq!(known.len(), 4);
    }

    #[test]
    fn glued_blocks() {
        let b = Bandage::parse(
            "UBR+UR | UFL+UF | ULB+UB | UL+U | DRB+DR | DFR+FR | DLF+FL | DBL+BL | BR+R | DF+F | DL+L | DB+D+B",
        )
        .unwrap();
        let p = human_profile(&b, &HumanConfig::default()).unwrap();
        assert_eq!(p.home_positions, 2520.0);
        assert_eq!(p.shapes, 440);
        let blocks = p.classes.iter().find(|c| c.name.starts_with("blocks")).unwrap();
        assert_eq!(blocks.units, 7);
        assert_eq!(blocks.configs, 2520.0);
        assert_eq!(blocks.tool_configs, 2520.0);
    }
}
