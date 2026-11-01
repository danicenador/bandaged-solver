//! Difficulty study of bandaged cubes: random bandage generation and difficulty KPIs.
//!
//! A [`Profile`] measures one puzzle with the 18 face turns (HTM):
//!
//! * **State space.** A breadth-first enumeration from solved (the [`Solver`] table). If it ends
//!   before the node limit, the puzzle is *complete*: the number of reachable states, God's
//!   number (largest distance from solved) and the distance distribution are exact.
//! * **Distances of large puzzles.** Uniform random states are drawn with a lazy random walk
//!   (propose one of the 18 moves, stay put if the bandage blocks it; the proposal is symmetric so
//!   the walk is uniform on the reachable states). From each sample a breadth-first *ball* is
//!   grown until it touches the table. The table holds every state within `d` moves, so the
//!   first hit at ball level `k` proves the distance is exactly `k + d`.
//!   The number of table hits inside the balls also estimates the size of the state space
//!   (`|table| * ball volume / hits`), and the counting argument turns that size into an
//!   estimate of God's number.
//! * **Algorithms.** Shortest edge 3-cycle, corner 3-cycle, 2-edge flip and 2-corner twist that
//!   leave everything else (bandaged blocks included) solved. On complete puzzles every state is
//!   scanned, so the lengths are optimal. Otherwise the meet-in-the-middle [`Solver::search`] is
//!   used and a length is proven optimal when it is within the exhaustively searched range.

use crate::cube::{diff_counts, Bandage, State};
use crate::group::{candidate_effects, count_states, point0, Chain, Perm};
use crate::geometry::{geom, piece_type, PieceType, EDGE0, ID_ROT, N_CUBIES};
use crate::moves::{base_of, format_alg, parse_moveset, Move};
use crate::search::{FastSet, Kind, Puzzle, Query, Solver};
use rayon::prelude::*;
use std::time::Instant;

/// SplitMix64: small, deterministic, good enough for sampling.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// Pairs of physically touching cubies that can be glued: corner-edge and edge-center.
pub fn adjacent_pairs() -> Vec<(usize, usize)> {
    let home = &geom().home;
    let mut v = Vec::new();
    for a in 0..N_CUBIES {
        for b in a + 1..N_CUBIES {
            let d: i32 = (0..3).map(|k| (home[a][k] - home[b][k]).abs() as i32).sum();
            if d == 1 {
                v.push((a, b));
            }
        }
    }
    v
}

/// A random bandage made of `bonds` distinct glued pairs of adjacent cubies (merged into blocks).
pub fn random_bandage(rng: &mut Rng, bonds: usize) -> Bandage {
    let mut pairs = adjacent_pairs();
    let n = bonds.min(pairs.len());
    for i in 0..n {
        let j = i + rng.below(pairs.len() - i);
        pairs.swap(i, j);
    }
    Bandage::from_masks(pairs[..n].iter().map(|&(a, b)| (1u32 << a) | (1 << b)).collect())
}

/// Number of face turns (out of 18) legal in state `s`.
pub fn legal_face_turns(bandage: &Bandage, s: &State, moves: &[Move]) -> usize {
    moves.iter().filter(|&&m| bandage.allows(s, base_of(m))).count()
}

/// The algorithm KPIs, in this order.
pub const ALG_KINDS: [Kind; 4] = [Kind::Edge3, Kind::Corner3, Kind::EdgeFlip, Kind::CornerTwist];

#[derive(Clone, Debug, Default)]
pub struct AlgKpi {
    /// Length (HTM) of the shortest algorithm found.
    pub min_len: Option<usize>,
    /// `min_len` is proven optimal.
    pub exact: bool,
    /// Every algorithm up to this length was examined.
    pub searched_to: usize,
    /// Number of distinct effects of this kind the puzzle can do at all (exact, from G0).
    pub possible: usize,
    /// Fraction of the free pieces of the right type involved in some possible effect.
    pub coverage: f64,
    pub example: String,
}

#[derive(Clone, Debug)]
pub struct StudyConfig {
    /// Node limit of the breadth-first table.
    pub max_nodes: usize,
    /// Shape limit of the exact state count (see [`crate::group`]).
    pub max_shapes: usize,
    /// Random states sampled on puzzles too large to enumerate.
    pub samples: usize,
    /// Node limit of each sample's ball.
    pub ball_nodes: usize,
    /// Random-walk length used to draw one sample.
    pub walk: usize,
    /// Extra streamed depth of the algorithm search (incomplete puzzles only).
    pub alg_extra_depth: usize,
    /// Time limit in seconds of each algorithm search stream.
    pub alg_time_limit: f64,
    pub seed: u64,
}

impl Default for StudyConfig {
    fn default() -> Self {
        StudyConfig {
            max_nodes: 10_000_000,
            max_shapes: 5_000_000,
            samples: 112,
            ball_nodes: 300_000,
            walk: 5_000,
            alg_extra_depth: 2,
            alg_time_limit: 20.0,
            seed: 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub spec: String,
    pub blocks: usize,
    pub bandaged_cubies: usize,
    pub free_corners: usize,
    pub free_edges: usize,
    /// Legal face turns in the solved state.
    pub legal_at_solved: usize,
    /// The whole reachable state space fits in the table.
    pub complete: bool,
    /// Last complete layer of the table.
    pub depth: usize,
    pub layer_sizes: Vec<usize>,
    pub table_states: usize,
    /// Number of reachable states (`shapes * group_order`), or only a lower bound (the table
    /// size) if the puzzle has more than `max_shapes` shapes.
    pub states: f64,
    pub states_exact: bool,
    /// Reachable configurations of the bandaged blocks.
    pub shapes: usize,
    /// Order of the group of shape-preserving loops acting on the corners and edges.
    pub group_order: u128,
    /// Exact God's number (largest distance from solved), complete puzzles only.
    pub gods_number: Option<usize>,
    /// Proven lower bound on God's number.
    pub gods_lower: usize,
    /// Counting-argument estimate of God's number: the depth at which the extrapolated layer
    /// sizes add up to `states`. Usually an underestimate (18 vs 20 on the 3x3x3).
    pub gods_counting: Option<usize>,
    /// Mean distance from solved of a uniform random state.
    pub mean_distance: f64,
    /// `mean_distance` is exact (complete) or every sample was resolved.
    pub mean_distance_exact: bool,
    pub samples: usize,
    pub samples_resolved: usize,
    /// Mean number of legal face turns in a uniform random state (18 on the 3x3x3).
    pub branching: f64,
    /// An optimal sequence to one of the farthest states found.
    pub antipode: String,
    pub antipode_distance: usize,
    pub algs: Vec<AlgKpi>,
    pub seconds: f64,
}

/// Every bandaged cubie is home and unrotated (the puzzle has its original shape).
fn restored(bandaged: u32, s: &State) -> bool {
    (0..N_CUBIES).all(|c| bandaged & (1 << c) == 0 || s.rot[c] == ID_ROT)
}

fn free_count(bandaged: u32, t: PieceType) -> usize {
    (0..N_CUBIES).filter(|&c| piece_type(c) == t && bandaged & (1 << c) == 0).count()
}

fn kind_piece_type(k: Kind) -> PieceType {
    match k {
        Kind::Edge3 | Kind::EdgeFlip => PieceType::Edge,
        _ => PieceType::Corner,
    }
}

/// Uniform random reachable state: lazy walk with the 18 face turns as symmetric proposals.
pub fn random_state(bandage: &Bandage, moves: &[Move], rng: &mut Rng, steps: usize) -> State {
    let mut s = State::solved();
    for _ in 0..steps {
        let m = moves[rng.below(moves.len())];
        if bandage.allows(&s, base_of(m)) {
            s.apply_move(m);
        }
    }
    s
}

/// Grow a breadth-first ball around `start` until it touches the table. The table holds every
/// state within `d` moves of solved, so the first level `k` that touches it gives exactly
/// distance `k + d`. Returns (exact distance if found, proven lower bound).
fn probe(solver: &Solver, start: State, budget: usize) -> (Option<usize>, usize) {
    let d = solver.depth;
    let key = start.pack();
    if let Some(v) = solver.distance_packed(key) {
        return (Some(v), v);
    }
    let mut seen = FastSet::default();
    seen.insert(key);
    let mut frontier = vec![start];
    let mut k = 0;
    while !frontier.is_empty() {
        k += 1;
        let mut next = Vec::new();
        let mut best: Option<usize> = None;
        for st in &frontier {
            for &m in &solver.moves {
                if !solver.bandage.allows(st, base_of(m)) {
                    continue;
                }
                let mut ns = *st;
                ns.apply_move(m);
                let key = ns.pack();
                if !seen.insert(key) {
                    continue;
                }
                if let Some(v) = solver.distance_packed(key) {
                    best = Some(best.map_or(v, |b: usize| b.min(v)));
                }
                next.push(ns);
            }
            if best.is_none() && seen.len() > budget {
                return (None, d + k); // level k is incomplete
            }
        }
        if let Some(v) = best {
            return (Some(k + v), k + v);
        }
        frontier = next;
    }
    unreachable!("the solved state is always reachable")
}

/// Smallest depth at which the layer sizes, extrapolated with the growth ratio of the last two
/// complete layers, add up to `n` states.
pub fn counting_estimate(layers: &[usize], n: f64) -> Option<usize> {
    let mut cum: f64 = layers.iter().map(|&x| x as f64).sum();
    let mut depth = layers.len() - 1;
    if cum >= n {
        return Some(depth);
    }
    if layers.len() < 2 {
        return None;
    }
    let r = layers[depth] as f64 / layers[depth - 1] as f64;
    if r <= 1.0 {
        return None;
    }
    let mut layer = layers[depth] as f64;
    while cum < n && depth < 1000 {
        layer *= r;
        cum += layer;
        depth += 1;
    }
    Some(depth)
}

/// Algorithm KPIs of a complete puzzle: every reachable state is an effect, and its distance is
/// the optimal length of that effect. Also returns the number of states matching each kind.
fn scan_algs(solver: &Solver, bandaged: u32) -> (Vec<AlgKpi>, [usize; 4]) {
    let solved = State::solved().pack();
    // per kind: (min depth, state at that depth, number of matching states)
    type Acc = [(usize, u128, usize); 4];
    let empty: Acc = [(usize::MAX, 0, 0); 4];
    let acc = (0..=solver.depth)
        .into_par_iter()
        .flat_map(|d| solver.layer(d).par_iter().map(move |&k| (d, k)))
        .fold(
            || empty,
            |mut a, (d, k)| {
                let s = State::unpack(k);
                if !restored(bandaged, &s) {
                    return a;
                }
                let diff = diff_counts(k, solved);
                for (i, kind) in ALG_KINDS.iter().enumerate() {
                    if kind.diff_ok(diff) && kind.matches(&s.effect(), false) {
                        if (d, k) < (a[i].0, a[i].1) {
                            a[i].0 = d;
                            a[i].1 = k;
                        }
                        a[i].2 += 1;
                    }
                }
                a
            },
        )
        .reduce(
            || empty,
            |mut a, b| {
                for i in 0..4 {
                    if (b[i].0, b[i].1) < (a[i].0, a[i].1) {
                        a[i].0 = b[i].0;
                        a[i].1 = b[i].1;
                    }
                    a[i].2 += b[i].2;
                }
                a
            },
        );
    let algs = acc
        .iter()
        .map(|&(d, k, _)| AlgKpi {
            min_len: (d != usize::MAX).then_some(d),
            exact: true,
            searched_to: solver.depth,
            example: if d == usize::MAX {
                String::new()
            } else {
                format_alg(&solver.path_to(&State::unpack(k)).unwrap())
            },
            ..AlgKpi::default()
        })
        .collect();
    (algs, acc.map(|x| x.2))
}

/// Algorithm KPIs of a large puzzle, with the meet-in-the-middle search.
fn search_algs(solver: &Solver, cfg: &StudyConfig) -> Vec<AlgKpi> {
    ALG_KINDS
        .iter()
        .map(|&kind| {
            let q = Query {
                extra_depth: cfg.alg_extra_depth,
                time_limit: Some(cfg.alg_time_limit),
                max_results: 1,
                ..Query::new(kind)
            };
            let (found, stats) = solver.search(&q).expect("valid query");
            let searched_to = if stats.truncated { 2 * solver.depth } else { 2 * solver.depth + cfg.alg_extra_depth };
            let min_len = found.first().map(|f| f.length);
            AlgKpi {
                min_len,
                exact: min_len.is_some_and(|l| l <= searched_to),
                searched_to,
                example: found.first().map(|f| format_alg(&f.alg)).unwrap_or_default(),
                ..AlgKpi::default()
            }
        })
        .collect()
}

/// Exact number of effects of each kind (`ALG_KINDS` order) the puzzle can do, and the fraction
/// of its free pieces they involve; None if it has more than `max_shapes` shapes.
pub fn effect_counts(bandage: &Bandage, max_shapes: usize) -> Option<[(usize, f64); 4]> {
    let moves = parse_moveset("htm").unwrap();
    count_states(bandage, &moves, max_shapes).map(|c| possible_effects(&c.chain, bandage.bandaged_cubies()))
}

/// Exact number of effects of each kind in G0, and the fraction of free pieces they involve.
fn possible_effects(chain: &Chain, bandaged: u32) -> [(usize, f64); 4] {
    ALG_KINDS.map(|kind| {
        let free = !bandaged;
        let ok: Vec<Perm> = candidate_effects(kind, free).into_iter().filter(|p| chain.contains(p)).collect();
        let lo = if kind_piece_type(kind) == PieceType::Edge { EDGE0 } else { 0 };
        let n_free = free_count(bandaged, kind_piece_type(kind));
        let involved = (lo..lo + if lo == 0 { 8 } else { 12 })
            .filter(|&c| ok.iter().any(|p| p[point0(c)] as usize != point0(c)))
            .count();
        (ok.len(), if n_free == 0 { 0.0 } else { involved as f64 / n_free as f64 })
    })
}

/// Measure every KPI of one bandaged cube (face turns only).
pub fn profile(bandage: &Bandage, cfg: &StudyConfig) -> Profile {
    let t0 = Instant::now();
    let moves = parse_moveset("htm").unwrap();
    let bandaged = bandage.bandaged_cubies();
    let solver = Solver::new(bandage.clone(), moves.clone(), 1000, cfg.max_nodes);
    let complete = solver.reached_max_depth && solver.depth < 1000;
    let layers = solver.layer_sizes();
    let mut p = Profile {
        spec: bandage.spec(),
        blocks: bandage.blocks.len(),
        bandaged_cubies: bandaged.count_ones() as usize,
        free_corners: free_count(bandaged, PieceType::Corner),
        free_edges: free_count(bandaged, PieceType::Edge),
        legal_at_solved: legal_face_turns(bandage, &State::solved(), &moves),
        complete,
        depth: solver.depth,
        layer_sizes: layers.clone(),
        table_states: solver.num_states(),
        ..Profile::default()
    };

    let count = count_states(bandage, &moves, cfg.max_shapes);
    match &count {
        Some(c) => {
            p.shapes = c.shapes;
            p.group_order = c.group_order;
            p.states = c.states();
            p.states_exact = true;
        }
        None => p.states = solver.num_states() as f64,
    }
    if complete {
        let n = solver.num_states();
        assert!(!p.states_exact || p.states == n as f64, "state count mismatch for {}", p.spec);
        p.states = n as f64;
        p.states_exact = true;
        p.gods_number = Some(solver.depth);
        p.gods_lower = solver.depth;
        p.gods_counting = Some(solver.depth);
        p.mean_distance = layers.iter().enumerate().map(|(d, &x)| (d * x) as f64).sum::<f64>() / n as f64;
        p.mean_distance_exact = true;
        let legal: usize = (0..=solver.depth)
            .into_par_iter()
            .flat_map(|d| solver.layer(d).par_iter())
            .map(|&k| legal_face_turns(bandage, &State::unpack(k), &moves))
            .sum();
        p.branching = legal as f64 / n as f64;
        let far = solver.layer(solver.depth)[0];
        p.antipode = format_alg(&solver.path_to(&State::unpack(far)).unwrap());
        p.antipode_distance = solver.depth;
        let (algs, matching) = scan_algs(&solver, bandaged);
        p.algs = algs;
        if let Some(c) = &count {
            for (i, (n, _)) in possible_effects(&c.chain, bandaged).iter().enumerate() {
                assert_eq!(*n, matching[i], "effect count mismatch for {} ({})", p.spec, ALG_KINDS[i].name());
            }
        }
    } else {
        let probes: Vec<(State, (Option<usize>, usize))> = (0..cfg.samples)
            .into_par_iter()
            .map(|i| {
                let mut rng = Rng::new(cfg.seed ^ (i as u64 + 1).wrapping_mul(0x2545_f491_4f6c_dd1d));
                let s = random_state(bandage, &moves, &mut rng, cfg.walk);
                (s, probe(&solver, s, cfg.ball_nodes))
            })
            .collect();
        p.samples = probes.len();
        p.samples_resolved = probes.iter().filter(|x| x.1.0.is_some()).count();
        p.mean_distance = probes.iter().map(|x| x.1.1 as f64).sum::<f64>() / p.samples as f64;
        p.mean_distance_exact = p.samples_resolved == p.samples;
        p.branching =
            probes.iter().map(|x| legal_face_turns(bandage, &x.0, &moves) as f64).sum::<f64>() / p.samples as f64;
        if p.states_exact {
            p.gods_counting = counting_estimate(&layers, p.states);
        }
        p.gods_lower = probes.iter().map(|x| x.1.1).max().unwrap_or(0).max(solver.depth + 1);
        if let Some((s, (Some(d), _))) = probes.iter().filter(|x| x.1.0.is_some()).max_by_key(|x| x.1.0) {
            p.antipode_distance = *d;
            p.antipode = format!("random state at distance {}: {}", d, s.facelets());
        }
        p.algs = search_algs(&solver, cfg);
    }
    if let Some(c) = &count {
        for (a, (n, cov)) in p.algs.iter_mut().zip(possible_effects(&c.chain, bandaged)) {
            a.possible = n;
            a.coverage = cov;
            if n == 0 {
                a.exact = true; // impossible, so "none" is the exact answer
            }
        }
    }
    p.seconds = t0.elapsed().as_secs_f64();
    p
}

/// Every state of the table (other than solved) with the home shape, possibly with blocks
/// exchanged, as (distance, packed state), nearest first. On a complete table these are all the
/// block-level effects of the puzzle, each with its optimal length.
pub fn home_shape_states<P: Puzzle>(solver: &Solver<P>) -> Vec<(usize, P::Key)> {
    let mut v: Vec<(usize, P::Key)> = (1..=solver.depth)
        .into_par_iter()
        .flat_map(|d| solver.layer(d).par_iter().map(move |&k| (d, k)))
        .filter(|&(_, k)| solver.bandage.home_shape(&solver.bandage.unpack(k)))
        .collect();
    v.par_sort_unstable();
    v
}

/// Literature values for the unbandaged 3x3x3 in HTM (Rokicki, Kociemba, Davidson, Dethridge,
/// 2010): 43,252,003,274,489,856,000 states, God's number 20, mean optimal distance about 17.7.
pub const CUBE333_STATES: f64 = 43_252_003_274_489_856_000.0;
pub const CUBE333_GODS_NUMBER: usize = 20;
pub const CUBE333_MEAN_DISTANCE: f64 = 17.7;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::Cube;

    #[test]
    fn adjacency() {
        // 12 edges x (2 corners + 2 centers)
        assert_eq!(adjacent_pairs().len(), 48);
        let mut rng = Rng::new(7);
        let b = random_bandage(&mut rng, 3);
        assert!(b.bandaged_cubies().count_ones() >= 4);
        assert_eq!(Bandage::parse(&b.spec()).unwrap(), b);
    }

    #[test]
    fn counting() {
        // 333 HTM layer sizes up to 4 moves, extrapolated to 4.3e19 states: about 18
        let est = counting_estimate(&[1, 18, 243, 3240, 43239], CUBE333_STATES).unwrap();
        assert!((17..=19).contains(&est), "{}", est);
    }

    /// Probes on a truncated table must reproduce the exact distances of the full enumeration.
    #[test]
    fn probe_distances_are_exact() {
        let b = Bandage::parse("DFR+DR | UFL+UL+FL | UR+R | DLF+UF+DF+F | BR+B").unwrap();
        let moves = parse_moveset("htm").unwrap();
        let full = Solver::new(b.clone(), moves.clone(), 1000, 50_000_000);
        assert!(full.reached_max_depth && full.depth < 1000);
        let small = Solver::new(b.clone(), moves.clone(), 8, usize::MAX);
        let mut rng = Rng::new(3);
        for _ in 0..30 {
            let s = random_state(&b, &moves, &mut rng, 2000);
            assert_eq!(probe(&small, s, 5_000_000).0, full.distance(&s));
        }
    }

    #[test]
    fn complete_profile() {
        // 792 states, but a God's number of 85
        let b = Bandage::parse("URF+UR | DLF+DBL+DL | DRB+DB+BR | UF+FR+U+F | DFR+DR+D").unwrap();
        let p = profile(&b, &StudyConfig::default());
        assert!(p.complete);
        assert_eq!(p.states, 792.0);
        assert_eq!(p.gods_number, Some(85));
        let b = Bandage::parse("URF+UF | DLF+DF | DRB+DB | DR+D | UL+FL+F+L").unwrap();
        let p = profile(&b, &StudyConfig::default());
        assert!(p.complete);
        assert_eq!(p.states as usize, p.layer_sizes.iter().sum::<usize>());
        assert_eq!(p.gods_number, Some(p.layer_sizes.len() - 1));
        // every reported algorithm really does what it claims
        for (kind, a) in ALG_KINDS.iter().zip(&p.algs) {
            if let Some(l) = a.min_len {
                let mut c = Cube::new(b.clone());
                c.apply(&crate::moves::parse_alg(&a.example).unwrap()).unwrap();
                assert!(c.bandage_restored());
                assert!(kind.matches(&c.state.effect(), false), "{:?} {}", kind, a.example);
                assert_eq!(crate::moves::length(&crate::moves::parse_alg(&a.example).unwrap()), l);
            }
        }
    }
}
