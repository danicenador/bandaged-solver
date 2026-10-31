//! Algorithm search by meet-in-the-middle.
//!
//! The solver enumerates (breadth first, deduplicated) every state reachable from the solved,
//! bandaged cube in at most `depth` legal moves. Two sequences A and C whose end states agree on
//! every piece outside a set P give the algorithm `A C'`, which only affects pieces in P.
//! Pieces are compared through a *key* (the packed state with the free pieces masked out), so
//! candidates are found by grouping equal keys. Every candidate is replayed on the bandaged cube
//! (legality + effect) before it is reported.
//!
//! Optionally the C side is extended past the table by a depth-first stream (`extra_depth`),
//! which finds algorithms up to `2*depth + extra_depth` moves without storing more states.

use crate::cube::{diff_counts, key_mask, Bandage, Cube, Effect, KeyPart, State};
use crate::geometry::{piece_type, PieceType, N_CUBIES};
use crate::moves::{axis_of_base, base_of, invert, length, qtm_length, simplify, Alg, Move, N_BASES};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const NO_MOVE: u8 = u8::MAX;
const NO_PARENT: u32 = u32::MAX;

/// Cheap, well mixing hasher for u128 keys.
#[derive(Default)]
pub(crate) struct MixHasher(u64);

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

impl Hasher for MixHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = mix64(self.0 ^ b as u64);
        }
    }
    fn write_u128(&mut self, x: u128) {
        self.0 = mix64(x as u64 ^ mix64((x >> 64) as u64 ^ self.0));
    }
}

pub(crate) type FastSet = HashSet<u128, BuildHasherDefault<MixHasher>>;
pub(crate) type FastMap<V> = HashMap<u128, V, BuildHasherDefault<MixHasher>>;

/// Number of shards of the build's dedup set.
const SHARDS: usize = 64;

/// Shard of a packed state. Uses a different mix than `MixHasher` so the hashes inside one
/// shard stay well spread.
#[inline]
fn shard_of(key: u128) -> usize {
    (mix64((key >> 64) as u64 ^ mix64(key as u64 ^ 0x9e37_79b9_7f4a_7c15)) >> 58) as usize
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// 3-cycle of edges, nothing else affected.
    Edge3,
    /// 3-cycle of corners, nothing else affected.
    Corner3,
    /// Two edges flipped in place.
    EdgeFlip,
    /// Two corners twisted in place.
    CornerTwist,
    /// Orientation only: up to 2 edges flipped and/or up to 2 corners twisted, all in place.
    Orient,
    /// Any non-trivial effect restricted to the given pieces.
    Any,
}

impl Kind {
    pub fn parse(s: &str) -> Result<Kind, String> {
        Ok(match s.to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "edge3" | "3e" | "edge_3cycle" | "edge_cycle" | "edges" => Kind::Edge3,
            "corner3" | "3c" | "corner_3cycle" | "corner_cycle" | "corners" => Kind::Corner3,
            "edge_flip" | "flip" | "2e" | "edge_flips" => Kind::EdgeFlip,
            "corner_twist" | "twist" | "2c" | "corner_twists" => Kind::CornerTwist,
            "orient" | "orientation" | "rotation" | "2e2c" => Kind::Orient,
            "any" | "custom" => Kind::Any,
            _ => return Err(format!("unknown kind '{}'", s)),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Kind::Edge3 => "edge3",
            Kind::Corner3 => "corner3",
            Kind::EdgeFlip => "edge_flip",
            Kind::CornerTwist => "corner_twist",
            Kind::Orient => "orient",
            Kind::Any => "any",
        }
    }

    fn key_part(&self, t: PieceType) -> KeyPart {
        use KeyPart::*;
        match (self, t) {
            (_, PieceType::Center) => Full,
            (Kind::Edge3, PieceType::Edge) | (Kind::Corner3, PieceType::Corner) => Ignore,
            (Kind::EdgeFlip, PieceType::Edge) | (Kind::CornerTwist, PieceType::Corner) => Position,
            (Kind::Orient, _) => Position,
            (Kind::Any, _) => Ignore,
            _ => Full,
        }
    }

    /// Cheap pre-filter on the number of differing cubies [corners, edges, centers].
    pub(crate) fn diff_ok(&self, d: [u8; 3]) -> bool {
        match self {
            Kind::Edge3 => d == [0, 3, 0],
            Kind::Corner3 => d == [3, 0, 0],
            Kind::EdgeFlip => d == [0, 2, 0],
            Kind::CornerTwist => d == [2, 0, 0],
            Kind::Orient => d[2] == 0 && d[0] <= 2 && d[1] <= 2 && d[0] + d[1] > 0,
            Kind::Any => d != [0, 0, 0],
        }
    }

    pub fn matches(&self, e: &Effect, pure: bool) -> bool {
        let no_centers = e.center_cycles.is_empty();
        match self {
            Kind::Edge3 => {
                no_centers
                    && e.n_corners() == 0
                    && e.flipped.is_empty()
                    && e.edge_cycles.len() == 1
                    && e.edge_cycles[0].len() == 3
                    && (!pure || e.is_pure())
            }
            Kind::Corner3 => {
                no_centers
                    && e.n_edges() == 0
                    && e.twisted.is_empty()
                    && e.corner_cycles.len() == 1
                    && e.corner_cycles[0].len() == 3
                    && (!pure || e.is_pure())
            }
            Kind::EdgeFlip => no_centers && e.n_corners() == 0 && e.edge_cycles.is_empty() && e.flipped.len() == 2,
            Kind::CornerTwist => no_centers && e.n_edges() == 0 && e.corner_cycles.is_empty() && e.twisted.len() == 2,
            Kind::Orient => {
                no_centers
                    && e.edge_cycles.is_empty()
                    && e.corner_cycles.is_empty()
                    && e.flipped.len() <= 2
                    && e.twisted.len() <= 2
                    && !e.is_identity()
            }
            Kind::Any => !e.is_identity(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Query {
    pub kind: Kind,
    /// Restrict the effect to these cubies (bitmask). Required for `Kind::Any`.
    pub pieces: Option<u32>,
    /// For 3-cycles: require every cycled piece to keep its orientation.
    pub pure: bool,
    /// Extra depth streamed on one side beyond the stored table.
    pub extra_depth: usize,
    pub max_length: Option<usize>,
    pub max_results: usize,
    /// Keep only the shortest algorithm for each distinct effect.
    pub unique_effects: bool,
    /// Stop collecting after this many verified algorithms (before sorting).
    pub max_candidates: usize,
    /// Wall-clock budget in seconds for the streamed extension.
    pub time_limit: Option<f64>,
    /// Let bandaged blocks in `pieces` move: an algorithm must bring back the home shape, but a
    /// block may end on the home of a congruent block (block swaps and cycles).
    pub move_blocks: bool,
}

impl Query {
    pub fn new(kind: Kind) -> Query {
        Query {
            kind,
            pieces: None,
            pure: false,
            extra_depth: 0,
            max_length: None,
            max_results: 100,
            unique_effects: false,
            max_candidates: 2_000_000,
            time_limit: None,
            move_blocks: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Found {
    pub alg: Alg,
    pub effect: Effect,
    pub length: usize,
    pub qtm: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    /// Pairs passing the cheap key/diff filter.
    pub candidates: usize,
    /// Candidates rejected because the bandage blocks the combined sequence.
    pub illegal: usize,
    /// Distinct algorithms matching the query (before truncation to max_results).
    pub matched: usize,
    /// True if max_candidates or time_limit stopped the search early.
    pub truncated: bool,
}

/// Breadth-first table of all states within `depth` moves of solved.
pub struct Solver {
    pub bandage: Bandage,
    pub moves: Vec<Move>,
    states: Vec<u128>,
    parent: Vec<u32>,
    last: Vec<u8>,
    /// layer_start[d]..layer_start[d+1] are the states at distance d. len = depth + 2.
    layer_start: Vec<usize>,
    /// Indices into `states`, sorted by state, for lookups.
    sorted: Vec<u32>,
    /// Depth of the last complete layer.
    pub depth: usize,
    /// False if the node limit stopped the enumeration before `max_depth`.
    pub reached_max_depth: bool,
}

#[inline]
fn canonical(last: u8, m: Move) -> bool {
    if last == NO_MOVE {
        return true;
    }
    let (lb, nb) = (base_of(last), base_of(m));
    // same-axis moves commute: only allow them in increasing base order
    !(axis_of_base(lb) == axis_of_base(nb) && nb <= lb)
}

impl Solver {
    pub fn new(bandage: Bandage, moves: Vec<Move>, max_depth: usize, max_nodes: usize) -> Solver {
        let mut s = Solver {
            bandage,
            moves,
            states: vec![State::solved().pack()],
            parent: vec![NO_PARENT],
            last: vec![NO_MOVE],
            layer_start: vec![0, 1],
            sorted: Vec::new(),
            depth: 0,
            reached_max_depth: true,
        };
        s.build(max_depth, max_nodes);
        s
    }

    fn bases(&self) -> Vec<usize> {
        let mut b: Vec<usize> = self.moves.iter().map(|&m| base_of(m)).collect();
        b.dedup();
        b
    }

    /// Legal, canonical successors of a state.
    fn successors(&self, st: &State, last: u8, bases: &[usize], out: &mut Vec<(State, Move)>) {
        let mut allowed = [false; N_BASES];
        for &b in bases {
            allowed[b] = self.bandage.allows(st, b);
        }
        for &m in &self.moves {
            if allowed[base_of(m)] && canonical(last, m) {
                let mut ns = *st;
                ns.apply_move(m);
                out.push((ns, m));
            }
        }
    }

    fn build(&mut self, max_depth: usize, max_nodes: usize) {
        let bases = self.bases();
        // The dedup set is sharded by key so each shard can be filled by its own thread.
        let mut seen: Vec<FastSet> = (0..SHARDS).map(|_| FastSet::default()).collect();
        seen[shard_of(self.states[0])].insert(self.states[0]);
        const BATCH: usize = 1 << 15;
        for d in 1..=max_depth {
            let (start, end) = (self.layer_start[d - 1], self.layer_start[d]);
            let mut overflow = false;
            let mut i = start;
            while i < end && !overflow {
                let j = (i + BATCH).min(end);
                let children: Vec<(u128, u32, u8)> = (i..j)
                    .into_par_iter()
                    .flat_map_iter(|n| {
                        let st = State::unpack(self.states[n]);
                        let mut succ = Vec::with_capacity(self.moves.len());
                        self.successors(&st, self.last[n], &bases, &mut succ);
                        succ.into_iter().map(move |(s, m)| (s.pack(), n as u32, m))
                    })
                    .collect();
                // Bucket child indices by shard, keeping their order, so the first occurrence of
                // a state wins exactly as in a sequential scan (the table stays deterministic).
                let chunk = children.len().div_ceil(rayon::current_num_threads()).max(1);
                let buckets: Vec<Vec<Vec<u32>>> = children
                    .par_chunks(chunk)
                    .enumerate()
                    .map(|(c, ch)| {
                        let mut b = vec![Vec::new(); SHARDS];
                        for (k, e) in ch.iter().enumerate() {
                            b[shard_of(e.0)].push((c * chunk + k) as u32);
                        }
                        b
                    })
                    .collect();
                let mut fresh: Vec<u32> = seen
                    .par_iter_mut()
                    .enumerate()
                    .flat_map_iter(|(sh, set)| {
                        let new: Vec<u32> = buckets
                            .iter()
                            .flat_map(|b| b[sh].iter().copied())
                            .filter(|&k| set.insert(children[k as usize].0))
                            .collect();
                        new
                    })
                    .collect();
                fresh.par_sort_unstable();
                for k in fresh {
                    let (key, p, m) = children[k as usize];
                    self.states.push(key);
                    self.parent.push(p);
                    self.last.push(m);
                }
                if self.states.len() > max_nodes {
                    overflow = true;
                }
                i = j;
            }
            if overflow {
                // keep only complete layers
                self.states.truncate(end);
                self.parent.truncate(end);
                self.last.truncate(end);
                self.reached_max_depth = false;
                break;
            }
            if self.states.len() == end {
                break; // the whole reachable group has been enumerated
            }
            self.layer_start.push(self.states.len());
            self.depth = d;
        }
        drop(seen);
        self.states.shrink_to_fit();
        let mut sorted: Vec<u32> = (0..self.states.len() as u32).collect();
        sorted.par_sort_unstable_by_key(|&i| self.states[i as usize]);
        self.sorted = sorted;
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Number of states at each distance 0..=depth.
    pub fn layer_sizes(&self) -> Vec<usize> {
        self.layer_start.windows(2).map(|w| w[1] - w[0]).collect()
    }

    fn depth_of_index(&self, i: usize) -> usize {
        self.layer_start.partition_point(|&s| s <= i) - 1
    }

    fn find_index(&self, key: u128) -> Option<usize> {
        self.sorted
            .binary_search_by_key(&key, |&i| self.states[i as usize])
            .ok()
            .map(|p| self.sorted[p] as usize)
    }

    /// Distance from solved if the state is in the table (centre spin is ignored).
    pub fn distance(&self, st: &State) -> Option<usize> {
        self.distance_packed(st.pack())
    }

    /// Same as `distance`, for a packed state.
    pub fn distance_packed(&self, key: u128) -> Option<usize> {
        self.find_index(key).map(|i| self.depth_of_index(i))
    }

    /// Packed states at distance `d` (0..=depth).
    pub fn layer(&self, d: usize) -> &[u128] {
        &self.states[self.layer_start[d]..self.layer_start[d + 1]]
    }

    /// An optimal sequence from solved to this state, if it is in the table.
    pub fn path_to(&self, st: &State) -> Option<Alg> {
        self.find_index(st.pack()).map(|i| self.path(i))
    }

    fn path(&self, mut i: usize) -> Alg {
        let mut p = Vec::new();
        while self.parent[i] != NO_PARENT {
            p.push(self.last[i]);
            i = self.parent[i] as usize;
        }
        p.reverse();
        p
    }

    fn key_mask_for(&self, q: &Query) -> Result<u128, String> {
        if q.kind == Kind::Any && q.pieces.is_none() {
            return Err("kind 'any' needs an explicit list of pieces".into());
        }
        // Bandaged cubies are pinned (algorithms must restore the bandage, see Collector::offer)
        // unless blocks may move.
        let pinned = if q.move_blocks { 0 } else { self.bandage.bandaged_cubies() };
        let free_mask = q.pieces.unwrap_or(u32::MAX) & !pinned;
        let mut parts = [KeyPart::Full; N_CUBIES];
        for (c, p) in parts.iter_mut().enumerate() {
            let free = free_mask & (1 << c) != 0;
            *p = if free { q.kind.key_part(piece_type(c)) } else { KeyPart::Full };
        }
        Ok(key_mask(&parts))
    }

    pub fn search(&self, q: &Query) -> Result<(Vec<Found>, SearchStats), String> {
        let mask = self.key_mask_for(q)?;
        let mut keyed: Vec<(u128, u32)> =
            self.states.par_iter().enumerate().map(|(i, &s)| (s & mask, i as u32)).collect();
        keyed.par_sort_unstable();

        let collector = Collector::new(self, q);

        // 1. pairs inside the table
        let mut runs = Vec::new();
        let mut a = 0;
        while a < keyed.len() {
            let mut b = a + 1;
            while b < keyed.len() && keyed[b].0 == keyed[a].0 {
                b += 1;
            }
            if b - a >= 2 {
                runs.push((a, b));
            }
            a = b;
        }
        runs.par_iter().for_each(|&(a, b)| {
            for i in a..b {
                for j in i + 1..b {
                    if collector.full() {
                        return;
                    }
                    let (x, y) = (keyed[i].1 as usize, keyed[j].1 as usize);
                    if q.kind.diff_ok(diff_counts(self.states[x], self.states[y])) {
                        let (pa, pc) = (self.path(x), self.path(y));
                        collector.offer(&pa, &pc);
                        collector.offer(&pc, &pa);
                    }
                }
            }
        });

        // 2. streamed extension of the C side beyond the table
        if q.extra_depth > 0 {
            let index = KeyIndex::new(&keyed);
            let roots = self.layer_start[self.depth]..self.layer_start[self.depth + 1];
            let bases = self.bases();
            roots.into_par_iter().for_each(|r| {
                if collector.full() || collector.out_of_time() {
                    return;
                }
                let root_path = self.path(r);
                let st = State::unpack(self.states[r]);
                let mut path = root_path.clone();
                self.stream(&st, self.last[r], q.extra_depth, &bases, mask, &index, &mut path, &collector);
            });
        }

        Ok(collector.finish(q))
    }

    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        st: &State,
        last: u8,
        remaining: usize,
        bases: &[usize],
        mask: u128,
        index: &KeyIndex,
        path: &mut Alg,
        col: &Collector,
    ) {
        let mut succ = Vec::with_capacity(self.moves.len());
        self.successors(st, last, bases, &mut succ);
        for (ns, m) in succ {
            if col.full() {
                return;
            }
            let packed = ns.pack();
            path.push(m);
            for e in index.group(packed & mask) {
                let x = e.1 as usize;
                if col.query.kind.diff_ok(diff_counts(self.states[x], packed)) {
                    let pa = self.path(x);
                    col.offer(&pa, path);
                    col.offer(path, &pa);
                }
            }
            if remaining > 1 {
                self.stream(&ns, m, remaining - 1, bases, mask, index, path, col);
            }
            path.pop();
        }
    }
}

/// Open-addressing hash index from key to its run in the sorted `keyed` array.
struct KeyIndex<'a> {
    keyed: &'a [(u128, u32)],
    slots: Vec<u32>, // run start + 1, 0 = empty
    mask: usize,
}

impl<'a> KeyIndex<'a> {
    fn new(keyed: &'a [(u128, u32)]) -> Self {
        let starts: Vec<u32> =
            (0..keyed.len()).filter(|&i| i == 0 || keyed[i].0 != keyed[i - 1].0).map(|i| i as u32).collect();
        let cap = (starts.len() * 2).next_power_of_two().max(16);
        let mut idx = KeyIndex { keyed, slots: vec![0; cap], mask: cap - 1 };
        for s in starts {
            let mut h = Self::hash(keyed[s as usize].0) & idx.mask;
            while idx.slots[h] != 0 {
                h = (h + 1) & idx.mask;
            }
            idx.slots[h] = s + 1;
        }
        idx
    }

    #[inline]
    fn hash(k: u128) -> usize {
        mix64(k as u64 ^ mix64((k >> 64) as u64)) as usize
    }

    #[inline]
    fn group(&self, key: u128) -> &'a [(u128, u32)] {
        let mut h = Self::hash(key) & self.mask;
        loop {
            let s = self.slots[h];
            if s == 0 {
                return &[];
            }
            let start = (s - 1) as usize;
            if self.keyed[start].0 == key {
                let len = self.keyed[start..].iter().take_while(|e| e.0 == key).count();
                return &self.keyed[start..start + len];
            }
            h = (h + 1) & self.mask;
        }
    }
}

/// Thread-safe verification and deduplication of candidate algorithms.
struct Collector<'a> {
    solver: &'a Solver,
    query: &'a Query,
    /// Verified algorithms, sharded by hash to keep lock contention low.
    found: Vec<Mutex<HashMap<Alg, Found>>>,
    n_found: AtomicUsize,
    candidates: AtomicUsize,
    illegal: AtomicUsize,
    stop: AtomicBool,
    deadline: Option<Instant>,
}

impl<'a> Collector<'a> {
    fn new(solver: &'a Solver, query: &'a Query) -> Self {
        Collector {
            solver,
            query,
            found: (0..SHARDS).map(|_| Mutex::new(HashMap::new())).collect(),
            n_found: AtomicUsize::new(0),
            candidates: AtomicUsize::new(0),
            illegal: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            deadline: query.time_limit.map(|t| Instant::now() + Duration::from_secs_f64(t)),
        }
    }

    fn out_of_time(&self) -> bool {
        if self.deadline.is_some_and(|d| Instant::now() > d) {
            self.stop.store(true, Ordering::Relaxed);
        }
        self.full()
    }

    fn full(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Candidate algorithm `a c'`.
    fn offer(&self, a: &[Move], c: &[Move]) {
        self.candidates.fetch_add(1, Ordering::Relaxed);
        let mut alg = a.to_vec();
        alg.extend(invert(c));
        let alg = simplify(&alg);
        let len = length(&alg);
        if len == 0 || self.query.max_length.is_some_and(|m| len > m) {
            return;
        }
        let mut cube = Cube::new(self.solver.bandage.clone());
        if cube.apply(&alg).is_err() {
            self.illegal.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let shape_ok = if self.query.move_blocks {
            cube.bandage.shape_at_home(&cube.state)
        } else {
            cube.bandage_restored()
        };
        if !shape_ok {
            return;
        }
        let effect = cube.state.effect();
        if !self.query.kind.matches(&effect, self.query.pure) {
            return;
        }
        if let Some(p) = self.query.pieces
            && effect.affected().iter().any(|&c| p & (1 << c) == 0)
        {
            return;
        }
        let qtm = qtm_length(&alg);
        let mut h = DefaultHasher::new();
        alg.hash(&mut h);
        let mut found = self.found[h.finish() as usize % SHARDS].lock().unwrap();
        if !found.contains_key(&alg) {
            found.insert(alg.clone(), Found { alg, effect, length: len, qtm });
            if self.n_found.fetch_add(1, Ordering::Relaxed) + 1 >= self.query.max_candidates {
                self.stop.store(true, Ordering::Relaxed);
            }
        }
    }

    fn finish(self, q: &Query) -> (Vec<Found>, SearchStats) {
        let found: Vec<Found> =
            self.found.into_iter().flat_map(|m| m.into_inner().unwrap().into_values()).collect();
        let mut stats = SearchStats {
            candidates: self.candidates.into_inner(),
            illegal: self.illegal.into_inner(),
            matched: found.len(),
            truncated: self.stop.into_inner(),
        };
        let mut v: Vec<(String, Found)> =
            found.into_par_iter().map(|f| (crate::moves::format_alg(&f.alg), f)).collect();
        // algs are unique, so the key is a total order and an unstable sort is deterministic
        v.par_sort_unstable_by(|a, b| (a.1.length, a.1.qtm, &a.0).cmp(&(b.1.length, b.1.qtm, &b.0)));
        let mut out: Vec<Found> = v.into_iter().map(|x| x.1).collect();
        if q.unique_effects {
            let descs: Vec<String> = out.par_iter().map(|f| f.effect.describe()).collect();
            let mut seen = HashSet::new();
            let mut keep = descs.into_iter().map(|d| seen.insert(d));
            out.retain(|_| keep.next().unwrap());
            stats.matched = out.len();
        }
        out.truncate(q.max_results);
        (out, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::parse_cubie;
    use crate::moves::{format_alg, parse_moveset};

    fn mask(names: &[&str]) -> u32 {
        names.iter().map(|n| 1u32 << parse_cubie(n).unwrap()).sum()
    }

    #[test]
    fn htm_layer_sizes() {
        let s = Solver::new(Bandage::none(), parse_moveset("htm").unwrap(), 3, 10_000_000);
        assert_eq!(s.layer_sizes(), vec![1, 18, 243, 3240]);
    }

    #[test]
    fn finds_corner_3cycle() {
        let s = Solver::new(Bandage::none(), parse_moveset("htm").unwrap(), 5, 10_000_000);
        let mut q = Query::new(Kind::Corner3);
        q.pieces = Some(mask(&["URF", "UBR", "ULB"]));
        let (found, _) = s.search(&q).unwrap();
        assert!(!found.is_empty());
        assert_eq!(found[0].length, 8); // e.g. [R U R', D]
        for f in &found {
            assert!(Kind::Corner3.matches(&f.effect, false), "{}", format_alg(&f.alg));
        }
    }

    #[test]
    fn finds_slice_edge_3cycle() {
        let s = Solver::new(Bandage::none(), parse_moveset("U M").unwrap(), 4, 10_000_000);
        let mut q = Query::new(Kind::Edge3);
        q.pure = true;
        let (found, _) = s.search(&q).unwrap();
        assert!(!found.is_empty());
        assert_eq!(found[0].length, 4); // [M, U2]
    }

    #[test]
    fn extra_depth_streaming() {
        let s = Solver::new(Bandage::none(), parse_moveset("htm").unwrap(), 3, 10_000_000);
        let q = Query::new(Kind::Corner3);
        assert!(s.search(&q).unwrap().0.is_empty());
        let (found, _) = s.search(&Query { extra_depth: 2, ..q }).unwrap();
        assert_eq!(found[0].length, 8);
    }

    /// Blocks may be exchanged for congruent ones: the 7 corner-edge blocks of this puzzle can be
    /// 3-cycled in 9 moves (the shortest, found by enumerating every home-shape position).
    #[test]
    fn block_3cycle() {
        let spec = "UBR+UR | UFL+UF | ULB+UB | UL+U | DRB+DR | DFR+FR | DLF+FL | DBL+BL | BR+R | DF+F | DL+L | DB+D+B";
        let b = Bandage::parse(spec).unwrap();
        let s = Solver::new(b.clone(), parse_moveset("htm").unwrap(), 5, 10_000_000);
        let q = Query {
            pieces: Some(mask(&["UBR", "UR", "UFL", "UF", "ULB", "UB"])),
            move_blocks: true,
            ..Query::new(Kind::Any)
        };
        let (found, _) = s.search(&q).unwrap();
        assert_eq!(found[0].length, 9);
        let mut c = Cube::new(b.clone());
        c.apply(&found[0].alg).unwrap();
        assert!(!c.bandage_restored() && b.shape_at_home(&c.state));
        assert_eq!(b.describe_blocks(&c.state).matches('+').count(), 3, "{}", b.describe_blocks(&c.state));
        // without move_blocks every block is pinned and nothing exists
        assert!(s.search(&Query { move_blocks: false, ..q }).unwrap().0.is_empty());
    }

    #[test]
    fn bandaged_results_are_legal() {
        let b = Bandage::from_names(&[vec!["DF", "DFR", "F"]]).unwrap();
        let s = Solver::new(b.clone(), parse_moveset("htm").unwrap(), 5, 10_000_000);
        let (found, _) = s.search(&Query::new(Kind::Corner3)).unwrap();
        assert!(!found.is_empty());
        for f in &found {
            let mut c = Cube::new(b.clone());
            c.apply(&f.alg).unwrap();
            assert!(Kind::Corner3.matches(&c.state.effect(), false));
        }
    }
}
