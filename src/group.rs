//! Exact size of the reachable state space of a (bandaged) cube.
//!
//! Whether a turn is legal depends only on where the bandaged cubies are, the *shape*. Every
//! shape is reached by the same number of states: if A and B both reach shape σ, `B A'` is a
//! legal loop through the solved shape. So
//!
//! `#states = #shapes * |G0|`
//!
//! where G0 is the group of the loops that start and end at the solved shape, acting on the 48
//! corner and edge stickers. The shapes are enumerated breadth first, which gives a spanning tree
//! of the shape graph. Closing every non-tree edge with tree paths gives the Schreier generators
//! of G0, and the order of G0 comes from a Schreier-Sims stabilizer chain.

use crate::cube::{key_mask, Bandage, KeyPart, State};
use crate::geometry::{geom, piece_type, sticker_normals, PieceType, EDGE0, N_CUBIES, N_ROT};
use crate::moves::{base_of, Move};
use crate::search::FastMap;
use std::sync::OnceLock;

/// Number of corner and edge stickers.
pub const N_POINTS: usize = 48;

/// A permutation of the 48 stickers: `p[x]` is where the sticker that starts at `x` ends up.
pub type Perm = [u8; N_POINTS];

pub const IDENTITY: Perm = {
    let mut p = [0u8; N_POINTS];
    let mut i = 0;
    while i < N_POINTS {
        p[i] = i as u8;
        i += 1;
    }
    p
};

/// `a ∘ b`: apply `b` first, then `a`.
#[inline]
pub fn compose(a: &Perm, b: &Perm) -> Perm {
    let mut c = [0u8; N_POINTS];
    for x in 0..N_POINTS {
        c[x] = a[b[x] as usize];
    }
    c
}

#[inline]
pub fn inverse(a: &Perm) -> Perm {
    let mut c = [0u8; N_POINTS];
    for x in 0..N_POINTS {
        c[a[x] as usize] = x as u8;
    }
    c
}

/// First sticker of a corner or edge cubie.
pub fn point0(c: usize) -> usize {
    if c < EDGE0 { 3 * c } else { 24 + 2 * (c - EDGE0) }
}

/// image[c][g]: points reached by the stickers of cubie `c` when it has rotation `g`.
fn image_table() -> &'static Vec<[[u8; 3]; N_ROT]> {
    static T: OnceLock<Vec<[[u8; 3]; N_ROT]>> = OnceLock::new();
    T.get_or_init(|| {
        let geo = geom();
        (0..EDGE0 + 12)
            .map(|c| {
                let own = sticker_normals(c);
                let mut t = [[0u8; 3]; N_ROT];
                for (g, row) in t.iter_mut().enumerate() {
                    let s = geo.slot[c][g] as usize;
                    let theirs = sticker_normals(s);
                    for (k, &n) in own.iter().enumerate() {
                        let moved = geo.apply(g as u8, n);
                        row[k] = (point0(s) + theirs.iter().position(|&x| x == moved).unwrap()) as u8;
                    }
                }
                t
            })
            .collect()
    })
}

/// Sticker permutation of a state (centers are ignored: face turns never move them).
pub fn perm_of(s: &State) -> Perm {
    let img = image_table();
    let mut p = [0u8; N_POINTS];
    for c in 0..N_CUBIES {
        let n = match piece_type(c) {
            PieceType::Corner => 3,
            PieceType::Edge => 2,
            PieceType::Center => continue,
        };
        let row = &img[c][s.rot[c] as usize];
        for k in 0..n {
            p[point0(c) + k] = row[k];
        }
    }
    p
}

/// Base and strong generating set of a permutation group (incremental Schreier-Sims).
#[derive(Default)]
pub struct Chain {
    base: Vec<u8>,
    /// Strong generators fixing base[..i], per level i.
    gens: Vec<Vec<Perm>>,
    /// trans[i][x]: an element mapping base[i] to x, for x in the basic orbit.
    trans: Vec<Vec<Option<Perm>>>,
    orbit: Vec<Vec<u8>>,
}

impl Chain {
    pub fn new() -> Chain {
        Chain::default()
    }

    /// Group order: product of the basic orbit lengths.
    pub fn order(&self) -> u128 {
        self.orbit.iter().map(|o| o.len() as u128).product()
    }

    /// Sift `g` from `level` down. Returns the residue and the level where sifting stopped
    /// (`base.len()` if it went through every level).
    fn strip(&self, mut g: Perm, level: usize) -> (Perm, usize) {
        for i in level..self.base.len() {
            match &self.trans[i][g[self.base[i] as usize] as usize] {
                Some(u) => g = compose(&inverse(u), &g),
                None => return (g, i),
            }
        }
        (g, self.base.len())
    }

    /// Strong generators of the whole group.
    pub fn generators(&self) -> &[Perm] {
        self.gens.first().map_or(&[], |g| g.as_slice())
    }

    pub fn contains(&self, g: &Perm) -> bool {
        let (h, _) = self.strip(*g, 0);
        h == IDENTITY
    }

    fn recompute_orbit(&mut self, i: usize) {
        let b = self.base[i];
        let mut trans = vec![None; N_POINTS];
        trans[b as usize] = Some(IDENTITY);
        let mut orbit = vec![b];
        let mut k = 0;
        while k < orbit.len() {
            let x = orbit[k];
            for s in &self.gens[i] {
                let y = s[x as usize];
                if trans[y as usize].is_none() {
                    trans[y as usize] = Some(compose(s, trans[x as usize].as_ref().unwrap()));
                    orbit.push(y);
                }
            }
            k += 1;
        }
        self.trans[i] = trans;
        self.orbit[i] = orbit;
    }

    /// Put the residue `h` (which fixes base[..=last]) on levels from..=last, adding a base
    /// point if it went through every level.
    fn insert(&mut self, h: Perm, from: usize, last: usize) {
        if last == self.base.len() {
            let x = (0..N_POINTS).find(|&x| h[x] as usize != x).unwrap();
            self.base.push(x as u8);
            self.gens.push(Vec::new());
            self.trans.push(Vec::new());
            self.orbit.push(Vec::new());
        }
        for l in from..=last {
            self.gens[l].push(h);
            self.recompute_orbit(l);
        }
    }

    /// Add a generator; returns true if the group grew.
    pub fn add(&mut self, g: &Perm) -> bool {
        let (h, j) = self.strip(*g, 0);
        if h == IDENTITY {
            return false;
        }
        self.insert(h, 0, j);
        // Make the chain complete again: every Schreier generator of every level must sift
        // through the levels below it. A failure adds a generator deeper and restarts there.
        let mut i = j as isize;
        while i >= 0 {
            let l = i as usize;
            let mut jumped = false;
            'level: for k in 0..self.orbit[l].len() {
                let beta = self.orbit[l][k] as usize;
                for gi in 0..self.gens[l].len() {
                    let s = &self.gens[l][gi];
                    let ub = self.trans[l][beta].as_ref().unwrap();
                    let usb = self.trans[l][s[beta] as usize].as_ref().unwrap();
                    let sg = compose(&inverse(usb), &compose(s, ub));
                    if sg == IDENTITY {
                        continue;
                    }
                    let (h, j) = self.strip(sg, l + 1);
                    if h != IDENTITY {
                        self.insert(h, l + 1, j);
                        i = j as isize;
                        jumped = true;
                        break 'level;
                    }
                }
            }
            if !jumped {
                i -= 1;
            }
        }
        true
    }
}

/// Number of shapes and the group G0 of a puzzle.
pub struct StateCount {
    pub shapes: usize,
    pub group_order: u128,
    /// Stabilizer chain of G0: exact membership tests for effects.
    pub chain: Chain,
}

impl StateCount {
    pub fn states(&self) -> f64 {
        self.shapes as f64 * self.group_order as f64
    }
}

/// Shapes and G0 of a puzzle, or None if it has more than `max_shapes` shapes.
pub fn count_states(bandage: &Bandage, moves: &[Move], max_shapes: usize) -> Option<StateCount> {
    let bandaged = bandage.bandaged_cubies();
    let mut parts = [KeyPart::Ignore; N_CUBIES];
    for (c, p) in parts.iter_mut().enumerate() {
        if bandaged & (1 << c) != 0 {
            *p = KeyPart::Full;
        }
    }
    let mask = key_mask(&parts);
    // shape key -> index; reps[i] = the state reached by the BFS tree path to shape i
    let mut index: FastMap<u32> = FastMap::default();
    let mut reps: Vec<State> = vec![State::solved()];
    let mut rep_perm_inv: Vec<Perm> = vec![IDENTITY];
    index.insert(State::solved().pack() & mask, 0);
    let mut chain = Chain::new();
    let mut i = 0;
    while i < reps.len() {
        let s = reps[i];
        let mut allowed = [false; crate::moves::N_BASES];
        for &m in moves {
            allowed[base_of(m)] = bandage.allows(&s, base_of(m));
        }
        for &m in moves {
            if !allowed[base_of(m)] {
                continue;
            }
            let mut t = s;
            t.apply_move(m);
            let key = t.pack() & mask;
            match index.get(&key) {
                Some(&j) => {
                    // loop: tree path to i, move m, tree path to j backwards
                    let g = compose(&rep_perm_inv[j as usize], &perm_of(&t));
                    chain.add(&g);
                }
                None => {
                    if reps.len() >= max_shapes {
                        return None;
                    }
                    index.insert(key, reps.len() as u32);
                    reps.push(t);
                    rep_perm_inv.push(inverse(&perm_of(&t)));
                }
            }
        }
        i += 1;
    }
    Some(StateCount { shapes: reps.len(), group_order: chain.order(), chain })
}

/// Permutation moving each cubie `cyc[k].0` to the slot of the next one, twisted by `cyc[k].1`
/// (a cycle of length 1 twists or flips a piece in place).
pub fn cycle_perm(cyc: &[(usize, u8)]) -> Perm {
    let mut p = IDENTITY;
    for (k, &(c, o)) in cyc.iter().enumerate() {
        let dest = cyc[(k + 1) % cyc.len()].0;
        let n = if c < EDGE0 { 3 } else { 2 };
        for x in 0..n {
            p[point0(c) + x] = (point0(dest) + (x + o as usize) % n) as u8;
        }
    }
    p
}

/// Every effect of `kind` on the cubies of `free`: 3-cycles in both directions with every
/// orientation, flips of 2 edges, twists of 2 corners. Each is the product of `cycle_perm`s.
pub fn candidate_effects(kind: crate::search::Kind, free: u32) -> Vec<Perm> {
    use crate::search::Kind;
    let (lo, hi, n) = match kind {
        Kind::Edge3 | Kind::EdgeFlip => (EDGE0, EDGE0 + 12, 2u8),
        _ => (0, EDGE0, 3u8),
    };
    let pieces: Vec<usize> = (lo..hi).filter(|&c| free & (1 << c) != 0).collect();
    let mut out = Vec::new();
    match kind {
        Kind::Edge3 | Kind::Corner3 => {
            for (i, &a) in pieces.iter().enumerate() {
                for (j, &b) in pieces.iter().enumerate().skip(i + 1) {
                    for &c in &pieces[j + 1..] {
                        for (x, y) in [(b, c), (c, b)] {
                            for o in 0..n * n * n {
                                out.push(cycle_perm(&[(a, o % n), (x, o / n % n), (y, o / n / n)]));
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for (i, &a) in pieces.iter().enumerate() {
                for &b in &pieces[i + 1..] {
                    for oa in 1..n {
                        for ob in 1..n {
                            out.push(compose(&cycle_perm(&[(a, oa)]), &cycle_perm(&[(b, ob)])));
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::{parse_alg, parse_moveset};

    #[test]
    fn perms_follow_states() {
        let a = parse_alg("R U F' L2 D B").unwrap();
        let b = parse_alg("U2 R' B L D'").unwrap();
        let mut s = State::solved();
        s.apply_alg(&a);
        let pa = perm_of(&s);
        let mut t = State::solved();
        t.apply_alg(&b);
        s.apply_alg(&b);
        assert_eq!(perm_of(&s), compose(&perm_of(&t), &pa));
    }

    #[test]
    fn effects_of_the_3x3x3() {
        use crate::search::Kind;
        let htm = parse_moveset("htm").unwrap();
        let g = count_states(&Bandage::none(), &htm, 10).unwrap().chain;
        let count = |k| candidate_effects(k, u32::MAX).iter().filter(|p| g.contains(p)).count();
        // 3-cycles: 2 directions x orientations with a zero sum; flips and twists: zero sum
        assert_eq!(count(Kind::Edge3), 220 * 2 * 4);
        assert_eq!(count(Kind::Corner3), 56 * 2 * 9);
        assert_eq!(count(Kind::EdgeFlip), 66);
        assert_eq!(count(Kind::CornerTwist), 28 * 2);
        // the U perm is one of them, as a sticker permutation
        let mut s = State::solved();
        s.apply_alg(&parse_alg("R2 U R U R' U' R' U' R' U R'").unwrap());
        assert!(candidate_effects(Kind::Edge3, u32::MAX).contains(&perm_of(&s)));
    }

    #[test]
    fn group_orders() {
        let htm = parse_moveset("htm").unwrap();
        let c = count_states(&Bandage::none(), &htm, 10).unwrap();
        assert_eq!(c.shapes, 1);
        assert_eq!(c.group_order, 43_252_003_274_489_856_000);
        // <U, R>: 73,483,200 states
        let ur = parse_moveset("U R").unwrap();
        assert_eq!(count_states(&Bandage::none(), &ur, 10).unwrap().group_order, 73_483_200);
    }

    /// shapes * |G0| equals a plain breadth-first enumeration.
    #[test]
    fn matches_enumeration() {
        let htm = parse_moveset("htm").unwrap();
        for spec in ["URF+UF | DLF+DF | DRB+DB | DR+D | UL+FL+F+L", "DFR+DR | UFL+UL+FL | UR+R | DLF+UF+DF+F | BR+B"] {
            let b = Bandage::parse(spec).unwrap();
            let solver = crate::search::Solver::new(b.clone(), htm.clone(), 1000, 20_000_000);
            assert!(solver.reached_max_depth && solver.depth < 1000, "{}", spec);
            let c = count_states(&b, &htm, 1_000_000).unwrap();
            assert_eq!(c.shapes as u128 * c.group_order, solver.num_states() as u128, "{}", spec);
        }
    }
}
